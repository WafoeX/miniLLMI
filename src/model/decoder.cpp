#include "model/decoder.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace model {
namespace {
using namespace runtime;

class BlockBuilder {
public:
    BlockBuilder(Graph& graph, GraphCursor& cursor, const DecoderBuildOptions& options)
        : graph_(graph), cursor_(cursor), options_(options) {}

    TensorId input(const std::string& name, const Tensor& tensor) {
        const auto id = cursor_.next_tensor++;
        graph_.add_input(id, name, tensor);
        return id;
    }

    TensorId op(Shape shape, OpCode code, std::vector<TensorId> inputs,
                OpAttrs attrs = {}, bool projection = false, DType dtype = DType::FP32) {
        const auto output = cursor_.next_tensor++;
        graph_.add_tensor(output, std::move(shape), dtype, Device{});
        const std::optional<Device> hint = projection ? std::optional<Device>(options_.projection_device) : std::nullopt;
        graph_.add_node(cursor_.next_node++, OpDesc(code, std::move(inputs), {output}, std::move(attrs), hint));
        return output;
    }

private:
    Graph& graph_;
    GraphCursor& cursor_;
    const DecoderBuildOptions& options_;
};

std::string layer_name(std::int64_t layer, const char* suffix) {
    return "layers." + std::to_string(layer) + "." + suffix;
}
} // namespace

BlockBuildResult build_decoder_block(Graph& graph, const DecoderConfig& config,
                                     const ParameterTable& parameters, std::int64_t layer,
                                     std::int64_t sequence_length, TensorId input,
                                     GraphCursor& cursor, DecoderBuildOptions options) {
    const auto valid = config.validate();
    if (!valid.ok()) throw std::invalid_argument(valid.message);
    const auto& bound = parameters.config();
    if (bound.batch != config.batch || bound.layers != config.layers || bound.hidden != config.hidden ||
        bound.heads != config.heads || bound.kv_heads != config.kv_heads || bound.head_dim != config.head_dim ||
        bound.ffn != config.ffn || bound.vocab != config.vocab || bound.max_seq != config.max_seq ||
        bound.rms_epsilon != config.rms_epsilon || bound.rope_base != config.rope_base || bound.bias != config.bias)
        throw std::invalid_argument("decoder block parameters use a different config");
    if (layer < 0 || layer >= config.layers) throw std::out_of_range("decoder block layer is out of range");
    if (sequence_length <= 0 || sequence_length > config.max_seq)
        throw std::out_of_range("decoder block sequence length is outside (0,max_seq]");
    const auto found = graph.tensors().find(input);
    if (found == graph.tensors().end() || found->second.shape != Shape({sequence_length, config.hidden}) ||
        found->second.dtype != DType::FP32 || found->second.device != Device{} ||
        (found->second.external && !found->second.external->is_contiguous()))
        throw std::invalid_argument("decoder block input must be contiguous logical CPU FP32 [sequence,hidden]");

    BlockBuilder build(graph, cursor, options);
    const auto first_node = cursor.next_node;
    const auto bind = [&](const char* suffix) {
        const auto name = layer_name(layer, suffix);
        return build.input(name, parameters.at(name));
    };
    const auto attn_norm_weight = bind("attn_norm");
    const auto q_weight = bind("q_proj");
    const auto k_weight = bind("k_proj");
    const auto v_weight = bind("v_proj");
    const auto o_weight = bind("o_proj");
    const auto ffn_norm_weight = bind("ffn_norm");
    const auto gate_weight = bind("gate_proj");
    const auto up_weight = bind("up_proj");
    const auto down_weight = bind("down_proj");

    auto scale = Tensor::allocate_cpu({sequence_length, sequence_length});
    const auto scale_value = 1.F / std::sqrt(static_cast<float>(config.head_dim));
    for (std::size_t index = 0; index < scale.numel(); ++index) scale.data<float>()[index] = scale_value;
    const auto scale_id = build.input(layer_name(layer, "attention_scale"), scale);

    const Shape hidden_shape{sequence_length, config.hidden};
    const Shape heads_shape{sequence_length, config.heads, config.head_dim};
    const auto normalized = build.op(hidden_shape, OpCode::RMSNORM, {input, attn_norm_weight}, NormAttrs{config.rms_epsilon});
    const auto q = build.op(hidden_shape, OpCode::MATMUL, {normalized, q_weight}, {}, true);
    const auto k = build.op(hidden_shape, OpCode::MATMUL, {normalized, k_weight}, {}, true);
    const auto v_projection = build.op(hidden_shape, OpCode::MATMUL, {normalized, v_weight}, {}, true);
    // V is sliced before its first CPU primitive. In a mixed graph, copying the
    // full contiguous projection to CPU here prevents a later D2H transfer from
    // targeting a noncontiguous [T,1,D] CUDA alias. The COPY remains explicit
    // and graph-visible; the CPU-only graph needs no redundant boundary copy.
    auto v = v_projection;
    if (options.projection_device.type() == DeviceType::CUDA)
        v = build.op(hidden_shape, OpCode::COPY, {v_projection},
                     CopyAttrs{CopyOverlap::RejectExceptExactSelf, Device{}});
    const auto q_heads = build.op(heads_shape, OpCode::RESHAPE, {q}, ReshapeAttrs{heads_shape});
    const auto k_heads = build.op(heads_shape, OpCode::RESHAPE, {k}, ReshapeAttrs{heads_shape});
    const auto v_heads = build.op(heads_shape, OpCode::RESHAPE, {v}, ReshapeAttrs{heads_shape});
    const auto q_rope = build.op(heads_shape, OpCode::ROPE, {q_heads}, RopeAttrs{0, config.rope_base, config.max_seq});
    const auto k_rope = build.op(heads_shape, OpCode::ROPE, {k_heads}, RopeAttrs{0, config.rope_base, config.max_seq});

    std::vector<TensorId> projected_heads;
    projected_heads.reserve(static_cast<std::size_t>(config.heads));
    for (std::int64_t head = 0; head < config.heads; ++head) {
        const Shape one_head{sequence_length, 1, config.head_dim};
        const Shape matrix{sequence_length, config.head_dim};
        const auto narrow = [&](TensorId source) {
            const auto view = build.op(one_head, OpCode::NARROW, {source}, SliceAttrs{1, head, 1, 1});
            const auto dense = build.op(one_head, OpCode::MATERIALIZE, {view});
            return build.op(matrix, OpCode::RESHAPE, {dense}, ReshapeAttrs{matrix});
        };
        const auto q_head = narrow(q_rope);
        const auto k_head = narrow(k_rope);
        const auto v_head = narrow(v_heads);
        const auto k_transposed = build.op({config.head_dim, sequence_length}, OpCode::TRANSPOSE, {k_head}, TransposeAttrs{0, 1});
        const auto k_matrix = build.op({config.head_dim, sequence_length}, OpCode::MATERIALIZE, {k_transposed});
        const auto scores = build.op({sequence_length, sequence_length}, OpCode::MATMUL, {q_head, k_matrix});
        const auto scaled = build.op({sequence_length, sequence_length}, OpCode::MUL, {scores, scale_id});
        const auto probabilities = build.op({sequence_length, sequence_length}, OpCode::SOFTMAX, {scaled},
                                            SoftmaxAttrs{true, 0, 0, config.max_seq});
        const auto context = build.op(matrix, OpCode::MATMUL, {probabilities, v_head});
        const auto weight_view = build.op({config.head_dim, config.hidden}, OpCode::NARROW, {o_weight},
                                          SliceAttrs{0, head * config.head_dim, config.head_dim, 1});
        const auto weight_dense = build.op({config.head_dim, config.hidden}, OpCode::MATERIALIZE, {weight_view});
        projected_heads.push_back(build.op(hidden_shape, OpCode::MATMUL, {context, weight_dense}, {}, true));
    }
    auto attention = projected_heads.front();
    for (std::size_t head = 1; head < projected_heads.size(); ++head)
        attention = build.op(hidden_shape, OpCode::ADD, {attention, projected_heads[head]});
    const auto attention_residual = build.op(hidden_shape, OpCode::ADD, {input, attention});
    const auto ffn_normalized = build.op(hidden_shape, OpCode::RMSNORM,
                                         {attention_residual, ffn_norm_weight}, NormAttrs{config.rms_epsilon});
    const Shape ffn_shape{sequence_length, config.ffn};
    const auto gate = build.op(ffn_shape, OpCode::MATMUL, {ffn_normalized, gate_weight}, {}, true);
    const auto up = build.op(ffn_shape, OpCode::MATMUL, {ffn_normalized, up_weight}, {}, true);
    const auto activated = build.op(ffn_shape, OpCode::SWIGLU, {gate, up});
    const auto down = build.op(hidden_shape, OpCode::MATMUL, {activated, down_weight}, {}, true);
    const auto output = build.op(hidden_shape, OpCode::ADD, {attention_residual, down});
    return {output, first_node, static_cast<std::size_t>(cursor.next_node - first_node)};
}

DecoderGraph build_decoder_prefill(const DecoderConfig& config, const ParameterTable& parameters,
                                   Tensor token_ids, DecoderBuildOptions options) {
    const auto valid = config.validate();
    if (!valid.ok()) throw std::invalid_argument(valid.message);
    if (token_ids.dtype() != DType::INT32 || token_ids.device() != Device{} ||
        token_ids.shape().rank() != 1 || !token_ids.is_contiguous())
        throw std::invalid_argument("decoder token IDs must be contiguous CPU INT32 [sequence]");
    const auto sequence_length = token_ids.shape()[0];
    if (sequence_length <= 0 || sequence_length > config.max_seq)
        throw std::out_of_range("decoder sequence length is outside (0,max_seq]");

    DecoderGraph result;
    result.sequence_length = sequence_length;
    GraphCursor cursor;
    BlockBuilder build(result.graph, cursor, options);
    const auto ids = build.input("token_ids", token_ids);
    const auto embedding_weight = build.input("token_embedding", parameters.at("token_embedding"));
    auto hidden = build.op({sequence_length, config.hidden}, OpCode::EMBEDDING, {ids, embedding_weight});
    for (std::int64_t layer = 0; layer < config.layers; ++layer) {
        const auto block = build_decoder_block(result.graph, config, parameters, layer,
                                               sequence_length, hidden, cursor, options);
        hidden = block.output;
    }
    const auto final_norm_weight = build.input("final_norm", parameters.at("final_norm"));
    const auto lm_head = build.input("lm_head", parameters.at("lm_head"));
    const auto normalized = build.op({sequence_length, config.hidden}, OpCode::RMSNORM,
                                     {hidden, final_norm_weight}, NormAttrs{config.rms_epsilon});
    result.logits = build.op({sequence_length, config.vocab}, OpCode::MATMUL,
                             {normalized, lm_head}, {}, true);
    result.graph.add_output("logits", result.logits);
    const auto frozen = result.graph.freeze();
    if (!frozen.ok()) throw std::invalid_argument(frozen.message);
    for (const auto& item : result.graph.nodes())
        if (item.second.descriptor.code() == OpCode::MATMUL && item.second.descriptor.backend_hint())
            ++result.projection_nodes;
    return result;
}

} // namespace model
