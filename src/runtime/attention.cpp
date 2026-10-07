#include "runtime/attention.hpp"

#include <stdexcept>

namespace runtime {
ExecutionResult execute_causal_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                         const std::vector<Tensor>& scales, Tensor& output,
                                         AttentionAttrs attrs, const Backend& backend) {
    try {
        const auto heads = as_size(attrs.heads);
        if (scales.size() != heads) throw std::invalid_argument("attention requires one declared scale tensor per head");
        if (query.shape().rank() != 3 || key.shape().rank() != 3 || value.shape() != key.shape() || output.shape() != query.shape())
            throw std::invalid_argument("attention Q/K/V/output shape mismatch");
        const auto queries = query.shape()[0], keys = key.shape()[0], dimension = query.shape()[2];
        Graph graph;
        graph.add_input(0, "query", query); graph.add_input(1, "key", key); graph.add_input(2, "value", value);
        graph.add_input(3, "output", output, true);
        TensorId next_tensor = 4; NodeId next_node = 0;
        const auto add = [&](Shape shape, OpDesc descriptor) {
            const auto id = next_tensor++; graph.add_tensor(id, std::move(shape), DType::FP32, query.device());
            graph.add_node(next_node++, std::move(descriptor)); return id;
        };
        for (std::size_t head = 0; head < heads; ++head) {
            const auto scale = next_tensor++;
            graph.add_input(scale, "scale-" + std::to_string(head), scales.at(head));
            const auto head_index = static_cast<std::int64_t>(head);
            const auto q_view = add({queries, 1, dimension}, OpDesc(OpCode::NARROW, {0}, {next_tensor}, SliceAttrs{1, head_index, 1, 1}));
            const auto q_dense = add({queries, 1, dimension}, OpDesc(OpCode::MATERIALIZE, {q_view}, {next_tensor}));
            const auto q_2d = add({queries, dimension}, OpDesc(OpCode::RESHAPE, {q_dense}, {next_tensor}, ReshapeAttrs{Shape{queries, dimension}}));
            const auto k_view = add({keys, 1, dimension}, OpDesc(OpCode::NARROW, {1}, {next_tensor}, SliceAttrs{1, head_index, 1, 1}));
            const auto k_dense = add({keys, 1, dimension}, OpDesc(OpCode::MATERIALIZE, {k_view}, {next_tensor}));
            const auto k_2d = add({keys, dimension}, OpDesc(OpCode::RESHAPE, {k_dense}, {next_tensor}, ReshapeAttrs{Shape{keys, dimension}}));
            const auto k_transposed = add({dimension, keys}, OpDesc(OpCode::TRANSPOSE, {k_2d}, {next_tensor}, TransposeAttrs{0, 1}));
            const auto k_matrix = add({dimension, keys}, OpDesc(OpCode::MATERIALIZE, {k_transposed}, {next_tensor}));
            const auto scores = add({queries, keys}, OpDesc(OpCode::MATMUL, {q_2d, k_matrix}, {next_tensor}));
            const auto scaled = add({queries, keys}, OpDesc(OpCode::MUL, {scores, scale}, {next_tensor}));
            const auto probabilities = add({queries, keys}, OpDesc(OpCode::SOFTMAX, {scaled}, {next_tensor},
                SoftmaxAttrs{attrs.causal, attrs.query_position, attrs.key_position, attrs.max_positions}));
            const auto v_view = add({keys, 1, dimension}, OpDesc(OpCode::NARROW, {2}, {next_tensor}, SliceAttrs{1, head_index, 1, 1}));
            const auto v_dense = add({keys, 1, dimension}, OpDesc(OpCode::MATERIALIZE, {v_view}, {next_tensor}));
            const auto v_2d = add({keys, dimension}, OpDesc(OpCode::RESHAPE, {v_dense}, {next_tensor}, ReshapeAttrs{Shape{keys, dimension}}));
            const auto context = add({queries, dimension}, OpDesc(OpCode::MATMUL, {probabilities, v_2d}, {next_tensor}));
            const auto context_3d = add({queries, 1, dimension}, OpDesc(OpCode::RESHAPE, {context}, {next_tensor}, ReshapeAttrs{Shape{queries, 1, dimension}}));
            const auto destination = add({queries, 1, dimension}, OpDesc(OpCode::NARROW, {3}, {next_tensor}, SliceAttrs{1, head_index, 1, 1}));
            const auto written = add({queries, 1, dimension}, OpDesc(OpCode::COPY, {context_3d, destination}, {next_tensor}, CopyAttrs{}));
            graph.add_output("head-" + std::to_string(head), written);
        }
        const auto frozen = graph.freeze();
        if (!frozen.ok()) return {frozen, std::nullopt, {}, {}};
        return execute_graph(graph, nullptr, nullptr, &backend);
    } catch (const std::exception& error) {
        return {Status::failure(StatusCode::InvalidArgument, error.what()), std::nullopt, {}, {}};
    }
}
} // namespace runtime
