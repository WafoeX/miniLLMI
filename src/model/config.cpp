#include "model/config.hpp"

#include <cmath>
#include <stdexcept>

namespace model {
namespace {
runtime::Status invalid(const std::string& message) {
    return runtime::Status::failure(runtime::StatusCode::InvalidArgument, "decoder config: " + message);
}
}

runtime::Status DecoderConfig::validate() const {
    if (batch != 1) return invalid("only batch=1 is supported");
    if (bias) return invalid("bias parameters are unsupported");
    if (layers <= 0 || hidden <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        ffn <= 0 || vocab <= 0 || max_seq <= 0)
        return invalid("all dimensions must be positive");
    if (kv_heads != heads) return invalid("GQA/MQA is unsupported; kv_heads must equal heads");
    if (hidden != heads * head_dim) return invalid("hidden must equal heads*head_dim");
    if (head_dim % 2 != 0) return invalid("rotary head_dim must be even");
    if (!std::isfinite(rms_epsilon) || rms_epsilon <= 0) return invalid("RMS epsilon must be finite and positive");
    if (!std::isfinite(rope_base) || rope_base <= 1) return invalid("RoPE base must be finite and greater than one");
    return runtime::Status::success();
}

std::vector<ParameterSpec> parameter_specs(const DecoderConfig& config) {
    const auto status = config.validate();
    if (!status.ok()) throw std::invalid_argument(status.message);
    std::vector<ParameterSpec> specs;
    specs.push_back({"token_embedding", {config.vocab, config.hidden}});
    for (std::int64_t layer = 0; layer < config.layers; ++layer) {
        const auto prefix = "layers." + std::to_string(layer) + ".";
        specs.push_back({prefix + "attn_norm", {config.hidden}});
        specs.push_back({prefix + "q_proj", {config.hidden, config.hidden}});
        specs.push_back({prefix + "k_proj", {config.hidden, config.hidden}});
        specs.push_back({prefix + "v_proj", {config.hidden, config.hidden}});
        specs.push_back({prefix + "o_proj", {config.hidden, config.hidden}});
        specs.push_back({prefix + "ffn_norm", {config.hidden}});
        specs.push_back({prefix + "gate_proj", {config.hidden, config.ffn}});
        specs.push_back({prefix + "up_proj", {config.hidden, config.ffn}});
        specs.push_back({prefix + "down_proj", {config.ffn, config.hidden}});
    }
    specs.push_back({"final_norm", {config.hidden}});
    specs.push_back({"lm_head", {config.hidden, config.vocab}});
    return specs;
}

std::size_t parameter_bytes(const DecoderConfig& config) {
    std::size_t bytes = 0;
    for (const auto& spec : parameter_specs(config))
        bytes = runtime::checked_add(bytes, runtime::nbytes(spec.shape, runtime::DType::FP32));
    return bytes;
}

ParameterTable::ParameterTable(DecoderConfig config, std::map<std::string, runtime::Tensor> tensors)
    : config_(std::move(config)), tensors_(std::move(tensors)) {
    const auto status = config_.validate();
    if (!status.ok()) throw std::invalid_argument(status.message);
    const auto specs = parameter_specs(config_);
    if (tensors_.size() != specs.size())
        throw std::invalid_argument("decoder parameters: expected " + std::to_string(specs.size()) +
                                    " named tensors, got " + std::to_string(tensors_.size()));
    for (const auto& spec : specs) {
        const auto found = tensors_.find(spec.name);
        if (found == tensors_.end()) throw std::invalid_argument("decoder parameters: missing '" + spec.name + "'");
        const auto& tensor = found->second;
        if (tensor.shape() != spec.shape) throw std::invalid_argument("decoder parameters: shape mismatch for '" + spec.name + "'");
        if (tensor.dtype() != runtime::DType::FP32) throw std::invalid_argument("decoder parameters: FP32 required for '" + spec.name + "'");
        if (tensor.device() != runtime::Device{}) throw std::invalid_argument("decoder parameters: CPU binding required for '" + spec.name + "'");
        if (!tensor.is_contiguous()) throw std::invalid_argument("decoder parameters: contiguous binding required for '" + spec.name + "'");
    }
}

const runtime::Tensor& ParameterTable::at(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) throw std::out_of_range("unknown decoder parameter: " + name);
    return found->second;
}

} // namespace model
