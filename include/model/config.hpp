#pragma once

#include "runtime/model_contract.hpp"
#include "runtime/operator.hpp"
#include "runtime/tensor.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace model {

struct DecoderConfig {
    std::int64_t batch = runtime::tiny_model::BATCH;
    std::int64_t layers = runtime::tiny_model::LAYERS;
    std::int64_t hidden = runtime::tiny_model::HIDDEN;
    std::int64_t heads = runtime::tiny_model::HEADS;
    std::int64_t kv_heads = runtime::tiny_model::HEADS;
    std::int64_t head_dim = runtime::tiny_model::HEAD_DIM;
    std::int64_t ffn = runtime::tiny_model::FFN;
    std::int64_t vocab = runtime::tiny_model::VOCAB;
    std::int64_t max_seq = runtime::tiny_model::MAX_SEQ;
    double rms_epsilon = runtime::tiny_model::RMS_EPSILON;
    double rope_base = runtime::tiny_model::ROPE_BASE;
    bool bias = false;

    static DecoderConfig tiny() { return {}; }
    runtime::Status validate() const;
};

struct ParameterSpec {
    std::string name;
    runtime::Shape shape;
};

std::vector<ParameterSpec> parameter_specs(const DecoderConfig& config);
std::size_t parameter_bytes(const DecoderConfig& config);

// Stage 13 binds already-created tensors by canonical name. File parsing remains
// Stage 15 scope. Bindings are immutable, contiguous CPU FP32 graph inputs.
class ParameterTable {
public:
    ParameterTable(DecoderConfig config, std::map<std::string, runtime::Tensor> tensors);

    const DecoderConfig& config() const noexcept { return config_; }
    const runtime::Tensor& at(const std::string& name) const;
    const std::map<std::string, runtime::Tensor>& tensors() const noexcept { return tensors_; }
    std::size_t size() const noexcept { return tensors_.size(); }

private:
    DecoderConfig config_;
    std::map<std::string, runtime::Tensor> tensors_;
};

} // namespace model
