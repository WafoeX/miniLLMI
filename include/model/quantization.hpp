#pragma once

#include "model/config.hpp"

#include <cstddef>

namespace model {

// On-disk and in-memory representation for Stage 16 weight-only INT8. Values
// retain W[in,out] layout; scale[output] is applied along output axis 1.
inline constexpr std::uint32_t INT8_QUANTIZATION_VERSION = 1;

struct QuantizedTensor {
    runtime::Tensor values;
    runtime::Tensor scales;
    std::size_t output_axis = 1;
};

// Only attention/MLP projections and lm_head are eligible. Embedding and norm
// tensors deliberately remain FP32.
bool is_weight_only_int8_eligible(const ParameterSpec& spec);

// W[in,out] -> q[in,out], scale[out]. Finite weights are required; ties round
// to even and q is clamped to [-127,127].
QuantizedTensor quantize_per_output_channel(const runtime::Tensor& weight);

// Validates the Stage 16 representation without allocating a dequant workspace.
void validate_per_output_channel_int8(const QuantizedTensor& quantized);

// Allocates the explicit persistent FP32 prepare_dequant workspace used by the
// unchanged backend MATMUL path.
runtime::Tensor dequantize_per_output_channel(const QuantizedTensor& quantized);

} // namespace model
