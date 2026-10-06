#pragma once
#include "runtime/shape_inference.hpp"
namespace runtime::detail {
// Frozen S6 v0: single-thread FP32 ijk; no explicit SIMD, tiling or pool.
// Bindings/finite inputs are validated by the shared CPU dispatch before entry.
Status matmul_ijk_fp32_v0(const Tensor& first, const Tensor& second, Tensor& output);
} // namespace runtime::detail
