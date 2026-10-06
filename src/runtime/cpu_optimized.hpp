#pragma once
#include "runtime/shape_inference.hpp"

namespace runtime::detail {
// S7-C1 candidate: single-thread FP32 i->k->j loop order.  This is separate
// from the immutable S6 ijk baseline and uses the same caller-owned bindings.
Status matmul_ikj_fp32_c1(const Tensor& first, const Tensor& second, Tensor& output);
} // namespace runtime::detail
