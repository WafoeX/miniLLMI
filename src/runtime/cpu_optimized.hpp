#pragma once
#include "runtime/operator.hpp"
#include "runtime/tensor.hpp"
namespace runtime { class ThreadPool; }

namespace runtime::detail {
// S7-C1 candidate: single-thread FP32 i->k->j loop order.  This is separate
// from the immutable S6 ijk baseline and uses the same caller-owned bindings.
Status matmul_ikj_fp32_c1(const Tensor& first, const Tensor& second, Tensor& output, ThreadPool*);
Status matmul_ikj_fifo_pool_fp32_c3(const Tensor& first, const Tensor& second, Tensor& output, ThreadPool*);
} // namespace runtime::detail
