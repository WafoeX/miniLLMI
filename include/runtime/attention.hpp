#pragma once

#include "runtime/backend.hpp"
#include "runtime/graph_executor.hpp"

namespace runtime {
// Builds and runs the declared ordinary-2D attention lowering: per-head
// NARROW/MATERIALIZE/RESHAPE, QK^T, declared same-shaped MUL scale, SOFTMAX,
// PV and checked COPY writes into the caller-owned persistent output ranges.
// `scales` supplies one [Q,K] tensor per head; no hidden broadcasting/allocation.
ExecutionResult execute_causal_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                         const std::vector<Tensor>& scales, Tensor& output,
                                         AttentionAttrs attrs, const Backend& backend = default_cpu_backend());
} // namespace runtime
