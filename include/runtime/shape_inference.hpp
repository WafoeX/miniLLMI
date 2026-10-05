#pragma once

#include "runtime/operator.hpp"
#include "runtime/layout.hpp"
#include "runtime/tensor.hpp"
#include <functional>
#include <optional>
#include <vector>

namespace runtime {
using TensorInputs = std::vector<std::reference_wrapper<const Tensor>>;
enum class OutputKind { NewTensor, Alias, Write };
// A declarative output requirement, NOT a second executable tensor/storage.
// NewTensor describes planned caller allocation. Alias/Write carry the existing
// Stage 1 Tensor handle so the source storage is retained without a new buffer.
struct OutputContract {
    Shape shape;
    Stride stride;
    DType dtype;
    Device device;
    OutputKind kind;
    std::optional<Tensor> alias;
};
struct InferenceResult {
    Status status;
    std::optional<OutputContract> output;
    bool ok() const noexcept { return status.ok() && output.has_value(); }
};
// Pure metadata; inputs are in descriptor ID order. Graph resolution is S3.
// Does not read values, allocate buffers, execute kernels or obey backend hints.
InferenceResult infer_operator(const OpDesc& descriptor, const TensorInputs& inputs);
Status validate_output_binding(const OutputContract& contract, const Tensor& output);

using LayoutInputs = std::vector<std::reference_wrapper<const Layout>>;
struct LayoutOutputContract { Layout layout; OutputKind kind; };
struct LayoutInferenceResult {
    Status status;
    std::optional<LayoutOutputContract> output;
    bool ok() const noexcept { return status.ok() && output.has_value(); }
};
// Same semantic implementation as infer_operator; no backing buffers required.
// storage_key must identify physical roots consistently within the caller.
LayoutInferenceResult infer_layout(const OpDesc& descriptor, const LayoutInputs& inputs);
} // namespace runtime
