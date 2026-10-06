#include "runtime/cpu_backend.hpp"
#include "runtime/reference.hpp"
#include <new>
#include <stdexcept>

namespace runtime {
Status CpuBackend::capability(OpCode code, Device requested, DType dtype) const {
    if (requested != device())
        return Status::failure(StatusCode::DeviceMismatch, "CPU backend requires CPU:0; no hidden transfers");
    if (dtype != DType::FP32 && dtype != DType::INT32)
        return Status::failure(StatusCode::DTypeMismatch, "CPU backend supports FP32/INT32 storage only");
    switch (code) {
    case OpCode::ADD: case OpCode::MUL: case OpCode::MATMUL:
        return dtype == DType::FP32 ? Status::success() : Status::failure(StatusCode::DTypeMismatch, "CPU arithmetic requires FP32");
    case OpCode::COPY: case OpCode::MATERIALIZE:
    case OpCode::RESHAPE: case OpCode::VIEW: case OpCode::NARROW: case OpCode::SLICE:
    case OpCode::TRANSPOSE: case OpCode::PERMUTE: return Status::success();
    default: return Status::failure(StatusCode::Unsupported, "operator has no CPU backend implementation");
    }
}
BackendBuffer CpuBackend::allocate(Shape shape, DType dtype, Device requested) const {
    const auto supported = capability(OpCode::MATERIALIZE, requested, dtype);
    if (!supported.ok()) return {supported, std::nullopt};
    try { return {Status::success(), Tensor::allocate_cpu(std::move(shape), dtype)}; }
    catch (const std::bad_alloc&) { return {Status::failure(StatusCode::ResourceExhausted, "CPU backend allocation failed"), std::nullopt}; }
    catch (const std::overflow_error& e) { return {Status::failure(StatusCode::Overflow, e.what()), std::nullopt}; }
    catch (const std::invalid_argument& e) { return {Status::failure(StatusCode::InvalidArgument, e.what()), std::nullopt}; }
}
Status CpuBackend::copy(const Tensor& source, Tensor& destination) const {
    return execute(OpDesc(OpCode::COPY, {0, 1}, {2}, CopyAttrs{}), {source, destination}, destination);
}
BackendPreparation CpuBackend::prepare(const OpDesc& desc, const TensorInputs& inputs, const Tensor& output) const {
    const auto supported = capability(desc.code(), output.device(), output.dtype());
    if (!supported.ok()) return {supported, 0};
    if (desc.backend_hint() && *desc.backend_hint() != device())
        return {Status::failure(StatusCode::DeviceMismatch, "CPU backend rejects a non-CPU placement hint"), 0};
    for (const auto& input : inputs)
        if (input.get().device() != device())
            return {Status::failure(StatusCode::DeviceMismatch, "CPU backend requires explicit CPU inputs"), 0};
    const auto inferred = infer_operator(desc, inputs);
    if (!inferred.ok()) return {inferred.status, 0};
    return {validate_output_binding(*inferred.output, output), 0};
}
Status CpuBackend::execute(const OpDesc& desc, const TensorInputs& inputs, Tensor& output, Workspace workspace) const {
    const auto prepared = prepare(desc, inputs, output);
    if (!prepared.ok()) return prepared.status;
    if (workspace.data || workspace.bytes)
        return Status::failure(StatusCode::InvalidArgument, "CPU backend requires empty workspace");
    const auto inferred = infer_operator(desc, inputs);
    if (inferred.output->kind == OutputKind::Alias) return Status::success();
    return reference::execute(desc, inputs, output);
}
const Backend& default_cpu_backend() { static const CpuBackend backend; return backend; }
} // namespace runtime
