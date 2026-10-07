#include "runtime/cpu_backend.hpp"
#include "runtime/reference.hpp"
#include "cpu_dispatch.hpp"
#include "cpu_scalar.hpp"
#include "cpu_optimized.hpp"
#include "runtime/thread_pool.hpp"
#include <new>
#include <stdexcept>

namespace runtime {
CpuBackend::CpuBackend(CpuMatmul matmul, std::size_t workers) : matmul_(matmul) {
    if (matmul != CpuMatmul::ReferenceFP64 && matmul != CpuMatmul::ScalarFP32V0 &&
        matmul != CpuMatmul::LoopIKJFP32C1 && matmul != CpuMatmul::LoopIKJFifoPoolFP32C3)
        throw std::invalid_argument("unknown CPU MATMUL algorithm; no silent fallback");
    if (matmul_ == CpuMatmul::LoopIKJFifoPoolFP32C3) pool_ = std::make_unique<ThreadPool>(workers);
    else if (workers != 1) throw std::invalid_argument("worker count is only valid for the FIFO pool algorithm");
}
CpuBackend::~CpuBackend() = default;
std::size_t CpuBackend::workers() const noexcept { return pool_ ? pool_->worker_count() : 1; }
const char* CpuBackend::name() const noexcept {
    switch (matmul_) {
    case CpuMatmul::ReferenceFP64: return "cpu-reference-fp64";
    case CpuMatmul::ScalarFP32V0: return "cpu-ijk-fp32-v0";
    case CpuMatmul::LoopIKJFP32C1: return "cpu-ikj-fp32-c1";
    case CpuMatmul::LoopIKJFifoPoolFP32C3: return "cpu-ikj-fp32-c3-fifo";
    }
    return "cpu-invalid";
}
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
    case OpCode::RMSNORM: case OpCode::SOFTMAX: case OpCode::ROPE: case OpCode::EMBEDDING: case OpCode::SWIGLU:
        return dtype == DType::FP32 ? Status::success() : Status::failure(StatusCode::DTypeMismatch, "CPU transformer arithmetic requires FP32");
    // S12 primitives are enabled independently. ATTENTION remains a graph
    // composition rather than an opaque backend kernel.
    case OpCode::ATTENTION:
        return Status::failure(StatusCode::Unsupported, std::string(op_name(code)) + " CPU primitive is not implemented");
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
    OutputKind kind{};
    return {detail::validate_cpu_core(desc, inputs, output, kind), 0};
}
Status CpuBackend::execute(const OpDesc& desc, const TensorInputs& inputs, Tensor& output, Workspace workspace) const {
    const auto prepared = prepare(desc, inputs, output);
    if (!prepared.ok()) return prepared.status;
    if (workspace.data || workspace.bytes)
        return Status::failure(StatusCode::InvalidArgument, "CPU backend requires empty workspace");
    const auto inferred = infer_operator(desc, inputs);
    if (inferred.output->kind == OutputKind::Alias) return Status::success();
    if (matmul_ == CpuMatmul::ReferenceFP64) return reference::execute(desc, inputs, output);
    const auto kernel = matmul_ == CpuMatmul::ScalarFP32V0 ? detail::matmul_ijk_fp32_v0
        : matmul_ == CpuMatmul::LoopIKJFP32C1 ? detail::matmul_ikj_fp32_c1
        : detail::matmul_ikj_fifo_pool_fp32_c3;
    return detail::execute_cpu_core(desc, inputs, output, kernel, pool_.get());
}
const Backend& default_cpu_backend() { static const CpuBackend backend; return backend; }
} // namespace runtime
