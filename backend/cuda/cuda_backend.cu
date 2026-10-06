#include "runtime/cuda_backend.hpp"
#include "stage0/gemm_cuda.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <climits>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace runtime {
namespace {
Status failure(const std::exception& error) { return Status::failure(StatusCode::InvalidArgument, error.what()); }
void require_contiguous(const Tensor& tensor) {
    if (!tensor.is_contiguous()) throw std::invalid_argument("CUDA backend requires contiguous tensors; insert MATERIALIZE explicitly");
}
const void* pointer(const Tensor& tensor) { return static_cast<const char*>(tensor.storage()->data()) + tensor.data_offset(); }
void* pointer(Tensor& tensor) { return static_cast<char*>(tensor.storage()->data()) + tensor.data_offset(); }
void check_binding(const Tensor& tensor, Device device, DType dtype) {
    if (tensor.device() != device) throw std::invalid_argument("CUDA binding has wrong device");
    if (tensor.dtype() != dtype) throw std::invalid_argument("CUDA binding has wrong dtype");
    require_contiguous(tensor);
}
MemoryPlan checked_cuda_plan(const Graph& graph, MemoryPlan plan) {
    const auto valid = validate_memory_plan(graph, plan);
    if (!valid.ok()) throw std::invalid_argument(valid.message);
    for (const auto& item : graph.tensors())
        if (item.second.device.type() != DeviceType::CUDA || item.second.dtype != DType::FP32)
            throw std::invalid_argument("CUDA planned execution requires CUDA FP32 graph tensors");
    if (plan.capacity_bytes % sizeof(float) != 0) throw std::invalid_argument("CUDA plan is not FP32-addressable");
    return plan;
}
} // namespace

struct CudaBackend::State {
    Device device;
    CudaMatmul matmul;
    cudaStream_t stream{};
    cublasHandle_t blas{};
    State(Device requested, CudaMatmul selected) : device(requested), matmul(selected) {
        CUDA_CHECK(cudaSetDevice(device.index()));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            CUBLAS_CHECK(cublasCreate(&blas));
            CUBLAS_CHECK(cublasSetStream(blas, stream));
            CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
            CUBLAS_CHECK(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
        } catch (...) { cudaStreamDestroy(stream); throw; }
    }
    ~State() {
        if (stream) cudaStreamSynchronize(stream);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
};

CudaBackend::CudaBackend(int index, CudaMatmul matmul)
    : state_(std::make_unique<State>(Device(DeviceType::CUDA, index), matmul)) {}
CudaBackend::~CudaBackend() = default;
const char* CudaBackend::name() const noexcept { return state_->matmul == CudaMatmul::Stage0Naive ? "cuda-stage0-naive" : "cuda-cublas"; }
Device CudaBackend::device() const noexcept { return state_->device; }
Status CudaBackend::capability(OpCode code, Device requested, DType dtype) const {
    if (requested != device()) return Status::failure(StatusCode::DeviceMismatch, "CUDA backend requires its configured CUDA device");
    if (dtype != DType::FP32) return Status::failure(StatusCode::DTypeMismatch, "CUDA backend currently supports FP32 only");
    if (code == OpCode::COPY || code == OpCode::MATMUL) return Status::success();
    return Status::failure(StatusCode::Unsupported, "CUDA operator is not implemented in Stage 8");
}
BackendBuffer CudaBackend::allocate(Shape shape, DType dtype, Device requested) const {
    const auto supported = capability(OpCode::COPY, requested, dtype);
    if (!supported.ok()) return {supported, std::nullopt};
    try {
        const auto bytes = nbytes(shape, dtype); void* allocation = nullptr;
        CUDA_CHECK(cudaSetDevice(device().index())); if (bytes) CUDA_CHECK(cudaMalloc(&allocation, bytes));
        auto storage = Storage::wrap(device(), bytes, allocation, [index = device().index()](void* data) noexcept {
            if (data) { cudaSetDevice(index); cudaFree(data); }
        });
        return {Status::success(), Tensor(std::move(storage), dtype, shape, contiguous_stride(shape))};
    } catch (const std::bad_alloc&) { return {Status::failure(StatusCode::ResourceExhausted, "CUDA storage allocation failed"), std::nullopt}; }
    catch (const std::exception& error) { return {failure(error), std::nullopt}; }
}
Status CudaBackend::copy(const Tensor& source, Tensor& destination) const {
    try {
        require_contiguous(source); require_contiguous(destination);
        if (source.dtype() != destination.dtype() || source.shape() != destination.shape()) return Status::failure(StatusCode::ShapeMismatch, "CUDA COPY requires equal contiguous shape and dtype");
        const auto bytes = source.nbytes(); if (!bytes) return Status::success();
        const auto sd = source.device().type(), dd = destination.device().type();
        if (sd == DeviceType::CUDA && source.device() != device()) return Status::failure(StatusCode::DeviceMismatch, "CUDA COPY source uses another device");
        if (dd == DeviceType::CUDA && destination.device() != device()) return Status::failure(StatusCode::DeviceMismatch, "CUDA COPY destination uses another device");
        if (sd == DeviceType::CUDA && dd == DeviceType::CUDA && source.storage() == destination.storage()) return Status::failure(StatusCode::Aliasing, "CUDA D2D COPY rejects aliased storage");
        const auto kind = sd == DeviceType::CPU ? cudaMemcpyHostToDevice : dd == DeviceType::CPU ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
        CUDA_CHECK(cudaMemcpyAsync(pointer(destination), pointer(source), bytes, kind, state_->stream)); CUDA_CHECK(cudaStreamSynchronize(state_->stream));
        return Status::success();
    } catch (const std::exception& error) { return failure(error); }
}
BackendPreparation CudaBackend::prepare(const OpDesc& desc, const TensorInputs& inputs, const Tensor& output) const {
    const auto supported = capability(desc.code(), output.device(), output.dtype());
    if (!supported.ok()) return {supported, 0};
    if (desc.backend_hint() && *desc.backend_hint() != device()) return {Status::failure(StatusCode::DeviceMismatch, "CUDA backend rejects a different placement hint"), 0};
    try {
        check_binding(output, device(), DType::FP32); for (const auto& input : inputs) check_binding(input.get(), device(), DType::FP32);
        const auto inferred = infer_operator(desc, inputs); if (!inferred.ok()) return {inferred.status, 0};
        if (inferred.output->shape != output.shape()) return {Status::failure(StatusCode::ShapeMismatch, "CUDA output shape differs from inferred shape"), 0};
        if (desc.code() == OpCode::MATMUL && (inputs.size() != 2 || output.shape().rank() != 2)) return {Status::failure(StatusCode::ShapeMismatch, "CUDA MATMUL supports rank-2 operands"), 0};
        return {Status::success(), 0};
    } catch (const std::exception& error) { return {failure(error), 0}; }
}
Status CudaBackend::execute(const OpDesc& desc, const TensorInputs& inputs, Tensor& output, Workspace workspace) const {
    if (workspace.data || workspace.bytes) return Status::failure(StatusCode::InvalidArgument, "CUDA Stage 8 requires empty workspace");
    const auto prepared = prepare(desc, inputs, output); if (!prepared.ok()) return prepared.status;
    try {
        if (desc.code() == OpCode::COPY) return copy(inputs[0].get(), output);
        const auto& left = inputs[0].get(); const auto& right = inputs[1].get();
        const auto m = as_size(left.shape()[0]), k = as_size(left.shape()[1]), n = as_size(right.shape()[1]);
        if (m > static_cast<std::size_t>(INT_MAX) || n > static_cast<std::size_t>(INT_MAX) || k > static_cast<std::size_t>(INT_MAX)) return Status::failure(StatusCode::Overflow, "CUDA Stage 0 adapter shape exceeds int range");
        const stage0::Shape shape{static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)};
        auto* a = static_cast<const float*>(pointer(left)); auto* b = static_cast<const float*>(pointer(right)); auto* c = static_cast<float*>(pointer(output));
        if (state_->matmul == CudaMatmul::Stage0Naive) stage0::launch_naive(a, b, c, shape, state_->stream); else stage0::launch_cublas(state_->blas, a, b, c, shape);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(state_->stream)); return Status::success();
    } catch (const std::exception& error) { return failure(error); }
}

CudaPlannedAllocationProvider::CudaPlannedAllocationProvider(const Graph& graph, const CudaBackend& backend, PlanPolicy policy)
    : backend_(backend), plan_(checked_cuda_plan(graph, plan_memory(graph, policy, 64))) {
    auto backing = backend_.allocate({static_cast<std::int64_t>(plan_.capacity_bytes / sizeof(float))}, DType::FP32, backend_.device());
    if (!backing.ok()) throw std::runtime_error(backing.status.message);
    storage_ = backing.tensor->storage();
    for (const auto& item : plan_.slots) bindings_.emplace(item.first, Binding{graph.tensors().at(item.first).shape, graph.tensors().at(item.first).dtype, false});
}
Status CudaPlannedAllocationProvider::validate_graph(const Graph& graph) const {
    try { (void)checked_cuda_plan(graph, plan_); return Status::success(); }
    catch (const std::exception& error) { return failure(error); }
}
Status CudaPlannedAllocationProvider::begin() {
    if (running_ || storage_.use_count() != 1) return Status::failure(StatusCode::InvalidArgument, "CUDA planned context busy: release all returned storage aliases");
    for (auto& item : bindings_) item.second.active = false; running_ = true; return Status::success();
}
Tensor CudaPlannedAllocationProvider::allocate(TensorId root, Shape shape, DType dtype) {
    const auto found = bindings_.find(root);
    if (!running_ || found == bindings_.end() || found->second.active || found->second.shape != shape || found->second.dtype != dtype) throw std::invalid_argument("invalid CUDA planned allocation binding; explicitly prepare again");
    found->second.active = true; return Tensor(storage_, dtype, std::move(shape), contiguous_stride(shape), plan_.slots.at(root).offset);
}
void CudaPlannedAllocationProvider::release(TensorId root) noexcept { const auto found = bindings_.find(root); if (found != bindings_.end()) found->second.active = false; }
} // namespace runtime
