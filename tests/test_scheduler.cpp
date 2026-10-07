#include "runtime/scheduler.hpp"
#include <iostream>
#include <stdexcept>

using namespace runtime;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
// Metadata-only capability double: never pretends to execute GPU work locally.
class MetadataCuda final : public Backend {
public:
    const char* name() const noexcept override { return "metadata-only-cuda"; }
    Device device() const noexcept override { return Device(DeviceType::CUDA); }
    Status capability(OpCode code, Device d, DType dtype) const override {
        if (d == device() && dtype == DType::FP32 && (code == OpCode::MATMUL || code == OpCode::COPY)) return {};
        return Status::failure(StatusCode::Unsupported, "unimplemented metadata capability");
    }
    BackendBuffer allocate(Shape, DType, Device) const override { throw std::logic_error("metadata only"); }
    Status copy(const Tensor&, Tensor&) const override { throw std::logic_error("metadata only"); }
    BackendPreparation prepare(const OpDesc&, const TensorInputs&, const Tensor&) const override { throw std::logic_error("metadata only"); }
    Status execute(const OpDesc&, const TensorInputs&, Tensor&, Workspace) const override { throw std::logic_error("metadata only"); }
};
void placement_tests() {
    const auto& cpu = default_cpu_backend(); MetadataCuda cuda;
    Scheduler scheduler(cpu, &cuda), strict(cpu, &cuda, PlacementFallback::Error), local;
    Layout a{{2, 3}, {3, 1}, DType::FP32, Device{}, 24, 0, 0};
    Layout b{{3, 2}, {2, 1}, DType::FP32, Device{}, 24, 0, 1};
    for (int i = 0; i < 10; ++i) {
        const auto mm = scheduler.place(OpCode::MATMUL, DType::FP32, {a, b}, cuda.device());
        require(mm.ok() && mm.device == cuda.device() && !mm.fell_back, "deterministic CUDA MATMUL placement");
        const auto add = scheduler.place(OpCode::ADD, DType::FP32, {a, a}, cuda.device());
        require(add.ok() && add.device == Device{} && add.fell_back && !add.reason.empty(), "explicit CPU fallback");
    }
    require(!strict.place(OpCode::ADD, DType::FP32, {a, a}, cuda.device()).ok(), "strict unsupported error");
    require(!scheduler.place(OpCode::RMSNORM, DType::FP32, {a, a}, cuda.device()).ok(), "S12 not implemented on either backend");
    require(local.place(OpCode::MATMUL, DType::FP32, {a, b}, cuda.device()).device == Device{}, "CPU-only fallback");
    require(!Scheduler(cpu, nullptr, PlacementFallback::Error).place(OpCode::MATMUL, DType::FP32, {a, b}, cuda.device()).ok(), "missing CUDA strict error");
    require(scheduler.place(OpCode::MATMUL, DType::FP32, {a, b}, Device(DeviceType::CUDA, 1)).fell_back, "unregistered device fallback");
    auto t = a.transpose(0, 1);
    require(scheduler.place(OpCode::COPY, DType::FP32, {t}, cuda.device()).fell_back, "CUDA strided COPY not claimed");
    auto integer = a; integer.dtype = DType::INT32;
    require(!scheduler.place(OpCode::MATMUL, DType::INT32, {integer}, cuda.device()).ok(), "no integer arithmetic");
    require(scheduler.place(OpCode::COPY, DType::INT32, {integer}, cuda.device()).fell_back, "INT32 CUDA unsupported, CPU COPY available");
}
}
int main() {
    try { placement_tests(); std::cout << "scheduler placement: PASS (CPU metadata tests only)\n"; return 0; }
    catch (const std::exception& e) { std::cerr << "test_scheduler: " << e.what() << '\n'; return 1; }
}
