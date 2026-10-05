#include "runtime/shape_inference.hpp"
#include "runtime/copy.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void expect(const InferenceResult& result, StatusCode code) {
    require(result.status.code == code, "inference status mismatch");
    require(result.ok() == (code == StatusCode::Ok), "inference output present iff successful");
    if (code != StatusCode::Ok) require(!result.status.message.empty() && !result.output, "error must not expose output");
}
OpDesc binary(OpCode code) { return OpDesc(code, {0, 1}, {2}, code == OpCode::COPY ? OpAttrs(CopyAttrs{}) : OpAttrs{}); }
void core_inference() {
    auto a = Tensor::allocate_cpu({2, 3});
    auto b = Tensor::allocate_cpu({3, 4});
    auto equal = Tensor::allocate_cpu({2, 3});
    auto matmul = infer_operator(binary(OpCode::MATMUL), {a, b});
    expect(matmul, StatusCode::Ok);
    require(matmul.output->shape == Shape{2, 4} && matmul.output->stride == Stride{4, 1} &&
            matmul.output->kind == OutputKind::NewTensor && !matmul.output->alias, "non-square MATMUL contract");
    for (auto op : {OpCode::ADD, OpCode::MUL}) {
        expect(infer_operator(binary(op), {a, equal}), StatusCode::Ok);
        expect(infer_operator(binary(op), {a, b}), StatusCode::ShapeMismatch);
        expect(infer_operator(binary(op), {a}), StatusCode::ArityMismatch);
    }
    auto scalar = Tensor::allocate_cpu({});
    expect(infer_operator(binary(OpCode::ADD), {scalar, scalar}), StatusCode::Ok);
    expect(infer_operator(binary(OpCode::MATMUL), {scalar, scalar}), StatusCode::ShapeMismatch);
    expect(infer_operator(binary(OpCode::ADD), {a, scalar}), StatusCode::ShapeMismatch); // no broadcasting
    auto ids = Tensor::allocate_cpu(a.shape(), DType::INT32);
    expect(infer_operator(binary(OpCode::ADD), {ids, ids}), StatusCode::DTypeMismatch);
    expect(infer_operator(binary(OpCode::MATMUL), {ids, b}), StatusCode::DTypeMismatch);
    auto strided = Tensor::allocate_cpu({3, 2}).transpose(0, 1);
    expect(infer_operator(binary(OpCode::ADD), {strided, a}), StatusCode::LayoutMismatch);
    expect(infer_operator(binary(OpCode::MATMUL), {strided, b}), StatusCode::LayoutMismatch);
    auto wrong_k = Tensor::allocate_cpu({4, 2});
    expect(infer_operator(binary(OpCode::MATMUL), {a, wrong_k}), StatusCode::ShapeMismatch);
    auto zero_a = Tensor::allocate_cpu({2, 0});
    auto zero_b = Tensor::allocate_cpu({0, 3});
    auto zero = infer_operator(binary(OpCode::MATMUL), {zero_a, zero_b});
    expect(zero, StatusCode::Ok); require(zero.output->shape == Shape{2, 3}, "K=0 output shape");
    auto no_rows = Tensor::allocate_cpu({0, 3});
    expect(infer_operator(binary(OpCode::MATMUL), {no_rows, b}), StatusCode::Ok);
    const auto huge = std::numeric_limits<std::int64_t>::max();
    auto huge_m = Tensor::allocate_cpu({huge, 0});
    auto huge_n = Tensor::allocate_cpu({0, huge});
    expect(infer_operator(binary(OpCode::MATMUL), {huge_m, huge_n}), StatusCode::Overflow);
    auto two_n = Tensor::allocate_cpu({0, 2});
    expect(infer_operator(binary(OpCode::MATMUL), {huge_m, two_n}), StatusCode::Overflow);
    auto gpu_storage = Storage::wrap(Device(DeviceType::CUDA, 0), equal.nbytes(), equal.data<float>(), [](void*) noexcept {});
    Tensor gpu(gpu_storage, equal.dtype(), equal.shape(), equal.stride());
    expect(infer_operator(binary(OpCode::ADD), {a, gpu}), StatusCode::DeviceMismatch);
    auto hinted = infer_operator(OpDesc(OpCode::ADD, {0, 1}, {2}, {}, Device(DeviceType::CUDA)), {a, equal});
    expect(hinted, StatusCode::Ok); require(hinted.output->device == Device{}, "backend hint must not change inference placement");
    auto output = Tensor::allocate_cpu({2, 4});
    require(validate_output_binding(*matmul.output, output).ok(), "caller output validation");
    require(validate_output_binding(*matmul.output, a).code == StatusCode::ShapeMismatch, "wrong output shape accepted");
    auto bad_type = Tensor::allocate_cpu({2, 4}, DType::INT32);
    require(validate_output_binding(*matmul.output, bad_type).code == StatusCode::DTypeMismatch, "wrong output dtype accepted");
    auto noncontig_output = Tensor::allocate_cpu({4, 2}).transpose(0, 1);
    require(validate_output_binding(*matmul.output, noncontig_output).code == StatusCode::LayoutMismatch, "strided arithmetic output accepted");
}
void aliases_and_copy() {
    auto owner = Tensor::allocate_cpu({2, 3, 4});
    const auto before = testing::cpu_allocation_counts();
    for (const OpDesc& desc : {
        OpDesc(OpCode::RESHAPE, {0}, {1}, ReshapeAttrs{Shape{6, 4}}),
        OpDesc(OpCode::VIEW, {0}, {1}, ViewAttrs{Shape{3, 4}, Stride{4, 1}, 48}),
        OpDesc(OpCode::NARROW, {0}, {1}, SliceAttrs{1, 1, 2, 1}),
        OpDesc(OpCode::SLICE, {0}, {1}, SliceAttrs{2, 1, 2, 2}),
        OpDesc(OpCode::TRANSPOSE, {0}, {1}, TransposeAttrs{0, 2}),
        OpDesc(OpCode::PERMUTE, {0}, {1}, PermuteAttrs{{2, 0, 1}})}) {
        auto inferred = infer_operator(desc, {owner});
        expect(inferred, StatusCode::Ok);
        require(inferred.output->kind == OutputKind::Alias && inferred.output->alias->storage() == owner.storage(), "metadata transform must keep Stage 1 storage");
        require(validate_output_binding(*inferred.output, *inferred.output->alias).ok(), "exact inferred alias output");
    }
    require(testing::cpu_allocation_counts().allocations == before.allocations, "inference must allocate zero buffers");
    auto sliced = owner.slice(2, 1, 2, 2);
    auto materialized = infer_operator(OpDesc(OpCode::MATERIALIZE, {0}, {1}), {sliced});
    expect(materialized, StatusCode::Ok);
    require(materialized.output->kind == OutputKind::NewTensor && materialized.output->stride == Stride{6, 2, 1}, "materialization must explicitly request new canonical storage");
    expect(infer_operator(OpDesc(OpCode::RESHAPE, {0}, {1}, ReshapeAttrs{Shape{12}}), {sliced}), StatusCode::LayoutMismatch);
    expect(infer_operator(OpDesc(OpCode::RESHAPE, {0}, {1}, ReshapeAttrs{Shape{25}}), {owner}), StatusCode::ShapeMismatch);
    expect(infer_operator(OpDesc(OpCode::SLICE, {0}, {1}, SliceAttrs{1, 3, 1, 1}), {owner}), StatusCode::OutOfRange);
    expect(infer_operator(OpDesc(OpCode::VIEW, {0}, {1}, ViewAttrs{Shape{2}, Stride{1}, 96}), {owner}), StatusCode::OutOfRange);
    expect(infer_operator(OpDesc(OpCode::VIEW, {0}, {1}, ViewAttrs{Shape{2, 2}, Stride{1, 1}, 0}), {owner}), StatusCode::InvalidArgument);
    expect(infer_operator(OpDesc(OpCode::PERMUTE, {0}, {1}, PermuteAttrs{{0, 1}}), {owner}), StatusCode::ShapeMismatch);
    auto flat = owner.reshape({24});
    auto first = flat.narrow(0, 0, 4), overlap = flat.narrow(0, 2, 4), disjoint = flat.narrow(0, 4, 4);
    expect(infer_operator(binary(OpCode::COPY), {first, overlap}), StatusCode::Aliasing);
    auto copied = infer_operator(binary(OpCode::COPY), {first, disjoint});
    expect(copied, StatusCode::Ok);
    require(copied.output->kind == OutputKind::Write && same_tensor_layout(*copied.output->alias, disjoint), "COPY explicit destination write/version contract");
    require(validate_output_binding(*copied.output, first).code == StatusCode::Aliasing, "COPY cannot redirect its declared destination");
    expect(infer_operator(binary(OpCode::COPY), {first, first}), StatusCode::Ok);
    auto ids = Tensor::allocate_cpu({4}, DType::INT32), id_dest = Tensor::allocate_cpu({4}, DType::INT32);
    expect(infer_operator(binary(OpCode::COPY), {ids, id_dest}), StatusCode::Ok);
    expect(infer_operator(binary(OpCode::COPY), {first, ids}), StatusCode::DTypeMismatch);
    auto gpu_storage = Storage::wrap(Device(DeviceType::CUDA), ids.nbytes(), ids.data<std::int32_t>(), [](void*) noexcept {});
    Tensor gpu(gpu_storage, ids.dtype(), ids.shape(), ids.stride());
    auto cross_device = infer_operator(binary(OpCode::COPY), {ids, gpu});
    expect(cross_device, StatusCode::Ok);
    require(cross_device.output->device == gpu.device(), "explicit cross-device COPY declaration targets destination device");
}
} // namespace
int main() {
    try {
        core_inference(); aliases_and_copy();
        require(testing::cpu_allocation_counts().live == 0, "inference tests leaked buffers");
        std::cout << "Pure core shape/dtype/contiguity/overflow/alias/COPY inference: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_shape_inference: " << error.what() << '\n'; return 1;
    }
}
