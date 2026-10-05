#include "runtime/reference.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void expect(const InferenceResult& result, StatusCode expected) {
    require(result.status.code == expected && result.ok() == (expected == StatusCode::Ok), "transformer inference status");
}
void schema_ranges() {
    for (double epsilon : {0.0, -1.0, 1e-100, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        require(validate_schema(OpCode::RMSNORM, {0, 1}, {2}, NormAttrs{epsilon}).code == StatusCode::AttributeMismatch, "invalid RMS epsilon accepted");
    for (double base : {0.0, 1.0, 1e100, std::numeric_limits<double>::quiet_NaN()})
        require(validate_schema(OpCode::ROPE, {0}, {1}, RopeAttrs{0, base, 1088}).code == StatusCode::AttributeMismatch, "invalid RoPE base accepted");
    require(validate_schema(OpCode::ROPE, {0}, {1}, RopeAttrs{-1, 10000, 1088}).code == StatusCode::AttributeMismatch, "negative RoPE position accepted");
    require(validate_schema(OpCode::SOFTMAX, {0}, {1}, SoftmaxAttrs{true, 0, 0, 0}).code == StatusCode::AttributeMismatch, "zero position capacity accepted");
    require(validate_schema(OpCode::ATTENTION, {0, 1}, {2}, AttentionAttrs{}).code == StatusCode::ArityMismatch, "attention arity accepted");
    require(validate_schema(OpCode::ATTENTION, {0, 1, 2}, {3}, AttentionAttrs{0, 16}).code == StatusCode::AttributeMismatch, "zero heads accepted");
    require(validate_schema(OpCode::ATTENTION, {0, 1, 2}, {3}, AttentionAttrs{4, 15}).code == StatusCode::AttributeMismatch, "odd MHA head dimension accepted");
    require(validate_schema(OpCode::ATTENTION, {0, 1, 2}, {3}, AttentionAttrs{std::numeric_limits<std::int64_t>::max(), 16}).code == StatusCode::Overflow, "head geometry overflow accepted");
    for (const OpDesc& desc : {
        OpDesc(OpCode::RMSNORM, {0, 1}, {2}, NormAttrs{}),
        OpDesc(OpCode::SOFTMAX, {0}, {1}, SoftmaxAttrs{true, 5, 2}),
        OpDesc(OpCode::ROPE, {0}, {1}, RopeAttrs{10}),
        OpDesc(OpCode::EMBEDDING, {0, 1}, {2}),
        OpDesc(OpCode::SWIGLU, {0, 1}, {2}),
        OpDesc(OpCode::ATTENTION, {0, 1, 2}, {3}, AttentionAttrs{})}) {
        require(desc.serialize() == desc.serialize() && desc.serialize().find(op_name(desc.code())) != std::string::npos, "transformer deterministic typed serialization");
    }
}
void validators() {
    auto x = Tensor::allocate_cpu({2, 64}), weight = Tensor::allocate_cpu({64}), wrong = Tensor::allocate_cpu({63});
    const auto norm = OpDesc(OpCode::RMSNORM, {0, 1}, {2}, NormAttrs{});
    expect(infer_operator(norm, {x, weight}), StatusCode::Ok);
    expect(infer_operator(norm, {x, wrong}), StatusCode::ShapeMismatch);
    auto scalar = Tensor::allocate_cpu({});
    expect(infer_operator(norm, {scalar, weight}), StatusCode::ShapeMismatch);
    auto scores = Tensor::allocate_cpu({2, 3});
    expect(infer_operator(OpDesc(OpCode::SOFTMAX, {0}, {1}, SoftmaxAttrs{true, 3, 1}), {scores}), StatusCode::Ok);
    expect(infer_operator(OpDesc(OpCode::SOFTMAX, {0}, {1}, SoftmaxAttrs{true, 1087, 0}), {scores}), StatusCode::OutOfRange);
    expect(infer_operator(OpDesc(OpCode::SOFTMAX, {0}, {1}, SoftmaxAttrs{}), {weight}), StatusCode::ShapeMismatch);
    auto q = Tensor::allocate_cpu({2, 4, 16}), k = Tensor::allocate_cpu({3, 4, 16}), v = Tensor::allocate_cpu({3, 4, 16});
    expect(infer_operator(OpDesc(OpCode::ROPE, {0}, {1}, RopeAttrs{1086}), {q}), StatusCode::Ok);
    expect(infer_operator(OpDesc(OpCode::ROPE, {0}, {1}, RopeAttrs{1087}), {q}), StatusCode::OutOfRange);
    auto odd = Tensor::allocate_cpu({2, 4, 15});
    expect(infer_operator(OpDesc(OpCode::ROPE, {0}, {1}, RopeAttrs{}), {odd}), StatusCode::ShapeMismatch);
    auto ids = Tensor::allocate_cpu({2}, DType::INT32), table = Tensor::allocate_cpu({258, 64});
    ids.data<std::int32_t>()[0] = 256; ids.data<std::int32_t>()[1] = 257;
    auto embedding = infer_operator(OpDesc(OpCode::EMBEDDING, {0, 1}, {2}), {ids, table});
    expect(embedding, StatusCode::Ok);
    require(embedding.output->shape == Shape{2, 64} && embedding.output->dtype == DType::FP32, "embedding token ID/output dtype contract");
    expect(infer_operator(OpDesc(OpCode::EMBEDDING, {0, 1}, {2}), {weight, table}), StatusCode::DTypeMismatch);
    auto bad_ids = Tensor::allocate_cpu({1, 2}, DType::INT32);
    expect(infer_operator(OpDesc(OpCode::EMBEDDING, {0, 1}, {2}), {bad_ids, table}), StatusCode::ShapeMismatch);
    // Pure inference must not read token IDs, including invalid runtime values.
    ids.data<std::int32_t>()[0] = -1;
    expect(infer_operator(OpDesc(OpCode::EMBEDDING, {0, 1}, {2}), {ids, table}), StatusCode::Ok);
    auto gate = Tensor::allocate_cpu({2, 128}), up = Tensor::allocate_cpu({2, 128});
    expect(infer_operator(OpDesc(OpCode::SWIGLU, {0, 1}, {2}), {gate, up}), StatusCode::Ok);
    expect(infer_operator(OpDesc(OpCode::SWIGLU, {0, 1}, {2}), {gate, x}), StatusCode::ShapeMismatch);
    const auto attention = OpDesc(OpCode::ATTENTION, {0, 1, 2}, {3}, AttentionAttrs{4, 16, true, 1, 0});
    auto result = infer_operator(attention, {q, k, v});
    expect(result, StatusCode::Ok); require(result.output->shape == q.shape(), "MHA output geometry");
    expect(infer_operator(attention, {odd, k, v}), StatusCode::ShapeMismatch);
    auto mismatch_v = Tensor::allocate_cpu({2, 4, 16});
    expect(infer_operator(attention, {q, k, mismatch_v}), StatusCode::ShapeMismatch);
    auto out = Tensor::allocate_cpu(q.shape());
    require(reference::execute(attention, {q, k, v}, out).code == StatusCode::Unsupported, "attention must not become an opaque reference kernel in S2");
    auto empty_q = Tensor::allocate_cpu({0, 4, 16});
    expect(infer_operator(OpDesc(OpCode::ROPE, {0}, {1}, RopeAttrs{1088}), {empty_q}), StatusCode::Ok);
}
void lowering_and_cache_ranges() {
    auto projection = Tensor::allocate_cpu({2, 64});
    auto q = projection.reshape({2, 4, 16});
    auto cache = Tensor::allocate_cpu({tiny_model::MAX_SEQ, 4, 16});
    auto active = cache.narrow(0, 0, 3);
    const auto before = testing::cpu_allocation_counts();
    auto q_head = infer_operator(OpDesc(OpCode::NARROW, {0}, {1}, SliceAttrs{1, 2, 1}), {q});
    auto k_head = infer_operator(OpDesc(OpCode::NARROW, {0}, {1}, SliceAttrs{1, 2, 1}), {active});
    expect(q_head, StatusCode::Ok); expect(k_head, StatusCode::Ok);
    require(!q_head.output->alias->is_contiguous() && !k_head.output->alias->is_contiguous(), "per-head token-major slices require explicit copies");
    require(testing::cpu_allocation_counts().allocations == before.allocations, "head inference cannot allocate backing buffers");
    // Manually supplied reference-harness buffers model the future explicit lowering,
    // not a graph/model implementation or hidden convenience allocator.
    auto q_dense = Tensor::allocate_cpu({2, 1, 16}), k_dense = Tensor::allocate_cpu({3, 1, 16});
    require(reference::execute(OpDesc(OpCode::MATERIALIZE, {0}, {1}), {*q_head.output->alias}, q_dense).ok(), "Q head explicit materialization");
    require(reference::execute(OpDesc(OpCode::MATERIALIZE, {0}, {1}), {*k_head.output->alias}, k_dense).ok(), "K head explicit materialization");
    auto q2d = q_dense.reshape({2, 16}), k2d = k_dense.reshape({3, 16});
    auto kt = k2d.transpose(0, 1);
    expect(infer_operator(OpDesc(OpCode::MATMUL, {0, 1}, {2}), {q2d, kt}), StatusCode::LayoutMismatch);
    auto kt_dense = Tensor::allocate_cpu({16, 3});
    require(reference::execute(OpDesc(OpCode::MATERIALIZE, {0}, {1}), {kt}, kt_dense).ok(), "K transpose explicit materialization");
    auto score_contract = infer_operator(OpDesc(OpCode::MATMUL, {0, 1}, {2}), {q2d, kt_dense});
    expect(score_contract, StatusCode::Ok); require(score_contract.output->shape == Shape{2, 3}, "ordinary 2D per-head attention MATMUL");
    auto append = cache.narrow(0, 3, 2);
    auto write = infer_operator(OpDesc(OpCode::COPY, {7, 8}, {9}, CopyAttrs{}), {q, append});
    expect(write, StatusCode::Ok);
    require(write.output->kind == OutputKind::Write && write.output->alias->data_offset() == 3 * 4 * 16 * sizeof(float), "persistent cache checked write and distinct state ID");
    expect(infer_operator(OpDesc(OpCode::NARROW, {0}, {1}, SliceAttrs{0, tiny_model::MAX_SEQ - 1, 2}), {cache}), StatusCode::OutOfRange);
}
} // namespace
int main() {
    try {
        require(tiny_model::BATCH == 1 && tiny_model::LAYERS == 2 && tiny_model::HIDDEN == 64 && tiny_model::HEADS == 4 &&
                tiny_model::HEAD_DIM == 16 && tiny_model::FFN == 128 && tiny_model::VOCAB == 258 && tiny_model::MAX_SEQ == 1088 &&
                tiny_model::BOS == 256 && tiny_model::EOS == 257 && tiny_model::PROJECTION_OUTPUT_CHANNEL_AXIS == 1, "frozen tiny-model contract changed");
        schema_ranges(); validators(); lowering_and_cache_ranges();
        require(testing::cpu_allocation_counts().live == 0, "transformer contract tests leaked buffers");
        std::cout << "Transformer schema/inference/positions/head-lowering/cache-write contract: PASS (no transformer kernels)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_transformer_contracts: " << error.what() << '\n'; return 1;
    }
}
