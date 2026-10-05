#include "runtime/operator.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace runtime;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void code(Status status, StatusCode expected) {
    require(status.code == expected, "unexpected structured status code");
    require(status.ok() || !status.message.empty(), "failure diagnostic missing");
}
template<class Exception, class F> void throws(F&& fn) {
    try { fn(); } catch (const Exception&) { return; }
    throw std::runtime_error("expected exception");
}
void schema_tests() {
    const OpDesc add(OpCode::ADD, {0, 1}, {2});
    require(add.serialize() == "{\"version\":1,\"op\":\"ADD\",\"inputs\":[0,1],\"outputs\":[2],\"attributes\":{},\"backend_hint\":null}", "canonical descriptor JSON");
    require(add.serialize() == add.serialize() && add.inputs() == std::vector<TensorId>{0, 1}, "immutable/deterministic descriptor");
    const OpDesc hinted(OpCode::MATMUL, {0, 1}, {2}, {}, Device(DeviceType::CUDA, 2));
    require(hinted.serialize().find("\"backend_hint\":\"cuda:2\"") != std::string::npos, "advisory device hint serialization");
    code(validate_schema(OpCode::ADD, {0}, {1}, {}), StatusCode::ArityMismatch);
    code(validate_schema(OpCode::ADD, {0, 1}, {}, {}), StatusCode::ArityMismatch);
    code(validate_schema(OpCode::ADD, {0, 1}, {2, 3}, {}), StatusCode::ArityMismatch);
    code(validate_schema(OpCode::ADD, {INVALID_TENSOR_ID, 1}, {2}, {}), StatusCode::InvalidArgument);
    code(validate_schema(OpCode::ADD, {0, 1}, {INVALID_TENSOR_ID}, {}), StatusCode::InvalidArgument);
    code(validate_schema(OpCode::ADD, {0, 1}, {0}, {}), StatusCode::InvalidArgument);
    code(validate_schema(OpCode::ADD, {0, 0}, {1}, {}), StatusCode::Ok); // repeated reads are legal
    code(validate_schema(static_cast<OpCode>(999), {0}, {1}, {}), StatusCode::InvalidArgument);
    code(validate_schema(OpCode::MUL, {0, 1}, {2}, CopyAttrs{}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::COPY, {0, 1}, {2}, {}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::COPY, {0, 1}, {2}, CopyAttrs{static_cast<CopyOverlap>(99)}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::SLICE, {0}, {1}, SliceAttrs{0, 0, 2, 0}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::SLICE, {0}, {1}, SliceAttrs{MAX_RANK, 0, 2, 1}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::SLICE, {0}, {1}, SliceAttrs{0, -1, 2, 1}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::NARROW, {0}, {1}, SliceAttrs{0, 0, 2, 2}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::VIEW, {0}, {1}, ViewAttrs{Shape{2}, Stride{}, 0}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::VIEW, {0}, {1}, ViewAttrs{Shape{2}, Stride{0}, 0}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::PERMUTE, {0}, {1}, PermuteAttrs{{0, 0}}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::PERMUTE, {0}, {1}, PermuteAttrs{{0, 2}}), StatusCode::AttributeMismatch);
    code(validate_schema(OpCode::TRANSPOSE, {0}, {1}, TransposeAttrs{0, MAX_RANK}), StatusCode::AttributeMismatch);
    for (const OpDesc& op : {
        OpDesc(OpCode::COPY, {0, 1}, {2}, CopyAttrs{}),
        OpDesc(OpCode::MATERIALIZE, {0}, {1}),
        OpDesc(OpCode::RESHAPE, {0}, {1}, ReshapeAttrs{Shape{2, 3}}),
        OpDesc(OpCode::VIEW, {0}, {1}, ViewAttrs{Shape{3}, Stride{2}, 4}),
        OpDesc(OpCode::NARROW, {0}, {1}, SliceAttrs{0, 1, 2, 1}),
        OpDesc(OpCode::SLICE, {0}, {1}, SliceAttrs{0, 1, 2, 2}),
        OpDesc(OpCode::TRANSPOSE, {0}, {1}, TransposeAttrs{0, 1}),
        OpDesc(OpCode::PERMUTE, {0}, {1}, PermuteAttrs{{2, 0, 1}})}) {
        require(op.serialize().find(op_name(op.code())) != std::string::npos, "operator serialize/print vocabulary");
        code(validate_schema(op.code(), op.inputs(), op.outputs(), op.attrs()), StatusCode::Ok);
    }
    throws<std::invalid_argument>([] { OpDesc bad(OpCode::ADD, {0}, {1}); });
    throws<std::invalid_argument>([] { (void)Status::failure(StatusCode::Ok, "invalid"); });
    throws<std::invalid_argument>([] { (void)Status::failure(StatusCode::InvalidArgument, ""); });
}
} // namespace
int main() {
    try {
        schema_tests();
        std::cout << "Operator schema/arity/typed attributes/ID/status/serialization: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_operator: " << error.what() << '\n';
        return 1;
    }
}
