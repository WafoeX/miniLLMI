#pragma once

#include "runtime/device.hpp"
#include "runtime/model_contract.hpp"
#include "runtime/shape.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace runtime {
using TensorId = std::uint32_t;
inline constexpr TensorId INVALID_TENSOR_ID = UINT32_MAX;
inline constexpr unsigned OP_SEMANTICS_VERSION = 1;

enum class StatusCode {
    Ok, InvalidArgument, ArityMismatch, AttributeMismatch, ShapeMismatch,
    DTypeMismatch, LayoutMismatch, DeviceMismatch, OutOfRange, Overflow,
    Aliasing, Unsupported, NonFinite, ResourceExhausted
};
struct Status {
    StatusCode code = StatusCode::Ok;
    std::string message;
    bool ok() const noexcept { return code == StatusCode::Ok; }
    static Status success() { return {}; }
    static Status failure(StatusCode code, std::string message);
};
const char* status_name(StatusCode code);

enum class OpCode {
    ADD, MUL, MATMUL, COPY, MATERIALIZE, RESHAPE, VIEW,
    NARROW, SLICE, TRANSPOSE, PERMUTE,
    RMSNORM, SOFTMAX, ROPE, EMBEDDING, SWIGLU, ATTENTION
};
const char* op_name(OpCode code);

enum class CopyOverlap { RejectExceptExactSelf };
struct CopyAttrs {
    CopyOverlap overlap = CopyOverlap::RejectExceptExactSelf;
    // One-input COPY allocates a canonical destination on this device.
    // Two-input COPY remains a versioned write to explicit persistent state.
    std::optional<Device> destination = std::nullopt;
};
struct ReshapeAttrs { Shape shape; };
struct ViewAttrs { Shape shape; Stride stride; std::size_t offset_bytes = 0; };
struct SliceAttrs {
    std::size_t axis = 0;
    std::int64_t start = 0;
    std::int64_t length = 0;
    std::int64_t step = 1;
};
struct TransposeAttrs { std::size_t first = 0; std::size_t second = 1; };
struct PermuteAttrs { std::vector<std::size_t> axes; };
struct NormAttrs { double epsilon = tiny_model::RMS_EPSILON; };
struct SoftmaxAttrs {
    bool causal = false;
    std::int64_t query_position = 0;
    std::int64_t key_position = 0;
    std::int64_t max_positions = tiny_model::MAX_SEQ;
};
struct RopeAttrs {
    std::int64_t position = 0;
    double base = tiny_model::ROPE_BASE;
    std::int64_t max_positions = tiny_model::MAX_SEQ;
};
struct AttentionAttrs {
    std::int64_t heads = tiny_model::HEADS;
    std::int64_t head_dim = tiny_model::HEAD_DIM;
    bool causal = true;
    std::int64_t query_position = 0;
    std::int64_t key_position = 0;
    std::int64_t max_positions = tiny_model::MAX_SEQ;
};
using OpAttrs = std::variant<std::monostate, CopyAttrs, ReshapeAttrs, ViewAttrs,
                             SliceAttrs, TransposeAttrs, PermuteAttrs, NormAttrs,
                             SoftmaxAttrs, RopeAttrs, AttentionAttrs>;

// Schema validation does not resolve graph IDs or access tensor values.
Status validate_schema(OpCode code, const std::vector<TensorId>& inputs,
                       const std::vector<TensorId>& outputs, const OpAttrs& attrs);

class OpDesc {
public:
    // Throws invalid_argument for a malformed schema; validate_schema provides
    // the structured nonthrowing semantic-error interface. No execute/backend.
    OpDesc(OpCode code, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
           OpAttrs attrs = {}, std::optional<Device> backend_hint = std::nullopt);
    OpCode code() const noexcept { return code_; }
    const std::vector<TensorId>& inputs() const noexcept { return inputs_; }
    const std::vector<TensorId>& outputs() const noexcept { return outputs_; }
    const OpAttrs& attrs() const noexcept { return attrs_; }
    const std::optional<Device>& backend_hint() const noexcept { return backend_hint_; }
    std::string serialize() const;
private:
    OpCode code_;
    std::vector<TensorId> inputs_;
    std::vector<TensorId> outputs_;
    OpAttrs attrs_;
    std::optional<Device> backend_hint_;
};
} // namespace runtime
