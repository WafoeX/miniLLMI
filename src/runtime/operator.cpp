#include "runtime/operator.hpp"
#include <algorithm>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace runtime {
Status Status::failure(StatusCode code, std::string message) {
    if (code == StatusCode::Ok || message.empty())
        throw std::invalid_argument("failure status requires a non-Ok code and diagnostic");
    return {code, std::move(message)};
}
const char* status_name(StatusCode code) {
    switch (code) {
#define RUNTIME_STATUS_NAME(name) case StatusCode::name: return #name
    RUNTIME_STATUS_NAME(Ok); RUNTIME_STATUS_NAME(InvalidArgument);
    RUNTIME_STATUS_NAME(ArityMismatch); RUNTIME_STATUS_NAME(AttributeMismatch);
    RUNTIME_STATUS_NAME(ShapeMismatch); RUNTIME_STATUS_NAME(DTypeMismatch);
    RUNTIME_STATUS_NAME(LayoutMismatch); RUNTIME_STATUS_NAME(DeviceMismatch);
    RUNTIME_STATUS_NAME(OutOfRange); RUNTIME_STATUS_NAME(Overflow);
    RUNTIME_STATUS_NAME(Aliasing); RUNTIME_STATUS_NAME(Unsupported);
    RUNTIME_STATUS_NAME(NonFinite);
#undef RUNTIME_STATUS_NAME
    }
    throw std::invalid_argument("unknown status code");
}
const char* op_name(OpCode code) {
    switch (code) {
#define RUNTIME_OP_NAME(name) case OpCode::name: return #name
    RUNTIME_OP_NAME(ADD); RUNTIME_OP_NAME(MUL); RUNTIME_OP_NAME(MATMUL);
    RUNTIME_OP_NAME(COPY); RUNTIME_OP_NAME(MATERIALIZE); RUNTIME_OP_NAME(RESHAPE);
    RUNTIME_OP_NAME(VIEW); RUNTIME_OP_NAME(NARROW); RUNTIME_OP_NAME(SLICE);
    RUNTIME_OP_NAME(TRANSPOSE); RUNTIME_OP_NAME(PERMUTE);
#undef RUNTIME_OP_NAME
    }
    throw std::invalid_argument("unknown operator code");
}
namespace {
template<class T> const T* attribute(const OpAttrs& attrs) { return std::get_if<T>(&attrs); }
Status bad_attribute() { return Status::failure(StatusCode::AttributeMismatch, "attribute type/range does not match operator"); }
std::size_t input_arity(OpCode code) {
    switch (code) {
    case OpCode::ADD: case OpCode::MUL: case OpCode::MATMUL: case OpCode::COPY: return 2;
    default: (void)op_name(code); return 1;
    }
}
template<class Container> void array(std::ostream& out, const Container& values) {
    out << '[';
    bool first = true;
    for (auto value : values) { if (!first) out << ','; out << value; first = false; }
    out << ']';
}
} // namespace
Status validate_schema(OpCode code, const std::vector<TensorId>& inputs,
                       const std::vector<TensorId>& outputs, const OpAttrs& attrs) {
    try {
        if (inputs.size() != input_arity(code) || outputs.size() != 1)
            return Status::failure(StatusCode::ArityMismatch, "operator requires its fixed input arity and one output");
        if (std::find(inputs.begin(), inputs.end(), INVALID_TENSOR_ID) != inputs.end() || outputs[0] == INVALID_TENSOR_ID)
            return Status::failure(StatusCode::InvalidArgument, "invalid tensor ID sentinel");
        if (std::find(inputs.begin(), inputs.end(), outputs[0]) != inputs.end())
            return Status::failure(StatusCode::InvalidArgument, "output must have a distinct logical tensor/state ID");
        switch (code) {
        case OpCode::ADD: case OpCode::MUL: case OpCode::MATMUL: case OpCode::MATERIALIZE:
            if (!attribute<std::monostate>(attrs)) return bad_attribute();
            break;
        case OpCode::COPY: {
            const auto* a = attribute<CopyAttrs>(attrs);
            if (!a || a->overlap != CopyOverlap::RejectExceptExactSelf) return bad_attribute();
            break;
        }
        case OpCode::RESHAPE:
            if (!attribute<ReshapeAttrs>(attrs)) return bad_attribute();
            (void)numel(attribute<ReshapeAttrs>(attrs)->shape);
            break;
        case OpCode::VIEW: {
            const auto* a = attribute<ViewAttrs>(attrs);
            if (!a || a->shape.rank() != a->stride.rank()) return bad_attribute();
            for (auto step : a->stride.values()) if (step == 0) return bad_attribute();
            (void)numel(a->shape);
            break;
        }
        case OpCode::NARROW: case OpCode::SLICE: {
            const auto* a = attribute<SliceAttrs>(attrs);
            if (!a || a->axis >= MAX_RANK || a->start < 0 || a->length < 0 || a->step <= 0 ||
                (code == OpCode::NARROW && a->step != 1)) return bad_attribute();
            break;
        }
        case OpCode::TRANSPOSE: {
            const auto* a = attribute<TransposeAttrs>(attrs);
            if (!a || a->first >= MAX_RANK || a->second >= MAX_RANK) return bad_attribute();
            break;
        }
        case OpCode::PERMUTE: {
            const auto* a = attribute<PermuteAttrs>(attrs);
            if (!a || a->axes.size() > MAX_RANK) return bad_attribute();
            auto sorted = a->axes;
            std::sort(sorted.begin(), sorted.end());
            for (std::size_t i = 0; i < sorted.size(); ++i) if (sorted[i] != i) return bad_attribute();
            break;
        }
        }
        return Status::success();
    } catch (const std::overflow_error& error) {
        return Status::failure(StatusCode::Overflow, error.what());
    } catch (const std::invalid_argument& error) {
        return Status::failure(StatusCode::InvalidArgument, error.what());
    }
}
OpDesc::OpDesc(OpCode code, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
               OpAttrs attrs, std::optional<Device> backend_hint)
    : code_(code), inputs_(std::move(inputs)), outputs_(std::move(outputs)),
      attrs_(std::move(attrs)), backend_hint_(backend_hint) {
    const auto status = validate_schema(code_, inputs_, outputs_, attrs_);
    if (!status.ok()) throw std::invalid_argument(std::string(status_name(status.code)) + ": " + status.message);
}
std::string OpDesc::serialize() const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"version\":" << OP_SEMANTICS_VERSION << ",\"op\":\"" << op_name(code_) << "\",\"inputs\":";
    array(out, inputs_); out << ",\"outputs\":"; array(out, outputs_); out << ",\"attributes\":";
    std::visit([&](const auto& a) {
        using A = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<A, std::monostate>) out << "{}";
        else if constexpr (std::is_same_v<A, CopyAttrs>) out << "{\"overlap\":\"reject_except_exact_self\"}";
        else if constexpr (std::is_same_v<A, ReshapeAttrs>) {
            out << "{\"shape\":"; array(out, a.shape.values()); out << '}';
        } else if constexpr (std::is_same_v<A, ViewAttrs>) {
            out << "{\"shape\":"; array(out, a.shape.values()); out << ",\"stride\":";
            array(out, a.stride.values()); out << ",\"offset_bytes\":" << a.offset_bytes << '}';
        } else if constexpr (std::is_same_v<A, SliceAttrs>) {
            out << "{\"axis\":" << a.axis << ",\"start\":" << a.start << ",\"length\":" << a.length << ",\"step\":" << a.step << '}';
        } else if constexpr (std::is_same_v<A, TransposeAttrs>) {
            out << "{\"first\":" << a.first << ",\"second\":" << a.second << '}';
        } else if constexpr (std::is_same_v<A, PermuteAttrs>) {
            out << "{\"axes\":"; array(out, a.axes); out << '}';
        }
    }, attrs_);
    out << ",\"backend_hint\":";
    if (backend_hint_) out << '"' << (backend_hint_->type() == DeviceType::CPU ? "cpu:" : "cuda:") << backend_hint_->index() << '"';
    else out << "null";
    out << '}';
    return out.str();
}
} // namespace runtime
