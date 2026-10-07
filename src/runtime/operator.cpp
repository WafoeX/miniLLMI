#include "runtime/operator.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
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
    RUNTIME_STATUS_NAME(NonFinite); RUNTIME_STATUS_NAME(ResourceExhausted);
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
    RUNTIME_OP_NAME(RMSNORM); RUNTIME_OP_NAME(SOFTMAX); RUNTIME_OP_NAME(ROPE);
    RUNTIME_OP_NAME(EMBEDDING); RUNTIME_OP_NAME(SWIGLU); RUNTIME_OP_NAME(ATTENTION);
#undef RUNTIME_OP_NAME
    }
    throw std::invalid_argument("unknown operator code");
}
namespace {
template<class T> const T* attribute(const OpAttrs& attrs) { return std::get_if<T>(&attrs); }
Status bad_attribute() { return Status::failure(StatusCode::AttributeMismatch, "attribute type/range does not match operator"); }
std::size_t input_arity(OpCode code) {
    switch (code) {
    case OpCode::ADD: case OpCode::MUL: case OpCode::MATMUL: case OpCode::COPY:
    case OpCode::RMSNORM: case OpCode::EMBEDDING: case OpCode::SWIGLU: return 2;
    case OpCode::ATTENTION: return 3;
    default: (void)op_name(code); return 1;
    }
}
bool positions_valid(std::int64_t query, std::int64_t key, std::int64_t maximum) {
    return maximum > 0 && query >= 0 && key >= 0 && query <= maximum && key <= maximum;
}
bool positive_fp32(double value) {
    return std::isfinite(value) && value > 0 && std::isfinite(static_cast<float>(value)) && static_cast<float>(value) > 0;
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
        const auto* copy = code == OpCode::COPY ? attribute<CopyAttrs>(attrs) : nullptr;
        const auto arity = copy && copy->destination ? std::size_t{1} : input_arity(code);
        if (inputs.size() != arity || outputs.size() != 1)
            return Status::failure(StatusCode::ArityMismatch, "operator requires its fixed input arity and one output");
        if (std::find(inputs.begin(), inputs.end(), INVALID_TENSOR_ID) != inputs.end() || outputs[0] == INVALID_TENSOR_ID)
            return Status::failure(StatusCode::InvalidArgument, "invalid tensor ID sentinel");
        if (std::find(inputs.begin(), inputs.end(), outputs[0]) != inputs.end())
            return Status::failure(StatusCode::InvalidArgument, "output must have a distinct logical tensor/state ID");
        switch (code) {
        case OpCode::ADD: case OpCode::MUL: case OpCode::MATMUL: case OpCode::MATERIALIZE:
        case OpCode::EMBEDDING: case OpCode::SWIGLU:
            if (!attribute<std::monostate>(attrs)) return bad_attribute();
            break;
        case OpCode::RMSNORM: {
            const auto* a = attribute<NormAttrs>(attrs);
            if (!a || !positive_fp32(a->epsilon)) return bad_attribute();
            break;
        }
        case OpCode::SOFTMAX: {
            const auto* a = attribute<SoftmaxAttrs>(attrs);
            if (!a || !positions_valid(a->query_position, a->key_position, a->max_positions)) return bad_attribute();
            break;
        }
        case OpCode::ROPE: {
            const auto* a = attribute<RopeAttrs>(attrs);
            if (!a || !positions_valid(a->position, 0, a->max_positions) || !positive_fp32(a->base) || static_cast<float>(a->base) <= 1)
                return bad_attribute();
            break;
        }
        case OpCode::ATTENTION: {
            const auto* a = attribute<AttentionAttrs>(attrs);
            if (!a || a->heads <= 0 || a->head_dim <= 0 || a->head_dim % 2 != 0 ||
                !positions_valid(a->query_position, a->key_position, a->max_positions)) return bad_attribute();
            (void)as_dimension(checked_mul(as_size(a->heads), as_size(a->head_dim)));
            break;
        }
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
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\"version\":" << OP_SEMANTICS_VERSION << ",\"op\":\"" << op_name(code_) << "\",\"inputs\":";
    array(out, inputs_); out << ",\"outputs\":"; array(out, outputs_); out << ",\"attributes\":";
    std::visit([&](const auto& a) {
        using A = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<A, std::monostate>) out << "{}";
        else if constexpr (std::is_same_v<A, CopyAttrs>) {
            out << "{\"overlap\":\"reject_except_exact_self\"";
            if (a.destination) out << ",\"destination\":\"" << (a.destination->type() == DeviceType::CPU ? "cpu:" : "cuda:") << a.destination->index() << '"';
            out << '}';
        }
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
        } else if constexpr (std::is_same_v<A, NormAttrs>) {
            out << "{\"epsilon\":" << a.epsilon << '}';
        } else if constexpr (std::is_same_v<A, SoftmaxAttrs>) {
            out << "{\"causal\":" << (a.causal ? "true" : "false") << ",\"query_position\":" << a.query_position
                << ",\"key_position\":" << a.key_position << ",\"max_positions\":" << a.max_positions << '}';
        } else if constexpr (std::is_same_v<A, RopeAttrs>) {
            out << "{\"position\":" << a.position << ",\"base\":" << a.base << ",\"max_positions\":" << a.max_positions << '}';
        } else if constexpr (std::is_same_v<A, AttentionAttrs>) {
            out << "{\"heads\":" << a.heads << ",\"head_dim\":" << a.head_dim << ",\"causal\":" << (a.causal ? "true" : "false")
                << ",\"query_position\":" << a.query_position << ",\"key_position\":" << a.key_position
                << ",\"max_positions\":" << a.max_positions << '}';
        }
    }, attrs_);
    out << ",\"backend_hint\":";
    if (backend_hint_) out << '"' << (backend_hint_->type() == DeviceType::CPU ? "cpu:" : "cuda:") << backend_hint_->index() << '"';
    else out << "null";
    out << '}';
    return out.str();
}
} // namespace runtime
