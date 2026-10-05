#include "runtime/shape_inference.hpp"
#include "runtime/copy.hpp"
#include <stdexcept>
#include <utility>

namespace runtime {
namespace {
InferenceResult fail(StatusCode code, const char* message) {
    return {Status::failure(code, message), std::nullopt};
}
InferenceResult canonical(const Tensor& prototype, Shape shape) {
    (void)nbytes(shape, prototype.dtype());
    auto stride = contiguous_stride(shape);
    return {Status::success(), OutputContract{std::move(shape), std::move(stride), prototype.dtype(),
            prototype.device(), OutputKind::NewTensor, std::nullopt}};
}
InferenceResult alias(Tensor tensor, OutputKind kind = OutputKind::Alias) {
    OutputContract result{tensor.shape(), tensor.stride(), tensor.dtype(), tensor.device(), kind, std::move(tensor)};
    return {Status::success(), std::move(result)};
}
bool position_window(std::int64_t start, std::int64_t length, std::int64_t maximum) {
    return checked_add(as_size(start), as_size(length)) <= as_size(maximum);
}
Status math_inputs(const TensorInputs& inputs) {
    for (const auto& input : inputs) {
        if (input.get().dtype() != DType::FP32)
            return Status::failure(StatusCode::DTypeMismatch, "arithmetic requires FP32; no INT32 arithmetic or promotion");
        if (!input.get().is_contiguous())
            return Status::failure(StatusCode::LayoutMismatch, "arithmetic requires explicit contiguous input materialization");
        if (input.get().device() != inputs[0].get().device())
            return Status::failure(StatusCode::DeviceMismatch, "arithmetic inputs require the same device; transfers must be explicit");
    }
    return Status::success();
}
} // namespace
InferenceResult infer_operator(const OpDesc& descriptor, const TensorInputs& inputs) {
    if (inputs.size() != descriptor.inputs().size()) return fail(StatusCode::ArityMismatch, "binding count differs from descriptor input arity");
    const auto& first = inputs[0].get();
    try {
        switch (descriptor.code()) {
        case OpCode::ADD: case OpCode::MUL: case OpCode::MATMUL: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            const auto& second = inputs[1].get();
            if (descriptor.code() == OpCode::MATMUL) {
                if (first.shape().rank() != 2 || second.shape().rank() != 2 || first.shape()[1] != second.shape()[0])
                    return fail(StatusCode::ShapeMismatch, "MATMUL requires A[M,K] and B[K,N], rank 2 only");
                return canonical(first, Shape{first.shape()[0], second.shape()[1]});
            }
            if (first.shape() != second.shape()) return fail(StatusCode::ShapeMismatch, "elementwise shapes must match exactly; broadcasting unsupported");
            return canonical(first, first.shape());
        }
        case OpCode::RMSNORM: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            const auto& scale = inputs[1].get();
            if (first.shape().rank() == 0 || scale.shape().rank() != 1 ||
                first.shape()[first.shape().rank() - 1] <= 0 || scale.shape()[0] != first.shape()[first.shape().rank() - 1])
                return fail(StatusCode::ShapeMismatch, "RMSNORM requires nonempty last channel dimension and scale[channel]");
            return canonical(first, first.shape());
        }
        case OpCode::SOFTMAX: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            if (first.shape().rank() != 2) return fail(StatusCode::ShapeMismatch, "SOFTMAX requires scores[queries,keys]");
            const auto& a = std::get<SoftmaxAttrs>(descriptor.attrs());
            if (!position_window(a.query_position, first.shape()[0], a.max_positions) ||
                !position_window(a.key_position, first.shape()[1], a.max_positions))
                return fail(StatusCode::OutOfRange, "SOFTMAX absolute position range exceeds declared maximum");
            return canonical(first, first.shape());
        }
        case OpCode::ROPE: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            const auto rank = first.shape().rank();
            if (rank != 2 && rank != 3) return fail(StatusCode::ShapeMismatch, "ROPE requires [tokens,dim] or [tokens,heads,dim]");
            const auto dim = first.shape()[rank - 1];
            if (dim <= 0 || dim % 2 != 0 || (rank == 3 && first.shape()[1] <= 0))
                return fail(StatusCode::ShapeMismatch, "ROPE requires positive even head dimension and positive heads");
            const auto& a = std::get<RopeAttrs>(descriptor.attrs());
            if (!position_window(a.position, first.shape()[0], a.max_positions))
                return fail(StatusCode::OutOfRange, "ROPE absolute positions exceed declared maximum");
            return canonical(first, first.shape());
        }
        case OpCode::EMBEDDING: {
            const auto& table = inputs[1].get();
            if (first.dtype() != DType::INT32 || table.dtype() != DType::FP32)
                return fail(StatusCode::DTypeMismatch, "EMBEDDING requires INT32 IDs and FP32 table");
            if (!first.is_contiguous() || !table.is_contiguous())
                return fail(StatusCode::LayoutMismatch, "EMBEDDING requires explicit contiguous inputs");
            if (first.device() != table.device()) return fail(StatusCode::DeviceMismatch, "EMBEDDING inputs require same device");
            if (first.shape().rank() != 1 || table.shape().rank() != 2 || table.shape()[0] <= 0 || table.shape()[1] <= 0)
                return fail(StatusCode::ShapeMismatch, "EMBEDDING requires IDs[tokens], table[vocab,hidden] with positive vocab/hidden");
            // ID values are runtime kernel checks, never host reads during pure inference.
            return canonical(table, Shape{first.shape()[0], table.shape()[1]});
        }
        case OpCode::SWIGLU: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            if (first.shape() != inputs[1].get().shape()) return fail(StatusCode::ShapeMismatch, "SWIGLU gate/up shapes must match exactly");
            return canonical(first, first.shape());
        }
        case OpCode::ATTENTION: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            const auto& key = inputs[1].get();
            const auto& value = inputs[2].get();
            const auto& a = std::get<AttentionAttrs>(descriptor.attrs());
            if (first.shape().rank() != 3 || key.shape().rank() != 3 || value.shape() != key.shape() ||
                first.shape()[1] != a.heads || key.shape()[1] != a.heads ||
                first.shape()[2] != a.head_dim || key.shape()[2] != a.head_dim)
                return fail(StatusCode::ShapeMismatch, "ATTENTION requires Q[queries,H,D], K/V[keys,H,D] matching declared H/D");
            if (!position_window(a.query_position, first.shape()[0], a.max_positions) ||
                !position_window(a.key_position, key.shape()[0], a.max_positions))
                return fail(StatusCode::OutOfRange, "ATTENTION absolute query/key ranges exceed declared maximum");
            return canonical(first, first.shape()); // composite only; future graph lowers to ordinary core ops
        }
        case OpCode::COPY: {
            const auto& destination = inputs[1].get();
            if (first.dtype() != destination.dtype()) return fail(StatusCode::DTypeMismatch, "COPY dtype mismatch");
            if (first.shape() != destination.shape()) return fail(StatusCode::ShapeMismatch, "COPY shape mismatch; ranges must be predeclared views");
            if (!same_tensor_layout(first, destination) && memory_spans_overlap(first, destination))
                return fail(StatusCode::Aliasing, "COPY rejects intersecting address spans except exact self-copy");
            // Cross-device COPY is a valid declaration, not implemented execution.
            return alias(destination, OutputKind::Write);
        }
        case OpCode::MATERIALIZE: return canonical(first, first.shape());
        case OpCode::RESHAPE: {
            const auto& shape = std::get<ReshapeAttrs>(descriptor.attrs()).shape;
            if (!first.is_contiguous()) return fail(StatusCode::LayoutMismatch, "RESHAPE requires contiguous source");
            if (numel(shape) != first.numel()) return fail(StatusCode::ShapeMismatch, "RESHAPE cannot change numel");
            return alias(first.reshape(shape));
        }
        case OpCode::VIEW: {
            const auto& a = std::get<ViewAttrs>(descriptor.attrs());
            return alias(first.view(a.shape, a.stride, a.offset_bytes));
        }
        case OpCode::NARROW: case OpCode::SLICE: {
            const auto& a = std::get<SliceAttrs>(descriptor.attrs());
            return alias(first.slice(a.axis, a.start, a.length, a.step));
        }
        case OpCode::TRANSPOSE: {
            const auto& a = std::get<TransposeAttrs>(descriptor.attrs());
            return alias(first.transpose(a.first, a.second));
        }
        case OpCode::PERMUTE: {
            const auto& a = std::get<PermuteAttrs>(descriptor.attrs());
            if (a.axes.size() != first.shape().rank()) return fail(StatusCode::ShapeMismatch, "PERMUTE rank mismatch");
            return alias(first.permute(a.axes));
        }
        }
        return fail(StatusCode::Unsupported, "operator inference not implemented");
    } catch (const std::overflow_error& error) {
        return {Status::failure(StatusCode::Overflow, error.what()), std::nullopt};
    } catch (const std::out_of_range& error) {
        return {Status::failure(StatusCode::OutOfRange, error.what()), std::nullopt};
    } catch (const std::invalid_argument& error) {
        return {Status::failure(StatusCode::InvalidArgument, error.what()), std::nullopt};
    }
}
Status validate_output_binding(const OutputContract& contract, const Tensor& output) {
    if (output.dtype() != contract.dtype) return Status::failure(StatusCode::DTypeMismatch, "output dtype differs from inference");
    if (output.shape() != contract.shape) return Status::failure(StatusCode::ShapeMismatch, "output shape differs from inference");
    if (output.device() != contract.device) return Status::failure(StatusCode::DeviceMismatch, "output device differs from inference");
    if (contract.kind == OutputKind::NewTensor) {
        if (!output.is_contiguous()) return Status::failure(StatusCode::LayoutMismatch, "new operator output requires contiguous layout");
    } else if (!contract.alias || !same_tensor_layout(*contract.alias, output)) {
        return Status::failure(StatusCode::Aliasing, "alias/write output must bind the exact inferred Storage and metadata");
    }
    return Status::success();
}
} // namespace runtime
