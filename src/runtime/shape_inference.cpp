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
