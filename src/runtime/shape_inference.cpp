#include "runtime/shape_inference.hpp"
#include "runtime/copy.hpp"
#include <stdexcept>
#include <utility>

namespace runtime {
namespace {
LayoutInferenceResult fail(StatusCode code, const char* message) {
    return {Status::failure(code, message), std::nullopt};
}
LayoutInferenceResult canonical(const Layout& prototype, Shape shape) {
    const auto bytes = nbytes(shape, prototype.dtype);
    auto stride = contiguous_stride(shape);
    return {Status::success(), LayoutOutputContract{Layout{std::move(shape), std::move(stride),
            prototype.dtype, prototype.device, bytes, 0, 0}, OutputKind::NewTensor}};
}
LayoutInferenceResult alias(Layout layout, OutputKind kind = OutputKind::Alias) {
    return {Status::success(), LayoutOutputContract{std::move(layout), kind}};
}
bool position_window(std::int64_t start, std::int64_t length, std::int64_t maximum) {
    return checked_add(as_size(start), as_size(length)) <= as_size(maximum);
}
Status math_inputs(const LayoutInputs& inputs) {
    for (const auto& input : inputs) {
        if (input.get().dtype != DType::FP32)
            return Status::failure(StatusCode::DTypeMismatch, "arithmetic requires FP32; no INT32 arithmetic or promotion");
        if (!input.get().is_contiguous())
            return Status::failure(StatusCode::LayoutMismatch, "arithmetic requires explicit contiguous input materialization");
        if (input.get().device != inputs[0].get().device)
            return Status::failure(StatusCode::DeviceMismatch, "arithmetic inputs require the same device; transfers must be explicit");
    }
    return Status::success();
}
} // namespace
LayoutInferenceResult infer_layout(const OpDesc& descriptor, const LayoutInputs& inputs) {
    const auto schema = validate_schema(descriptor.code(), descriptor.inputs(), descriptor.outputs(), descriptor.attrs());
    if (!schema.ok()) return {schema, std::nullopt};
    if (inputs.empty() || inputs.size() != descriptor.inputs().size())
        return fail(StatusCode::ArityMismatch, "binding count differs from descriptor input arity");
    const auto& first = inputs[0].get();
    try {
        for (const auto& input : inputs) input.get().validate();
        switch (descriptor.code()) {
        case OpCode::ADD: case OpCode::MUL: case OpCode::MATMUL: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            const auto& second = inputs[1].get();
            if (descriptor.code() == OpCode::MATMUL) {
                if (first.shape.rank() != 2 || second.shape.rank() != 2 || first.shape[1] != second.shape[0])
                    return fail(StatusCode::ShapeMismatch, "MATMUL requires A[M,K] and B[K,N], rank 2 only");
                return canonical(first, Shape{first.shape[0], second.shape[1]});
            }
            if (first.shape != second.shape)
                return fail(StatusCode::ShapeMismatch, "elementwise shapes must match exactly; broadcasting unsupported");
            return canonical(first, first.shape);
        }
        case OpCode::RMSNORM: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            const auto& scale = inputs[1].get();
            if (first.shape.rank() == 0 || scale.shape.rank() != 1 ||
                first.shape[first.shape.rank() - 1] <= 0 || scale.shape[0] != first.shape[first.shape.rank() - 1])
                return fail(StatusCode::ShapeMismatch, "RMSNORM requires nonempty last channel dimension and scale[channel]");
            return canonical(first, first.shape);
        }
        case OpCode::SOFTMAX: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            if (first.shape.rank() != 2) return fail(StatusCode::ShapeMismatch, "SOFTMAX requires scores[queries,keys]");
            const auto& a = std::get<SoftmaxAttrs>(descriptor.attrs());
            if (!position_window(a.query_position, first.shape[0], a.max_positions) ||
                !position_window(a.key_position, first.shape[1], a.max_positions))
                return fail(StatusCode::OutOfRange, "SOFTMAX absolute position range exceeds declared maximum");
            return canonical(first, first.shape);
        }
        case OpCode::ROPE: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            const auto rank = first.shape.rank();
            if (rank != 2 && rank != 3) return fail(StatusCode::ShapeMismatch, "ROPE requires [tokens,dim] or [tokens,heads,dim]");
            const auto dim = first.shape[rank - 1];
            if (dim <= 0 || dim % 2 != 0 || (rank == 3 && first.shape[1] <= 0))
                return fail(StatusCode::ShapeMismatch, "ROPE requires positive even head dimension and positive heads");
            const auto& a = std::get<RopeAttrs>(descriptor.attrs());
            if (!position_window(a.position, first.shape[0], a.max_positions))
                return fail(StatusCode::OutOfRange, "ROPE absolute positions exceed declared maximum");
            return canonical(first, first.shape);
        }
        case OpCode::EMBEDDING: {
            const auto& table = inputs[1].get();
            if (first.dtype != DType::INT32 || table.dtype != DType::FP32)
                return fail(StatusCode::DTypeMismatch, "EMBEDDING requires INT32 IDs and FP32 table");
            if (!first.is_contiguous() || !table.is_contiguous())
                return fail(StatusCode::LayoutMismatch, "EMBEDDING requires explicit contiguous inputs");
            if (first.device != table.device) return fail(StatusCode::DeviceMismatch, "EMBEDDING inputs require same device");
            if (first.shape.rank() != 1 || table.shape.rank() != 2 || table.shape[0] <= 0 || table.shape[1] <= 0)
                return fail(StatusCode::ShapeMismatch, "EMBEDDING requires IDs[tokens], table[vocab,hidden] with positive vocab/hidden");
            return canonical(table, Shape{first.shape[0], table.shape[1]});
        }
        case OpCode::SWIGLU: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            if (first.shape != inputs[1].get().shape) return fail(StatusCode::ShapeMismatch, "SWIGLU gate/up shapes must match exactly");
            return canonical(first, first.shape);
        }
        case OpCode::ATTENTION: {
            const auto valid = math_inputs(inputs);
            if (!valid.ok()) return {valid, std::nullopt};
            const auto& key = inputs[1].get();
            const auto& value = inputs[2].get();
            const auto& a = std::get<AttentionAttrs>(descriptor.attrs());
            if (first.shape.rank() != 3 || key.shape.rank() != 3 || value.shape != key.shape ||
                first.shape[1] != a.heads || key.shape[1] != a.heads ||
                first.shape[2] != a.head_dim || key.shape[2] != a.head_dim)
                return fail(StatusCode::ShapeMismatch, "ATTENTION requires Q[queries,H,D], K/V[keys,H,D] matching declared H/D");
            if (!position_window(a.query_position, first.shape[0], a.max_positions) ||
                !position_window(a.key_position, key.shape[0], a.max_positions))
                return fail(StatusCode::OutOfRange, "ATTENTION absolute query/key ranges exceed declared maximum");
            return canonical(first, first.shape);
        }
        case OpCode::COPY: {
            const auto& attrs = std::get<CopyAttrs>(descriptor.attrs());
            if (attrs.destination) {
                auto prototype = first;
                prototype.device = *attrs.destination;
                return canonical(prototype, first.shape);
            }
            const auto& destination = inputs[1].get();
            if (first.dtype != destination.dtype) return fail(StatusCode::DTypeMismatch, "COPY dtype mismatch");
            if (first.shape != destination.shape) return fail(StatusCode::ShapeMismatch, "COPY shape mismatch; ranges must be predeclared views");
            if (!same_layout(first, destination) && spans_overlap(first, destination))
                return fail(StatusCode::Aliasing, "COPY rejects intersecting address spans except exact self-copy");
            return alias(destination, OutputKind::Write);
        }
        case OpCode::MATERIALIZE: return canonical(first, first.shape);
        case OpCode::RESHAPE: {
            const auto& shape = std::get<ReshapeAttrs>(descriptor.attrs()).shape;
            if (!first.is_contiguous()) return fail(StatusCode::LayoutMismatch, "RESHAPE requires contiguous source");
            if (numel(shape) != numel(first.shape)) return fail(StatusCode::ShapeMismatch, "RESHAPE cannot change numel");
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
            if (a.axes.size() != first.shape.rank()) return fail(StatusCode::ShapeMismatch, "PERMUTE rank mismatch");
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
InferenceResult infer_operator(const OpDesc& descriptor, const TensorInputs& inputs) {
    std::vector<Layout> layouts;
    layouts.reserve(inputs.size());
    for (const auto& input : inputs) {
        const auto& t = input.get();
        layouts.push_back({t.shape(), t.stride(), t.dtype(), t.device(), t.storage()->capacity_bytes(),
                           t.data_offset(), static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t.storage().get()))});
    }
    LayoutInputs metadata;
    for (const auto& layout : layouts) metadata.emplace_back(layout);
    const auto inferred = infer_layout(descriptor, metadata);
    if (!inferred.ok()) return {inferred.status, std::nullopt};
    // Distinct wrapped Storage objects may still refer to overlapping addresses.
    // Layout inference knows symbolic roots only; preserve Tensor's address check.
    if (descriptor.code() == OpCode::COPY && inputs.size() == 2 && !same_tensor_layout(inputs[0].get(), inputs[1].get()) &&
        memory_spans_overlap(inputs[0].get(), inputs[1].get()))
        return {Status::failure(StatusCode::Aliasing, "COPY rejects intersecting address spans except exact self-copy"), std::nullopt};
    const auto& layout = inferred.output->layout;
    std::optional<Tensor> binding;
    if (inferred.output->kind != OutputKind::NewTensor) {
        const auto& source = inputs[descriptor.code() == OpCode::COPY ? 1 : 0].get();
        binding.emplace(source.storage(), layout.dtype, layout.shape, layout.stride, layout.offset_bytes);
    }
    return {Status::success(), OutputContract{layout.shape, layout.stride, layout.dtype, layout.device,
                                             inferred.output->kind, std::move(binding)}};
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
