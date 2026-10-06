#include "runtime/reference.hpp"
#include "runtime/copy.hpp"
#include "cpu_dispatch.hpp"
#include <cmath>
#include <stdexcept>

namespace runtime::reference {
namespace {
Status finite_inputs(const TensorInputs& inputs) {
    for (const auto& input : inputs) {
        const auto* data = input.get().data<float>();
        for (std::size_t i = 0; i < input.get().numel(); ++i)
            if (!std::isfinite(data[i])) return Status::failure(StatusCode::NonFinite, "arithmetic input contains NaN/Inf");
    }
    return Status::success();
}
Status elementwise(OpCode code, const Tensor& first, const Tensor& second, Tensor& output) {
    const auto* a = first.data<float>();
    const auto* b = second.data<float>();
    auto* c = output.data<float>();
    for (std::size_t i = 0; i < output.numel(); ++i) {
        const float value = code == OpCode::ADD ? a[i] + b[i] : a[i] * b[i];
        if (!std::isfinite(value)) return Status::failure(StatusCode::NonFinite, "elementwise result is nonfinite");
        c[i] = value;
    }
    return Status::success();
}
Status matmul(const Tensor& first, const Tensor& second, Tensor& output) {
    const auto m = as_size(first.shape()[0]), k = as_size(first.shape()[1]), n = as_size(second.shape()[1]);
    const auto* a = first.data<float>();
    const auto* b = second.data<float>();
    auto* c = output.data<float>();
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            double sum = 0; // correctness oracle, not the future FP32 optimization baseline
            for (std::size_t inner = 0; inner < k; ++inner)
                sum += static_cast<double>(a[row * k + inner]) * static_cast<double>(b[inner * n + column]);
            const auto value = static_cast<float>(sum);
            if (!std::isfinite(value)) return Status::failure(StatusCode::NonFinite, "MATMUL result is nonfinite");
            c[row * n + column] = value;
        }
    }
    return Status::success();
}
} // namespace
Status execute(const OpDesc& descriptor, const TensorInputs& inputs, Tensor& output) {
    return detail::execute_cpu_core(descriptor, inputs, output, matmul);
}
} // namespace runtime::reference

namespace runtime::detail {
Status validate_cpu_core(const OpDesc& descriptor, const TensorInputs& inputs, const Tensor& output, OutputKind& kind) {
    const auto inferred = infer_operator(descriptor, inputs);
    if (!inferred.ok()) return inferred.status;
    const auto binding = validate_output_binding(*inferred.output, output);
    if (!binding.ok()) return binding;
    if (output.device().type() != DeviceType::CPU)
        return Status::failure(StatusCode::DeviceMismatch, "CPU reference requires CPU output");
    for (const auto& input : inputs)
        if (input.get().device().type() != DeviceType::CPU)
            return Status::failure(StatusCode::DeviceMismatch, "CPU reference requires explicit CPU inputs, no hidden transfers");
    kind = inferred.output->kind;
    if (descriptor.code() == OpCode::MATERIALIZE && memory_spans_overlap(inputs[0].get(), output))
        return Status::failure(StatusCode::Aliasing, "MATERIALIZE requires independent output storage");
    if (descriptor.code() == OpCode::ADD || descriptor.code() == OpCode::MUL || descriptor.code() == OpCode::MATMUL)
        for (const auto& input : inputs)
            if (memory_spans_overlap(input.get(), output))
                return Status::failure(StatusCode::Aliasing, "CPU arithmetic rejects overlapping input/output spans");
    return Status::success();
}
Status execute_cpu_core(const OpDesc& descriptor, const TensorInputs& inputs, Tensor& output, MatmulKernel matmul) {
    OutputKind kind{};
    const auto binding = validate_cpu_core(descriptor, inputs, output, kind);
    if (!binding.ok()) return binding;
    try {
        switch (descriptor.code()) {
        case OpCode::COPY:
            (void)copy_cpu(inputs[0].get(), output);
            return Status::success();
        case OpCode::MATERIALIZE:
            (void)copy_cpu(inputs[0].get(), output);
            return Status::success();
        case OpCode::ADD: case OpCode::MUL: case OpCode::MATMUL: {
            const auto finite = reference::finite_inputs(inputs);
            if (!finite.ok()) return finite;
            if (descriptor.code() == OpCode::MATMUL) return matmul(inputs[0].get(), inputs[1].get(), output);
            return reference::elementwise(descriptor.code(), inputs[0].get(), inputs[1].get(), output);
        }
        default:
            return Status::failure(StatusCode::Unsupported, "operator has no Stage 2 CPU reference kernel; metadata aliases bind through inference");
        }
    } catch (const std::overflow_error& error) {
        return Status::failure(StatusCode::Overflow, error.what());
    } catch (const std::invalid_argument& error) {
        return Status::failure(StatusCode::InvalidArgument, error.what());
    }
}
} // namespace runtime::detail
