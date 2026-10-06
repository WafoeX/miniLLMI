#include "cpu_optimized.hpp"
#include "runtime/thread_pool.hpp"
#include <cmath>

namespace runtime::detail {
namespace {
Status multiply_rows_ikj(const Tensor& first, const Tensor& second, Tensor& output,
                         std::size_t first_row, std::size_t last_row) {
    const auto k = as_size(first.shape()[1]), n = as_size(second.shape()[1]);
    const auto* a = first.data<float>();
    const auto* b = second.data<float>();
    auto* c = output.data<float>();
    for (std::size_t i = first_row; i < last_row; ++i) {
        auto* row = c + i * n;
        for (std::size_t inner = 0; inner < k; ++inner) {
            const float left = a[i * k + inner];
            const auto* right = b + inner * n;
            for (std::size_t j = 0; j < n; ++j) row[j] += left * right[j];
        }
    }
    return Status::success();
}

Status check_finite(const Tensor& output) {
    const auto* c = output.data<float>();
    for (std::size_t index = 0; index < output.numel(); ++index)
        if (!std::isfinite(c[index])) return Status::failure(StatusCode::NonFinite, "FP32 ikj MATMUL result is nonfinite");
    return Status::success();
}

void initialize_output(Tensor& output) {
    auto* c = output.data<float>();
    for (std::size_t index = 0; index < output.numel(); ++index) c[index] = 0.0F;
}
} // namespace

Status matmul_ikj_fp32_c1(const Tensor& first, const Tensor& second, Tensor& output, ThreadPool*) {
    const auto m = as_size(first.shape()[0]);
    initialize_output(output); // K=0 has the ordinary zero-product result.
    const auto status = multiply_rows_ikj(first, second, output, 0, m);
    return status.ok() ? check_finite(output) : status;
}

Status matmul_ikj_fifo_pool_fp32_c3(const Tensor& first, const Tensor& second, Tensor& output, ThreadPool* pool) {
    if (!pool) return Status::failure(StatusCode::InvalidArgument, "FIFO pool MATMUL requires a persistent thread pool");
    const auto m = as_size(first.shape()[0]);
    initialize_output(output);
    const auto status = pool->parallel_for(m, [&first, &second, &output](std::size_t row) {
        return multiply_rows_ikj(first, second, output, row, row + 1);
    });
    return status.ok() ? check_finite(output) : status;
}
} // namespace runtime::detail
