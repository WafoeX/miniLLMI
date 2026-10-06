#include "cpu_optimized.hpp"
#include <cmath>

namespace runtime::detail {
Status matmul_ikj_fp32_c1(const Tensor& first, const Tensor& second, Tensor& output) {
    const auto m = as_size(first.shape()[0]), k = as_size(first.shape()[1]), n = as_size(second.shape()[1]);
    const auto* a = first.data<float>();
    const auto* b = second.data<float>();
    auto* c = output.data<float>();

    // Explicitly initialize so K=0 has the ordinary zero-product result.
    for (std::size_t index = 0; index < m * n; ++index) c[index] = 0.0F;
    for (std::size_t i = 0; i < m; ++i) {
        auto* row = c + i * n;
        for (std::size_t inner = 0; inner < k; ++inner) {
            const float left = a[i * k + inner];
            const auto* right = b + inner * n;
            for (std::size_t j = 0; j < n; ++j) row[j] += left * right[j];
        }
    }
    for (std::size_t index = 0; index < m * n; ++index)
        if (!std::isfinite(c[index])) return Status::failure(StatusCode::NonFinite, "FP32 ikj MATMUL result is nonfinite");
    return Status::success();
}
} // namespace runtime::detail
