#include "cpu_scalar.hpp"
#include <cmath>

namespace runtime::detail {
Status matmul_ijk_fp32_v0(const Tensor& first, const Tensor& second, Tensor& output) {
    const auto m = as_size(first.shape()[0]), k = as_size(first.shape()[1]), n = as_size(second.shape()[1]);
    const auto* a = first.data<float>();
    const auto* b = second.data<float>();
    auto* c = output.data<float>();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0;
            for (std::size_t inner = 0; inner < k; ++inner)
                sum += a[i * k + inner] * b[inner * n + j];
            if (!std::isfinite(sum)) return Status::failure(StatusCode::NonFinite, "FP32 ijk MATMUL result is nonfinite");
            c[i * n + j] = sum;
        }
    }
    return Status::success();
}
} // namespace runtime::detail
