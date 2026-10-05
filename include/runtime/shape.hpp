#pragma once

#include "runtime/dtype.hpp"
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

namespace runtime {
inline constexpr std::size_t MAX_RANK = 8;

inline std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("tensor size addition overflow");
    return a + b;
}
inline std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("tensor size multiplication overflow");
    return a * b;
}
inline std::size_t as_size(std::int64_t value) {
    if (value < 0) throw std::invalid_argument("negative dimension or stride");
    if (static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("dimension exceeds size_t");
    return static_cast<std::size_t>(value);
}
inline std::int64_t as_dimension(std::size_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("stride exceeds int64_t");
    return static_cast<std::int64_t>(value);
}

namespace detail {
struct ShapeTag {};
struct StrideTag {};
template<class Tag> class Metadata {
public:
    Metadata() = default;
    Metadata(std::initializer_list<std::int64_t> values)
        : Metadata(std::vector<std::int64_t>(values)) {}
    explicit Metadata(std::vector<std::int64_t> values) : values_(std::move(values)) {
        if (values_.size() > MAX_RANK) throw std::invalid_argument("rank exceeds MAX_RANK");
        for (auto value : values_) (void)as_size(value);
    }
    std::size_t rank() const noexcept { return values_.size(); }
    std::int64_t operator[](std::size_t axis) const { return values_.at(axis); }
    const std::vector<std::int64_t>& values() const noexcept { return values_; }
    bool operator==(const Metadata& other) const noexcept { return values_ == other.values_; }
    bool operator!=(const Metadata& other) const noexcept { return !(*this == other); }
private:
    std::vector<std::int64_t> values_;
};
} // namespace detail
using Shape = detail::Metadata<detail::ShapeTag>;
using Stride = detail::Metadata<detail::StrideTag>;

inline std::size_t numel(const Shape& shape) {
    // Empty shapes short-circuit before multiplication (even with huge other axes).
    for (auto dim : shape.values()) if (dim == 0) return 0;
    std::size_t result = 1; // rank-0 scalar
    for (auto dim : shape.values()) result = checked_mul(result, as_size(dim));
    return result;
}
inline std::size_t nbytes(const Shape& shape, DType dtype) {
    return checked_mul(numel(shape), dtype_size(dtype));
}
inline Stride contiguous_stride(const Shape& shape) {
    std::vector<std::int64_t> stride(shape.rank());
    std::size_t running = 1;
    for (std::size_t axis = shape.rank(); axis-- > 0;) {
        stride[axis] = as_dimension(running);
        // Canonical strides stay positive for empty tensors; no broadcast strides.
        if (axis != 0) running = checked_mul(running, std::max<std::size_t>(as_size(shape[axis]), 1));
    }
    return Stride(std::move(stride));
}
// Bounding span, not logical payload size; zero strides are rejected by Tensor.
inline std::size_t storage_span_bytes(const Shape& shape, const Stride& stride, DType dtype) {
    const auto item_size = dtype_size(dtype);
    if (shape.rank() != stride.rank()) throw std::invalid_argument("shape/stride rank mismatch");
    if (numel(shape) == 0) return 0;
    std::size_t last = 0;
    for (std::size_t axis = 0; axis < shape.rank(); ++axis)
        last = checked_add(last, checked_mul(as_size(shape[axis]) - 1, as_size(stride[axis])));
    return checked_mul(checked_add(last, 1), item_size);
}
} // namespace runtime
