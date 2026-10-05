#pragma once

#include "runtime/device.hpp"
#include "runtime/shape.hpp"
#include <cstdint>

namespace runtime {
// Declarative metadata only: no pointer, buffer ownership, allocation or access.
// Shared by Tensor's checked views and pure operator/graph inference.
struct Layout {
    Shape shape;
    Stride stride;
    DType dtype = DType::FP32;
    Device device;
    std::size_t capacity_bytes = 0;
    std::size_t offset_bytes = 0;
    std::uint64_t storage_key = 0; // equality only; assigned by the caller
    void validate() const;
    bool is_contiguous() const;
    Layout reshape(Shape dimensions) const;
    Layout view(Shape dimensions, Stride steps, std::size_t offset = 0) const;
    Layout slice(std::size_t axis, std::int64_t start, std::int64_t length, std::int64_t step = 1) const;
    Layout permute(const std::vector<std::size_t>& axes) const;
    Layout transpose(std::size_t first, std::size_t second) const;
};
bool same_layout(const Layout& first, const Layout& second);
bool spans_overlap(const Layout& first, const Layout& second);
} // namespace runtime
