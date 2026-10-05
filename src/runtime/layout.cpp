#include "runtime/layout.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace runtime {
void Layout::validate() const {
    const auto item_size = dtype_size(dtype);
    if (shape.rank() != stride.rank()) throw std::invalid_argument("shape/stride rank mismatch");
    for (auto value : stride.values())
        if (value == 0) throw std::invalid_argument("zero strides/broadcast views unsupported");
    if (numel(shape) != 0) {
        std::vector<std::size_t> axes;
        for (std::size_t axis = 0; axis < shape.rank(); ++axis)
            if (shape[axis] > 1) axes.push_back(axis);
        std::sort(axes.begin(), axes.end(), [&](auto a, auto b) { return stride[a] < stride[b]; });
        std::size_t span = 1;
        for (auto axis : axes) {
            const auto step = as_size(stride[axis]);
            if (step < span) throw std::invalid_argument("overlapping or unproven strided layout");
            span = checked_add(span, checked_mul(as_size(shape[axis]) - 1, step));
        }
    }
    if (offset_bytes % item_size != 0) throw std::invalid_argument("tensor offset must be a multiple of dtype size");
    if (checked_add(offset_bytes, storage_span_bytes(shape, stride, dtype)) > capacity_bytes)
        throw std::out_of_range("tensor range exceeds storage capacity");
}
bool Layout::is_contiguous() const {
    const bool empty = numel(shape) == 0;
    std::size_t expected = 1;
    for (std::size_t axis = shape.rank(); axis-- > 0;) {
        if ((empty || shape[axis] > 1) && as_size(stride[axis]) != expected) return false;
        if (axis != 0) {
            const auto dimension = std::max<std::size_t>(as_size(shape[axis]), 1);
            if (expected > std::numeric_limits<std::size_t>::max() / dimension) return false;
            expected *= dimension;
        }
    }
    return true;
}
Layout Layout::reshape(Shape dimensions) const {
    if (!is_contiguous()) throw std::invalid_argument("reshape requires contiguous tensor");
    if (numel(dimensions) != numel(shape)) throw std::invalid_argument("reshape cannot change numel");
    auto steps = contiguous_stride(dimensions);
    return view(std::move(dimensions), std::move(steps));
}
Layout Layout::view(Shape dimensions, Stride steps, std::size_t offset) const {
    Layout result{std::move(dimensions), std::move(steps), dtype, device, capacity_bytes,
                  checked_add(offset_bytes, offset), storage_key};
    result.validate();
    return result;
}
Layout Layout::slice(std::size_t axis, std::int64_t start, std::int64_t length, std::int64_t step) const {
    if (axis >= shape.rank()) throw std::out_of_range("slice axis out of bounds");
    if (start < 0 || length < 0 || step <= 0) throw std::invalid_argument("invalid slice start/length/step");
    if (start > shape[axis]) throw std::out_of_range("slice start out of bounds");
    if (length > 0 && checked_add(as_size(start), checked_mul(as_size(length) - 1, as_size(step))) >= as_size(shape[axis]))
        throw std::out_of_range("slice range out of bounds");
    auto dimensions = shape.values();
    auto steps = stride.values();
    dimensions[axis] = length;
    steps[axis] = as_dimension(checked_mul(as_size(steps[axis]), as_size(step)));
    Shape result_shape(std::move(dimensions));
    std::size_t offset = 0;
    if (numel(result_shape) != 0)
        offset = checked_mul(checked_mul(as_size(start), as_size(stride[axis])), dtype_size(dtype));
    return view(std::move(result_shape), Stride(std::move(steps)), offset);
}
Layout Layout::permute(const std::vector<std::size_t>& axes) const {
    if (axes.size() != shape.rank()) throw std::invalid_argument("permutation rank mismatch");
    std::vector<bool> seen(axes.size(), false);
    std::vector<std::int64_t> dimensions(axes.size()), steps(axes.size());
    for (std::size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] >= axes.size() || seen[axes[i]]) throw std::invalid_argument("invalid dimension permutation");
        seen[axes[i]] = true;
        dimensions[i] = shape[axes[i]];
        steps[i] = stride[axes[i]];
    }
    Layout result{Shape(std::move(dimensions)), Stride(std::move(steps)), dtype, device,
                  capacity_bytes, offset_bytes, storage_key};
    result.validate();
    return result;
}
Layout Layout::transpose(std::size_t first, std::size_t second) const {
    if (first >= shape.rank() || second >= shape.rank()) throw std::out_of_range("transpose axis out of bounds");
    std::vector<std::size_t> axes(shape.rank());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    std::swap(axes[first], axes[second]);
    return permute(axes);
}
bool same_layout(const Layout& a, const Layout& b) {
    return a.storage_key == b.storage_key && a.shape == b.shape && a.stride == b.stride &&
           a.dtype == b.dtype && a.device == b.device && a.offset_bytes == b.offset_bytes;
}
bool spans_overlap(const Layout& a, const Layout& b) {
    if (a.storage_key != b.storage_key || a.device != b.device || numel(a.shape) == 0 || numel(b.shape) == 0) return false;
    return a.offset_bytes < checked_add(b.offset_bytes, storage_span_bytes(b.shape, b.stride, b.dtype)) &&
           b.offset_bytes < checked_add(a.offset_bytes, storage_span_bytes(a.shape, a.stride, a.dtype));
}
} // namespace runtime
