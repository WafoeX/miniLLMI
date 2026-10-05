#include "runtime/tensor.hpp"
#include "runtime/copy.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace runtime {
namespace {
void validate_layout(const Shape& shape, const Stride& stride) {
    if (shape.rank() != stride.rank()) throw std::invalid_argument("shape/stride rank mismatch");
    for (auto value : stride.values())
        if (value == 0) throw std::invalid_argument("zero strides/broadcast views unsupported");
    if (runtime::numel(shape) == 0) return;
    // Conservative non-overlap proof: sorted axes must not intersect the span
    // of faster axes. Some disjoint exotic layouts are deliberately rejected.
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
} // namespace
Tensor::Tensor(std::shared_ptr<Storage> storage, DType dtype, Shape shape,
               Stride stride, std::size_t offset_bytes)
    : storage_(std::move(storage)), dtype_(dtype), shape_(std::move(shape)),
      stride_(std::move(stride)), offset_bytes_(offset_bytes) {
    if (!storage_) throw std::invalid_argument("tensor requires storage");
    const auto item_size = dtype_size(dtype_);
    validate_layout(shape_, stride_);
    if (offset_bytes_ % item_size != 0)
        throw std::invalid_argument("tensor offset must be a multiple of dtype size");
    if (storage_->data() && reinterpret_cast<std::uintptr_t>(storage_->data()) % dtype_alignment(dtype_) != 0)
        throw std::invalid_argument("storage pointer is not dtype-aligned");
    const auto end = checked_add(offset_bytes_, storage_span_bytes(shape_, stride_, dtype_));
    // Empty offset may equal capacity (one-past), but may never exceed it.
    if (end > storage_->capacity_bytes()) throw std::out_of_range("tensor range exceeds storage capacity");
}
Tensor Tensor::allocate_cpu(Shape shape, DType dtype) {
    const auto stride = contiguous_stride(shape);
    const auto count = runtime::numel(shape);
    auto storage = Storage::allocate_cpu(runtime::nbytes(shape, dtype));
    // Start typed object lifetimes explicitly in C++17; all elements are zero.
    if (count != 0) {
        switch (dtype) {
        case DType::FP32:
            std::uninitialized_value_construct_n(static_cast<float*>(storage->data()), count);
            break;
        case DType::INT32:
            std::uninitialized_value_construct_n(static_cast<std::int32_t*>(storage->data()), count);
            break;
        }
    }
    return Tensor(std::move(storage), dtype, std::move(shape), stride);
}
bool Tensor::is_contiguous() const {
    if (numel() == 0) return stride_ == contiguous_stride(shape_);
    std::size_t expected = 1;
    for (std::size_t axis = shape_.rank(); axis-- > 0;) {
        if (shape_[axis] > 1 && as_size(stride_[axis]) != expected) return false;
        expected = checked_mul(expected, as_size(shape_[axis]));
    }
    return true;
}
Tensor Tensor::reshape(Shape shape) const {
    if (!is_contiguous()) throw std::invalid_argument("reshape requires contiguous tensor");
    if (runtime::numel(shape) != numel()) throw std::invalid_argument("reshape cannot change numel");
    auto stride = contiguous_stride(shape);
    return Tensor(storage_, dtype_, std::move(shape), std::move(stride), offset_bytes_);
}
Tensor Tensor::view(Shape shape, Stride stride, std::size_t offset_bytes) const {
    return Tensor(storage_, dtype_, std::move(shape), std::move(stride),
                  checked_add(offset_bytes_, offset_bytes));
}
Tensor Tensor::narrow(std::size_t axis, std::int64_t start, std::int64_t length) const {
    return slice(axis, start, length, 1);
}
Tensor Tensor::slice(std::size_t axis, std::int64_t start, std::int64_t length, std::int64_t step) const {
    if (axis >= shape_.rank()) throw std::out_of_range("slice axis out of bounds");
    if (start < 0 || length < 0 || step <= 0) throw std::invalid_argument("invalid slice start/length/step");
    if (start > shape_[axis]) throw std::out_of_range("slice start out of bounds");
    if (length > 0) {
        const auto last = checked_add(as_size(start), checked_mul(as_size(length) - 1, as_size(step)));
        if (last >= as_size(shape_[axis])) throw std::out_of_range("slice range out of bounds");
    }
    auto dimensions = shape_.values();
    auto strides = stride_.values();
    dimensions[axis] = length;
    strides[axis] = as_dimension(checked_mul(as_size(strides[axis]), as_size(step)));
    Shape result_shape(std::move(dimensions));
    auto offset = offset_bytes_;
    // Empty slices keep the source offset, including end-of-axis slices of a
    // strided tensor. They must never manufacture an out-of-capacity address.
    if (runtime::numel(result_shape) != 0)
        offset = checked_add(offset, checked_mul(checked_mul(as_size(start), as_size(stride_[axis])), dtype_size(dtype_)));
    return Tensor(storage_, dtype_, std::move(result_shape), Stride(std::move(strides)), offset);
}
Tensor Tensor::permute(const std::vector<std::size_t>& axes) const {
    if (axes.size() != shape_.rank()) throw std::invalid_argument("permutation rank mismatch");
    std::vector<bool> seen(axes.size(), false);
    std::vector<std::int64_t> dimensions(axes.size()), strides(axes.size());
    for (std::size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] >= axes.size() || seen[axes[i]]) throw std::invalid_argument("invalid dimension permutation");
        seen[axes[i]] = true;
        dimensions[i] = shape_[axes[i]];
        strides[i] = stride_[axes[i]];
    }
    return Tensor(storage_, dtype_, Shape(std::move(dimensions)), Stride(std::move(strides)), offset_bytes_);
}
Tensor Tensor::transpose(std::size_t first, std::size_t second) const {
    if (first >= shape_.rank() || second >= shape_.rank()) throw std::out_of_range("transpose axis out of bounds");
    std::vector<std::size_t> axes(shape_.rank());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    std::swap(axes[first], axes[second]);
    return permute(axes);
}
Tensor Tensor::contiguous() const {
    if (device().type() != DeviceType::CPU) throw std::runtime_error("contiguous requires CPU storage in Stage 1");
    if (is_contiguous()) return *this;
    auto result = allocate_cpu(shape_, dtype_);
    copy_cpu(*this, result);
    return result;
}
void Tensor::check_access(DType requested) const {
    if (requested != dtype_) throw std::invalid_argument("tensor typed access dtype mismatch");
    if (device().type() != DeviceType::CPU)
        throw std::runtime_error("host tensor access requires CPU storage");
}
std::size_t Tensor::element_offset(const std::vector<std::int64_t>& indices) const {
    if (indices.size() != shape_.rank()) throw std::invalid_argument("index rank mismatch");
    std::size_t element = 0;
    for (std::size_t axis = 0; axis < shape_.rank(); ++axis) {
        if (indices[axis] < 0 || indices[axis] >= shape_[axis])
            throw std::out_of_range("tensor index out of bounds");
        element = checked_add(element, checked_mul(as_size(indices[axis]), as_size(stride_[axis])));
    }
    return checked_add(offset_bytes_, checked_mul(element, dtype_size(dtype_)));
}
} // namespace runtime
