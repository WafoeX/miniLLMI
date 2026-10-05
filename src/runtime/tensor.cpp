#include "runtime/tensor.hpp"
#include "runtime/copy.hpp"
#include "runtime/layout.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace runtime {
namespace {
Layout metadata(const Tensor& tensor) {
    return {tensor.shape(), tensor.stride(), tensor.dtype(), tensor.device(),
            tensor.storage()->capacity_bytes(), tensor.data_offset(), 0};
}
Tensor bind(const Tensor& source, const Layout& layout) {
    return Tensor(source.storage(), layout.dtype, layout.shape, layout.stride, layout.offset_bytes);
}
} // namespace
Tensor::Tensor(std::shared_ptr<Storage> storage, DType dtype, Shape shape,
               Stride stride, std::size_t offset_bytes)
    : storage_(std::move(storage)), dtype_(dtype), shape_(std::move(shape)),
      stride_(std::move(stride)), offset_bytes_(offset_bytes) {
    if (!storage_) throw std::invalid_argument("tensor requires storage");
    const auto item_size = dtype_size(dtype_);
    (void)item_size;
    metadata(*this).validate();
    if (storage_->data() && reinterpret_cast<std::uintptr_t>(storage_->data()) % dtype_alignment(dtype_) != 0)
        throw std::invalid_argument("storage pointer is not dtype-aligned");

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
bool Tensor::is_contiguous() const { return metadata(*this).is_contiguous(); }
Tensor Tensor::reshape(Shape shape) const { return bind(*this, metadata(*this).reshape(std::move(shape))); }
Tensor Tensor::view(Shape shape, Stride stride, std::size_t offset_bytes) const {
    return bind(*this, metadata(*this).view(std::move(shape), std::move(stride), offset_bytes));
}
Tensor Tensor::narrow(std::size_t axis, std::int64_t start, std::int64_t length) const {
    return slice(axis, start, length, 1);
}
Tensor Tensor::slice(std::size_t axis, std::int64_t start, std::int64_t length, std::int64_t step) const {
    return bind(*this, metadata(*this).slice(axis, start, length, step));
}
Tensor Tensor::permute(const std::vector<std::size_t>& axes) const {
    return bind(*this, metadata(*this).permute(axes));
}
Tensor Tensor::transpose(std::size_t first, std::size_t second) const {
    return bind(*this, metadata(*this).transpose(first, second));
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
