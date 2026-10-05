#pragma once

#include "runtime/shape.hpp"
#include "runtime/storage.hpp"
#include <memory>
#include <vector>

namespace runtime {
class Tensor {
public:
    Tensor(std::shared_ptr<Storage> storage, DType dtype, Shape shape,
           Stride stride, std::size_t offset_bytes = 0);
    // Value-initialized CPU elements. No implicit device allocation or copies.
    static Tensor allocate_cpu(Shape shape, DType dtype = DType::FP32);
    const Shape& shape() const noexcept { return shape_; }
    const Stride& stride() const noexcept { return stride_; }
    DType dtype() const noexcept { return dtype_; }
    Device device() const noexcept { return storage_->device(); }
    const std::shared_ptr<Storage>& storage() const noexcept { return storage_; }
    std::size_t data_offset() const noexcept { return offset_bytes_; }
    std::size_t numel() const { return runtime::numel(shape_); }
    std::size_t nbytes() const { return runtime::nbytes(shape_, dtype_); }
    bool is_contiguous() const;

    // Points at the first logical element, NOT a promise of contiguous layout.
    // Empty data() returns null; dtype/device checks still apply.
    template<class T> T* data() {
        check_access(dtype_of<T>());
        if (numel() == 0) return nullptr;
        return reinterpret_cast<T*>(static_cast<unsigned char*>(storage_->data()) + offset_bytes_);
    }
    template<class T> const T* data() const {
        check_access(dtype_of<T>());
        if (numel() == 0) return nullptr;
        return reinterpret_cast<const T*>(static_cast<const unsigned char*>(storage_->data()) + offset_bytes_);
    }
    template<class T> T& at(const std::vector<std::int64_t>& indices) {
        check_access(dtype_of<T>());
        return *reinterpret_cast<T*>(static_cast<unsigned char*>(storage_->data()) + element_offset(indices));
    }
    template<class T> const T& at(const std::vector<std::int64_t>& indices) const {
        check_access(dtype_of<T>());
        return *reinterpret_cast<const T*>(static_cast<const unsigned char*>(storage_->data()) + element_offset(indices));
    }
private:
    void check_access(DType requested) const;
    std::size_t element_offset(const std::vector<std::int64_t>& indices) const;
    std::shared_ptr<Storage> storage_;
    DType dtype_;
    Shape shape_;
    Stride stride_;
    std::size_t offset_bytes_;
};
} // namespace runtime
