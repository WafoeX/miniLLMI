#include "runtime/copy.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace runtime {
namespace {
std::size_t address_begin(const Tensor& tensor) {
    return checked_add(reinterpret_cast<std::uintptr_t>(tensor.storage()->data()), tensor.data_offset());
}
std::size_t byte_offset(const Tensor& tensor, const std::vector<std::size_t>& indices) {
    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < indices.size(); ++axis)
        offset = checked_add(offset, checked_mul(indices[axis], as_size(tensor.stride()[axis])));
    return checked_add(tensor.data_offset(), checked_mul(offset, dtype_size(tensor.dtype())));
}
} // namespace
bool same_tensor_layout(const Tensor& first, const Tensor& second) {
    return first.storage() == second.storage() && first.dtype() == second.dtype() &&
           first.shape() == second.shape() && first.stride() == second.stride() &&
           first.data_offset() == second.data_offset();
}
bool memory_spans_overlap(const Tensor& first, const Tensor& second) {
    if (first.device() != second.device() || first.numel() == 0 || second.numel() == 0) return false;
    const auto first_begin = address_begin(first);
    const auto second_begin = address_begin(second);
    const auto first_end = checked_add(first_begin, storage_span_bytes(first.shape(), first.stride(), first.dtype()));
    const auto second_end = checked_add(second_begin, storage_span_bytes(second.shape(), second.stride(), second.dtype()));
    return first_begin < second_end && second_begin < first_end;
}
std::size_t copy_cpu(const Tensor& source, Tensor& destination) {
    if (source.device().type() != DeviceType::CPU || destination.device().type() != DeviceType::CPU)
        throw std::runtime_error("copy_cpu requires CPU source and destination");
    if (source.dtype() != destination.dtype() || source.shape() != destination.shape())
        throw std::invalid_argument("copy requires identical dtype and shape");
    if (source.numel() == 0) return 0;
    if (same_tensor_layout(source, destination)) return 0;
    if (memory_spans_overlap(source, destination))
        throw std::invalid_argument("copy of overlapping storage spans unsupported");
    const auto* input = static_cast<const unsigned char*>(source.storage()->data());
    auto* output = static_cast<unsigned char*>(destination.storage()->data());
    if (source.is_contiguous() && destination.is_contiguous()) {
        std::memcpy(output + destination.data_offset(), input + source.data_offset(), source.nbytes());
    } else {
        std::vector<std::size_t> indices(source.shape().rank(), 0);
        for (std::size_t element = 0; element < source.numel(); ++element) {
            std::memcpy(output + byte_offset(destination, indices), input + byte_offset(source, indices), dtype_size(source.dtype()));
            for (std::size_t axis = indices.size(); axis-- > 0;) {
                if (++indices[axis] < as_size(source.shape()[axis])) break;
                indices[axis] = 0;
            }
        }
    }
    return source.nbytes();
}
} // namespace runtime
