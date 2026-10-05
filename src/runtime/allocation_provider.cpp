#include "runtime/allocation_provider.hpp"
#include <memory>
#include <stdexcept>

namespace runtime {
Status ArenaAllocationProvider::begin() {
    if (running_ || arena_.storage().use_count() != 1)
        return Status::failure(StatusCode::InvalidArgument, "arena context busy: release all returned Tensor/alias handles or use a separate provider");
    for (const auto& entry : blocks_) arena_.free(entry.second);
    blocks_.clear();
    arena_.rewind();
    running_ = true;
    return Status::success();
}
Tensor ArenaAllocationProvider::allocate(TensorId root, Shape shape, DType dtype) {
    if (!running_ || blocks_.count(root)) throw std::logic_error("invalid arena provider session/root");
    const auto bytes = nbytes(shape, dtype);
    auto stride = contiguous_stride(shape);
    if (!bytes) return Tensor(arena_.storage(), dtype, std::move(shape), std::move(stride));
    const auto block = arena_.allocate(bytes, arena_.alignment());
    try {
        auto* pointer = static_cast<unsigned char*>(arena_.storage()->data()) + block.offset();
        // Explicit C++17 typed object lifetime; reused FP32/INT32 are trivial.
        if (dtype == DType::FP32) std::uninitialized_value_construct_n(reinterpret_cast<float*>(pointer), numel(shape));
        else std::uninitialized_value_construct_n(reinterpret_cast<std::int32_t*>(pointer), numel(shape));
        Tensor output(arena_.storage(), dtype, std::move(shape), std::move(stride), block.offset());
        blocks_.emplace(root, block);
        return output;
    } catch (...) { arena_.free(block); throw; }
}
void ArenaAllocationProvider::release(TensorId root) noexcept {
    const auto found = blocks_.find(root);
    if (found == blocks_.end()) return;
    arena_.free(found->second); // known-valid handle; reserved free-list metadata, no allocation
    blocks_.erase(found);
}
} // namespace runtime
