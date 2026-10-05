#pragma once

#include "runtime/device.hpp"
#include <cstddef>
#include <functional>
#include <memory>

namespace runtime {
class Storage final {
public:
    using Deleter = std::function<void(void*)>;
    static std::shared_ptr<Storage> allocate_cpu(std::size_t capacity_bytes);
    // Arena backing, same Storage/hook ownership path. Power-of-two alignment
    // must be at least alignof(max_align_t); aligned delete matches aligned new.
    static std::shared_ptr<Storage> allocate_cpu_aligned(std::size_t capacity_bytes, std::size_t alignment);
    // Arguments are validated before ownership transfers. After validation, the
    // pointer is owned even if creating the shared control block throws.
    // Deleter must not throw and must capture everything it needs by ownership.
    static std::shared_ptr<Storage> wrap(Device device, std::size_t capacity_bytes,
                                         void* data, Deleter deleter);
    ~Storage() noexcept;
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) = delete;
    Storage& operator=(Storage&&) = delete;
    Device device() const noexcept { return device_; }
    std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
private:
    Storage(Device device, std::size_t capacity_bytes, void* data, Deleter deleter);
    const Device device_;
    const std::size_t capacity_bytes_;
    void* const data_;
    const Deleter deleter_;
};

#ifdef RUNTIME_TESTING
namespace testing {
struct AllocationCounts {
    std::size_t allocations;
    std::size_t frees;
    std::size_t live;
};
// Count nonzero CPU backing buffers, not metadata/control-block heap allocations.
AllocationCounts cpu_allocation_counts() noexcept;
void fail_next_cpu_allocation() noexcept;
} // namespace testing
#endif
} // namespace runtime
