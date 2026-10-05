#include "runtime/storage.hpp"
#include <atomic>
#include <new>
#include <stdexcept>
#include <utility>

namespace runtime {
#ifdef RUNTIME_TESTING
namespace {
std::atomic<std::size_t> allocations{0}, frees{0}, live{0};
std::atomic<bool> fail_next{false};
}
namespace testing {
AllocationCounts cpu_allocation_counts() noexcept {
    return {allocations.load(), frees.load(), live.load()};
}
void fail_next_cpu_allocation() noexcept { fail_next.store(true); }
}
#endif

Storage::Storage(Device device, std::size_t capacity_bytes, void* data, Deleter deleter)
    : device_(device), capacity_bytes_(capacity_bytes), data_(data), deleter_(std::move(deleter)) {}
Storage::~Storage() noexcept { deleter_(data_); }

std::shared_ptr<Storage> Storage::wrap(Device device, std::size_t capacity_bytes,
                                     void* data, Deleter deleter) {
    if ((capacity_bytes == 0) != (data == nullptr))
        throw std::invalid_argument("storage requires null iff capacity is zero");
    if (!deleter) throw std::invalid_argument("storage requires a deleter");
    // Also call the deleter exactly once for null/zero-byte wrapped storage.
    // A unique_ptr<void> skips null, so use a scope guard over the deleter itself.
    struct Guard {
        void* data;
        Deleter& deleter;
        bool active = true;
        ~Guard() noexcept { if (active) deleter(data); }
    } guard{data, deleter};
    auto* raw = new Storage(device, capacity_bytes, data, deleter);
    guard.active = false;
    return std::shared_ptr<Storage>(raw); // deletes raw if control-block allocation fails
}

std::shared_ptr<Storage> Storage::allocate_cpu(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) return wrap(Device{}, 0, nullptr, [](void*) noexcept {});
    Deleter deleter = [](void* pointer) noexcept {
        ::operator delete(pointer);
#ifdef RUNTIME_TESTING
        ++frees;
        --live;
#endif
    };
#ifdef RUNTIME_TESTING
    if (fail_next.exchange(false)) throw std::bad_alloc();
#endif
    void* pointer = ::operator new(capacity_bytes);
#ifdef RUNTIME_TESTING
    ++allocations;
    ++live;
#endif
    return wrap(Device{}, capacity_bytes, pointer, std::move(deleter));
}
} // namespace runtime
