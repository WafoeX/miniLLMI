#pragma once

#include "runtime/shape.hpp"
#include "runtime/storage.hpp"
#include <map>
#include <new>

namespace runtime {
class ArenaExhausted final : public std::bad_alloc {
public: const char* what() const noexcept override { return "arena capacity/fragmentation exhausted; no malloc fallback"; }
};
class Block {
public:
    Block() = default;
    std::size_t offset() const noexcept { return offset_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t alignment() const noexcept { return alignment_; }
    std::uint64_t ticket() const noexcept { return ticket_; }
private:
    friend class Arena;
    Block(std::size_t offset, std::size_t size, std::size_t alignment, std::uint64_t ticket,
          const std::shared_ptr<const void>& owner)
        : offset_(offset), size_(size), alignment_(alignment), ticket_(ticket), owner_(owner) {}
    std::size_t offset_ = 0, size_ = 0, alignment_ = 1;
    std::uint64_t ticket_ = 0;
    std::weak_ptr<const void> owner_;
};
struct ArenaCounts {
    std::size_t requests = 0, allocations = 0, free_calls = 0, frees = 0, failures = 0;
    std::size_t live_blocks = 0, live_bytes = 0, peak_live_bytes = 0, reused_allocations = 0;
};
class Arena {
public:
    explicit Arena(std::size_t capacity_bytes, std::size_t alignment = 64);
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Block allocate(std::size_t bytes, std::size_t alignment = 64);
    void free(const Block& block); // valid nonzero frees need no metadata allocation
    void rewind(); // no live blocks; resets bump cursor, not cumulative counters/tickets
    void validate() const; // independent diagnostic invariant checker, throws on corruption
    const std::shared_ptr<Storage>& storage() const noexcept { return storage_; }
    std::size_t capacity() const noexcept { return storage_->capacity_bytes(); }
    std::size_t alignment() const noexcept { return alignment_; }
    const ArenaCounts& counts() const noexcept { return counts_; }
private:
    std::shared_ptr<const void> identity_;
    std::shared_ptr<Storage> storage_;
    std::size_t alignment_, cursor_ = 0;
    std::uint64_t next_ticket_ = 1;
    std::map<std::uint64_t, Block> active_;
    ArenaCounts counts_;
};
} // namespace runtime
