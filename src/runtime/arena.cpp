#include "runtime/arena.hpp"
#include <algorithm>
#include <stdexcept>

namespace runtime {
namespace {
std::size_t aligned_offset(std::size_t offset, std::size_t alignment) {
    const auto remainder = offset & (alignment - 1);
    return checked_add(offset, remainder ? alignment - remainder : 0);
}
}
Arena::Arena(std::size_t capacity_bytes, std::size_t alignment)
    : identity_(std::make_shared<int>(0)), storage_(Storage::allocate_cpu_aligned(capacity_bytes, alignment)), alignment_(alignment) {}
Block Arena::allocate(std::size_t bytes, std::size_t alignment) {
    ++counts_.requests;
    try {
        if (!alignment || (alignment & (alignment - 1)) != 0 || alignment > alignment_)
            throw std::invalid_argument("arena request alignment must be a power of two <= backing alignment");
        if (!bytes) return Block(0, 0, alignment, 0, identity_);
        const auto offset = aligned_offset(cursor_, alignment);
        const auto end = checked_add(offset, bytes);
        if (end > capacity()) throw ArenaExhausted();
        if (next_ticket_ == UINT64_MAX) throw std::overflow_error("arena ticket overflow");
        Block block(offset, bytes, alignment, next_ticket_, identity_);
        active_.emplace(next_ticket_, block); // commit only after all throwing metadata work
        ++next_ticket_;
        cursor_ = end;
        ++counts_.allocations; ++counts_.live_blocks;
        counts_.live_bytes += bytes;
        counts_.peak_live_bytes = std::max(counts_.peak_live_bytes, counts_.live_bytes);
        return block;
    } catch (...) { ++counts_.failures; throw; }
}
void Arena::free(const Block& block) {
    ++counts_.free_calls;
    if (block.owner_.lock() != identity_) throw std::invalid_argument("foreign/expired arena block");
    if (!block.size_) return; // arena-produced zero handles never reserve ranges
    const auto found = active_.find(block.ticket_);
    if (found == active_.end() || found->second.offset_ != block.offset_ || found->second.size_ != block.size_ ||
        found->second.alignment_ != block.alignment_) throw std::invalid_argument("stale/double free arena block");
    active_.erase(found);
    ++counts_.frees; --counts_.live_blocks; counts_.live_bytes -= block.size_;
}
void Arena::rewind() {
    if (!active_.empty()) throw std::logic_error("arena rewind requires no live blocks");
    cursor_ = 0;
}
void Arena::validate() const {
    if (cursor_ > capacity() || counts_.live_blocks != active_.size() || counts_.allocations - counts_.frees != active_.size())
        throw std::logic_error("arena accounting corruption");
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    std::size_t bytes = 0;
    for (const auto& entry : active_) {
        const auto& block = entry.second;
        if (block.owner_.lock() != identity_ || block.ticket_ != entry.first || !block.size_ ||
            block.offset_ % block.alignment_ != 0 || checked_add(block.offset_, block.size_) > capacity())
            throw std::logic_error("arena block range corruption");
        ranges.emplace_back(block.offset_, checked_add(block.offset_, block.size_));
        bytes = checked_add(bytes, block.size_);
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i - 1].second > ranges[i].first) throw std::logic_error("overlapping arena blocks");
    if (bytes != counts_.live_bytes || bytes > counts_.peak_live_bytes || counts_.peak_live_bytes > capacity())
        throw std::logic_error("arena byte accounting corruption");
}
} // namespace runtime
