#include "runtime/arena.hpp"
#include "runtime/shape.hpp"
#include <algorithm>
#include <stdexcept>

namespace runtime {
namespace {
std::size_t aligned_offset(std::size_t offset, std::size_t alignment) {
    const auto remainder = offset & (alignment - 1);
    return checked_add(offset, remainder ? alignment - remainder : 0);
}
}
Arena::Arena(std::size_t capacity_bytes, std::size_t alignment, ArenaPolicy policy)
    : identity_(std::make_shared<int>(0)), storage_(Storage::allocate_cpu_aligned(capacity_bytes, alignment)), alignment_(alignment), policy_(policy) {
    if (policy != ArenaPolicy::FirstFit && policy != ArenaPolicy::BumpNoReuse) throw std::invalid_argument("invalid arena policy");
    free_.reserve(1);
    if (capacity_bytes && policy_ == ArenaPolicy::FirstFit) free_.push_back({0, capacity_bytes});
}
Block Arena::allocate(std::size_t bytes, std::size_t alignment) {
    ++counts_.requests;
    try {
        if (!alignment || (alignment & (alignment - 1)) != 0 || alignment > alignment_)
            throw std::invalid_argument("arena request alignment must be a power of two <= backing alignment");
        if (!bytes) return Block(0, 0, alignment, 0, identity_);
        auto offset = aligned_offset(cursor_, alignment);
        std::size_t index = 0;
        if (policy_ == ArenaPolicy::FirstFit) {
            for (; index < free_.size(); ++index) {
                offset = aligned_offset(free_[index].offset, alignment);
                if (checked_add(offset, bytes) <= checked_add(free_[index].offset, free_[index].bytes)) break;
            }
            if (index == free_.size()) throw ArenaExhausted();
            // Allocation is the only place free-list capacity may grow. This
            // guarantees valid free/coalesce never needs metadata allocation.
            free_.reserve(checked_add(active_.size(), 3));
        }
        const auto end = checked_add(offset, bytes);
        if (end > capacity()) throw ArenaExhausted();
        if (next_ticket_ == UINT64_MAX) throw std::overflow_error("arena ticket overflow");
        Block block(offset, bytes, alignment, next_ticket_, identity_);
        active_.emplace(next_ticket_, block); // commit only after all throwing metadata work
        ++next_ticket_;
        if (policy_ == ArenaPolicy::FirstFit) {
            const auto range = free_[index];
            const auto finish = range.offset + range.bytes;
            free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(index));
            if (range.offset < offset) free_.insert(free_.begin() + static_cast<std::ptrdiff_t>(index++), {range.offset, offset - range.offset});
            if (end < finish) free_.insert(free_.begin() + static_cast<std::ptrdiff_t>(index), {end, finish - end});
            if (offset < cursor_) ++counts_.reused_allocations;
        }
        cursor_ = std::max(cursor_, end);
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
    if (policy_ == ArenaPolicy::FirstFit) {
        const auto position = std::lower_bound(free_.begin(), free_.end(), block.offset_,
                                              [](const FreeRange& range, auto offset) { return range.offset < offset; });
        auto index = static_cast<std::size_t>(position - free_.begin());
        free_.insert(position, {block.offset_, block.size_});
        if (index && free_[index - 1].offset + free_[index - 1].bytes == free_[index].offset) {
            free_[index - 1].bytes += free_[index].bytes;
            free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(index--));
        }
        if (index + 1 < free_.size() && free_[index].offset + free_[index].bytes == free_[index + 1].offset) {
            free_[index].bytes += free_[index + 1].bytes;
            free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(index + 1));
        }
    }
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
    if (policy_ == ArenaPolicy::FirstFit) {
        for (std::size_t i = 0; i < free_.size(); ++i) {
            const auto& range = free_[i];
            if (!range.bytes || checked_add(range.offset, range.bytes) > capacity() ||
                (i && free_[i - 1].offset + free_[i - 1].bytes >= range.offset))
                throw std::logic_error("free list not sorted/disjoint/coalesced");
            ranges.emplace_back(range.offset, range.offset + range.bytes);
        }
        std::sort(ranges.begin(), ranges.end());
        std::size_t end = 0;
        for (const auto& range : ranges) {
            if (range.first != end) throw std::logic_error("arena partition gap/overlap");
            end = range.second;
        }
        if (end != capacity()) throw std::logic_error("arena partition capacity mismatch");
    }
}
} // namespace runtime
