#include "runtime/arena.hpp"
#include <iostream>
#include <random>
#include <cstring>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class E, class F> void rejects(F call) {
    try { call(); } catch (const E&) { return; }
    throw std::runtime_error("expected arena failure");
}
void accounting() {
    const auto before = testing::cpu_allocation_counts();
    {
        Arena arena(129);
        require(reinterpret_cast<std::uintptr_t>(arena.storage()->data()) % 64 == 0, "backing alignment");
        const auto zero = arena.allocate(0);
        arena.free(zero);
        auto first = arena.allocate(17, 16), second = arena.allocate(33, 64);
        require(first.offset() == 0 && second.offset() == 64, "aligned offsets");
        require(arena.counts().live_bytes == 50 && arena.counts().peak_live_bytes == 50, "exact payload bytes exclude padding");
        rejects<std::invalid_argument>([&] { (void)arena.allocate(1, 3); });
        rejects<std::invalid_argument>([&] { (void)arena.allocate(1, 128); });
        rejects<std::overflow_error>([&] { (void)arena.allocate(SIZE_MAX, 64); });
        rejects<ArenaExhausted>([&] { (void)arena.allocate(33, 64); });
        require(arena.counts().live_bytes == 50 && arena.counts().failures == 4, "failed allocations preserve accounting");
        arena.validate();
        Arena foreign(64);
        rejects<std::invalid_argument>([&] { foreign.free(first); });
        rejects<std::invalid_argument>([&] { arena.free(Block{}); });
        rejects<std::logic_error>([&] { arena.rewind(); });
        arena.free(first); rejects<std::invalid_argument>([&] { arena.free(first); });
        arena.free(second); arena.rewind();
        const auto full = arena.allocate(129);
        require(full.offset() == 0 && full.ticket() != first.ticket(), "rewind retains stale-handle identity");
        arena.free(full); arena.validate();
        require(arena.counts().allocations == 3 && arena.counts().frees == 3 && arena.counts().live_bytes == 0, "exact allocate/free counts");
    }
    require(testing::cpu_allocation_counts().live == before.live, "aligned Storage deleted exactly once");
    Arena empty(0); empty.free(empty.allocate(0)); empty.validate();
    rejects<ArenaExhausted>([&] { (void)empty.allocate(1); });
    rejects<std::invalid_argument>([] { Arena bad(64, 3); });
    testing::fail_next_cpu_allocation(); rejects<std::bad_alloc>([] { Arena failed(64); });
    Block expired;
    { Arena owner(64); expired = owner.allocate(1); }
    Arena replacement(64); rejects<std::invalid_argument>([&] { replacement.free(expired); });
}
void reuse() {
    Arena arena(256);
    auto a = arena.allocate(64), b = arena.allocate(64), c = arena.allocate(64), d = arena.allocate(64);
    arena.free(a); arena.free(c);
    rejects<ArenaExhausted>([&] { (void)arena.allocate(128); }); // aggregate free != contiguous fit
    arena.free(b);
    const auto large = arena.allocate(128);
    require(large.offset() == 0, "first fit coalesced prefix");
    rejects<std::invalid_argument>([&] { arena.free(a); }); // same address, stale ticket
    arena.free(large); arena.free(d); arena.validate();
    require(arena.free_ranges().size() == 1 && arena.free_ranges()[0].bytes == 256 && arena.counts().reused_allocations == 1, "fully freed arena coalesces to one range");
    Arena gaps(129); const auto small = gaps.allocate(20, 1), aligned = gaps.allocate(23, 64), gap = gaps.allocate(4, 4);
    require(aligned.offset() == 64 && gap.offset() == 20, "alignment prefix preserved and reused");
    gaps.free(aligned); gaps.free(small); gaps.free(gap); gaps.validate();
    Arena bump(128, 64, ArenaPolicy::BumpNoReuse);
    const auto x = bump.allocate(64), y = bump.allocate(64); bump.free(x);
    rejects<ArenaExhausted>([&] { (void)bump.allocate(64); });
    bump.free(y); bump.validate(); bump.rewind();
    const auto reset = bump.allocate(128); bump.free(reset);
    require(bump.counts().reused_allocations == 0, "preserved bump no-reuse baseline");
}
void randomized() {
    for (unsigned seed = 0; seed < 6; ++seed) {
        std::mt19937 random(0x5402 + seed); Arena arena(257); std::vector<Block> active;
        const auto before = testing::cpu_allocation_counts();
        for (unsigned step = 0; step < 3000; ++step) {
            if (!active.empty() && random() % 3 == 0) {
                const auto index = random() % active.size(); arena.free(active[index]); active.erase(active.begin() + index);
            } else {
                try {
                    const auto block = arena.allocate(random() % 56, std::size_t{1} << (random() % 7));
                    if (!block.size()) arena.free(block);
                    else {
                        std::memset(static_cast<unsigned char*>(arena.storage()->data()) + block.offset(), static_cast<int>(block.ticket() % 251 + 1), block.size());
                        active.push_back(block);
                    }
                } catch (const ArenaExhausted&) {}
            }
            arena.validate();
            std::vector<bool> covered(257, false); std::size_t payload = 0;
            for (const auto& block : active) {
                payload += block.size();
                require(block.offset() % block.alignment() == 0, "random alignment");
                for (std::size_t i = block.offset(); i < block.offset() + block.size(); ++i) {
                    require(!covered[i], "independent bitmap allocation overlap"); covered[i] = true;
                    require(static_cast<unsigned char*>(arena.storage()->data())[i] == block.ticket() % 251 + 1, "live bytes corrupted by reuse");
                }
            }
            require(payload == arena.counts().live_bytes && active.size() == arena.counts().live_blocks, "independent bitmap accounting");
            for (const auto& range : arena.free_ranges()) for (std::size_t i = range.offset; i < range.offset + range.bytes; ++i) {
                require(!covered[i], "free/live bitmap overlap"); covered[i] = true;
            }
            for (auto bit : covered) require(bit, "independent full capacity partition");
        }
        for (const auto& block : active) arena.free(block);
        arena.validate(); require(arena.free_ranges().size() == 1 && arena.free_ranges()[0].bytes == 257, "random final coalescing");
        require(testing::cpu_allocation_counts().allocations == before.allocations, "all requests reuse one backing buffer");
    }
    std::cout << "arena properties: PASS seeds=6 operations=18000\n";
}
}
int main() {
    try { accounting(); reuse(); randomized(); require(testing::cpu_allocation_counts().live == 0, "arena tests leak"); std::cout << "arena accounting/reuse: PASS\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
