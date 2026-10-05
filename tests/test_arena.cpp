#include "runtime/arena.hpp"
#include <iostream>
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
}
int main() {
    try { accounting(); require(testing::cpu_allocation_counts().live == 0, "arena tests leak"); std::cout << "arena accounting: PASS\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
