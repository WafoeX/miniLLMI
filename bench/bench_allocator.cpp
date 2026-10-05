#include "runtime/graph_executor.hpp"
#include "stage0/build_info.hpp"
#include "stage0/common.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <cmath>

namespace {
using namespace runtime;
using namespace stage0;
struct Metrics {
    std::size_t requests = 0, releases = 0, allocs = 0, frees = 0, peak = 0, live = 0, reused = 0;
    double checksum = 0;
};
struct GraphWork {
    Graph graph;
    std::array<float, 64> expected{};
    std::size_t external = 0;
};
GraphWork make_graph(bool diamond) {
    GraphWork work;
    auto x = Tensor::allocate_cpu({8, 8}), scale = Tensor::allocate_cpu({8, 8});
    for (std::size_t i = 0; i < 64; ++i) { x.data<float>()[i] = work.expected[i] = 0.125f + static_cast<float>(i) / 16; scale.data<float>()[i] = 0.5f; }
    work.graph.add_input(0, "x", x); work.graph.add_input(1, "scale", scale); work.external = 512;
    if (diamond) {
        auto identity = Tensor::allocate_cpu({8, 8}); for (std::size_t i = 0; i < 8; ++i) identity.data<float>()[i * 8 + i] = 1;
        work.graph.add_input(7, "identity", identity); work.external += 256;
        for (TensorId id = 2; id <= 6; ++id) work.graph.add_tensor(id, {8, 8});
        work.graph.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {2}));
        work.graph.add_node(1, OpDesc(OpCode::MUL, {2, 1}, {3}));
        work.graph.add_node(2, OpDesc(OpCode::ADD, {2, 0}, {4}));
        work.graph.add_node(3, OpDesc(OpCode::ADD, {3, 4}, {5}));
        work.graph.add_node(4, OpDesc(OpCode::MATMUL, {5, 7}, {6}));
        work.graph.add_output("output", 6); for (auto& value : work.expected) value *= 4;
    } else {
        for (TensorId id = 2; id < 14; ++id) {
            work.graph.add_tensor(id, {8, 8}); const auto code = id % 2 == 0 ? OpCode::MUL : OpCode::ADD;
            work.graph.add_node(id, OpDesc(code, {id == 2 ? 0u : id - 1, 1}, {id}));
            for (auto& value : work.expected) value = code == OpCode::MUL ? value * 0.5f : value + 0.5f;
        }
        work.graph.add_output("output", 13);
    }
    const auto status = work.graph.freeze(); if (!status.ok()) throw std::runtime_error(status.message);
    return work;
}
Metrics graph_once(const GraphWork& work, ArenaAllocationProvider* arena) {
    const auto reused = arena ? arena->arena().counts().reused_allocations : 0;
    const auto result = execute_graph(work.graph, nullptr, arena);
    if (!result.ok()) throw std::runtime_error(result.status.message);
    const auto& output = result.outputs.at("output");
    Metrics metrics{result.counts.allocation_requests, result.counts.releases, result.counts.allocations,
                    result.counts.frees, result.counts.peak_live_bytes, result.counts.live_bytes,
                    arena ? arena->arena().counts().reused_allocations - reused : 0, 0};
    for (std::size_t i = 0; i < 64; ++i) {
        const auto value = output.data<float>()[i];
        if (!std::isfinite(value) || std::abs(value - work.expected[i]) > 1e-6f + 1e-6f * std::abs(work.expected[i])) throw std::runtime_error("graph oracle mismatch");
        metrics.checksum += value;
    }
    return metrics; // outputs destroyed inside the same per-call timed boundary
}
Metrics synthetic_once(Arena* arena) {
    if (arena) arena->rewind();
    const auto reuse_before = arena ? arena->counts().reused_allocations : 0;
    Metrics metrics; std::size_t live = 0;
    std::array<std::shared_ptr<Storage>, 4> dynamic;
    std::array<std::optional<Block>, 4> blocks;
    std::array<std::size_t, 4> sizes{}; std::array<unsigned char*, 4> pointers{}; std::array<int, 4> values{};
    const auto allocate = [&](std::size_t slot, std::size_t bytes, int value) {
        if (arena) { blocks[slot] = arena->allocate(bytes); pointers[slot] = static_cast<unsigned char*>(arena->storage()->data()) + blocks[slot]->offset(); }
        else { dynamic[slot] = Storage::allocate_cpu_aligned(bytes, 64); pointers[slot] = static_cast<unsigned char*>(dynamic[slot]->data()); ++metrics.allocs; }
        std::memset(pointers[slot], value, bytes); sizes[slot] = bytes; values[slot] = value;
        ++metrics.requests; live += bytes; metrics.peak = std::max(metrics.peak, live);
    };
    const auto release = [&](std::size_t slot) {
        if (pointers[slot][0] != values[slot] || pointers[slot][sizes[slot] - 1] != values[slot]) throw std::runtime_error("synthetic live-byte corruption");
        metrics.checksum += pointers[slot][0] + pointers[slot][sizes[slot] - 1];
        if (arena) { arena->free(*blocks[slot]); blocks[slot].reset(); } else { dynamic[slot].reset(); ++metrics.frees; }
        ++metrics.releases; live -= sizes[slot];
    };
    for (unsigned round = 0; round < 8; ++round) {
        allocate(0, 64, 1); allocate(1, 128, 2); allocate(2, 192, 3); allocate(3, 256, 4);
        release(1); release(3); allocate(1, 128, 5); allocate(3, 256, 6);
        release(0); release(2); release(1); release(3);
    }
    if (metrics.checksum != 336 || live != 0) throw std::runtime_error("synthetic oracle mismatch");
    metrics.reused = arena ? arena->counts().reused_allocations - reuse_before : 0;
    return metrics;
}
void probe() {
    std::cout << "{\"schema_version\":1,\"commit\":" << json_quote(kCommit) << ",\"source_digest\":" << json_quote(kSourceDigest)
              << ",\"source_dirty\":" << (kSourceDirty ? "true" : "false") << ",\"build_type\":" << json_quote(kBuildType)
              << ",\"compiler\":" << json_quote(kCompiler) << ",\"testing\":"
#ifdef RUNTIME_TESTING
              << "true"
#else
              << "false"
#endif
              << ",\"backend\":\"cpu\"}\n";
}
void self_test() {
    Arena arena(1024); const auto a = synthetic_once(nullptr), b = synthetic_once(&arena);
    if (a.requests != 48 || b.requests != 48 || a.peak != 640 || b.peak != 640 || b.reused == 0) throw std::runtime_error("synthetic counters");
    for (auto diamond : {false, true}) {
        const auto work = make_graph(diamond); ArenaAllocationProvider provider(1024);
        const auto dynamic = graph_once(work, nullptr), reused = graph_once(work, &provider);
        if (dynamic.checksum != reused.checksum || dynamic.peak != reused.peak || dynamic.allocs != (diamond ? 5u : 12u) || reused.allocs != 0)
            throw std::runtime_error("graph counters/equivalence");
    }
    std::cout << "allocator benchmark workload correctness: PASS (untimed)\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--probe") { probe(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        if (std::string(kBuildType) != "Release" || kSourceDirty) throw std::runtime_error("benchmark requires clean Release source");
#ifdef RUNTIME_TESTING
        throw std::runtime_error("timing requires BUILD_TESTING=OFF; no backing test-hook overhead");
#endif
        std::string policy, workload; std::filesystem::path raw;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument("missing argument");
            const std::string key = argv[i]; if (key == "--policy") policy = argv[i + 1]; else if (key == "--workload") workload = argv[i + 1];
            else if (key == "--raw") raw = argv[i + 1]; else throw std::invalid_argument("unknown argument");
        }
        if ((policy != "dynamic" && policy != "arena") || (workload != "synthetic" && workload != "chain" && workload != "diamond") || raw.empty() || std::filesystem::exists(raw))
            throw std::invalid_argument("invalid policy/workload/new raw file");
        const bool use_arena = policy == "arena";
        std::unique_ptr<Arena> synthetic;
        std::unique_ptr<ArenaAllocationProvider> provider;
        std::optional<GraphWork> graph;
        if (workload == "synthetic") { if (use_arena) synthetic = std::make_unique<Arena>(1024); }
        else { graph = make_graph(workload == "diamond"); if (use_arena) provider = std::make_unique<ArenaAllocationProvider>(1024); }
        const auto once = [&] { return graph ? graph_once(*graph, provider.get()) : synthetic_once(synthetic.get()); };
        (void)once(); // independent scalar/value checks before timing
        for (int i = 0; i < 3; ++i) (void)once();
        const std::vector<std::string> columns{"sample", "ms", "batch", "requests", "releases", "backing_allocs", "backing_frees", "peak_payload_bytes", "output_live_bytes", "reused", "arena_capacity", "external_bytes", "peak_backing_bytes", "prepared_backing_allocs", "checksum", "correctness"};
        for (int sample = 0; sample < 10; ++sample) {
            Metrics metrics; const auto start = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < 20; ++repeat) metrics = once();
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - start).count() / 20;
            const auto external = graph ? graph->external : 0;
            Record row{{"sample", std::to_string(sample)}, {"ms", number(ms)}, {"batch", "20"}, {"requests", std::to_string(metrics.requests)}, {"releases", std::to_string(metrics.releases)},
                       {"backing_allocs", std::to_string(metrics.allocs)}, {"backing_frees", std::to_string(metrics.frees)}, {"peak_payload_bytes", std::to_string(metrics.peak)},
                       {"output_live_bytes", std::to_string(metrics.live)}, {"reused", std::to_string(metrics.reused)}, {"arena_capacity", use_arena ? "1024" : "0"},
                       {"external_bytes", std::to_string(external)}, {"peak_backing_bytes", std::to_string(external + (use_arena ? 1024 : metrics.peak))},
                       {"prepared_backing_allocs", use_arena ? "1" : "0"}, {"checksum", number(metrics.checksum)}, {"correctness", "passed"}};
            append_csv(raw, columns, row);
        }
        (void)once(); // recheck after all timed samples
        std::cout << workload << '/' << policy << ": PASS samples=10 warmup=3 batch=20\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "bench_allocator: " << e.what() << '\n'; return 1; }
}
