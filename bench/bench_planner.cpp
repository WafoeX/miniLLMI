#include "runtime/planned_executor.hpp"
#include "stage0/build_info.hpp"
#include "stage0/common.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>

namespace {
using namespace runtime;
using namespace stage0;
using Clock = std::chrono::steady_clock;
struct Work {
    Graph graph;
    std::array<float, 64> expected{};
    std::size_t external = 512, requests = 12, peak = 512, no_reuse = 3072, reuse = 512, reused = 10;
    std::size_t intermediate_roots = 0, output_roots = 0, peak_intermediate = 0;
};
Work workload(bool diamond) {
    Work work;
    auto x = Tensor::allocate_cpu({8, 8}), scale = Tensor::allocate_cpu({8, 8});
    for (std::size_t i = 0; i < 64; ++i) { x.data<float>()[i] = work.expected[i] = 0.125f + static_cast<float>(i) / 16; scale.data<float>()[i] = 0.5f; }
    work.graph.add_input(0, "x", x); work.graph.add_input(1, "scale", scale);
    if (diamond) {
        work.external = 768; work.requests = 5; work.peak = 768; work.no_reuse = 1280; work.reuse = 768; work.reused = 2;
        auto identity = Tensor::allocate_cpu({8, 8});
        for (std::size_t i = 0; i < 8; ++i) identity.data<float>()[i * 8 + i] = 1;
        work.graph.add_input(7, "identity", identity);
        for (TensorId id = 2; id <= 6; ++id) work.graph.add_tensor(id, {8, 8});
        work.graph.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {2}));
        work.graph.add_node(1, OpDesc(OpCode::MUL, {2, 1}, {3}));
        work.graph.add_node(2, OpDesc(OpCode::ADD, {2, 0}, {4}));
        work.graph.add_node(3, OpDesc(OpCode::ADD, {3, 4}, {5}));
        work.graph.add_node(4, OpDesc(OpCode::MATMUL, {5, 7}, {6}));
        work.graph.add_output("output", 6);
        for (auto& value : work.expected) value *= 4;
    } else {
        for (TensorId id = 2; id < 14; ++id) {
            work.graph.add_tensor(id, {8, 8}); const auto code = id % 2 == 0 ? OpCode::MUL : OpCode::ADD;
            work.graph.add_node(id, OpDesc(code, {id == 2 ? 0u : id - 1, 1}, {id}));
            for (auto& value : work.expected) value = code == OpCode::MUL ? value * 0.5f : value + 0.5f;
        }
        work.graph.add_output("output", 13);
    }
    const auto status = work.graph.freeze(); if (!status.ok()) throw std::runtime_error(status.message);
    const auto life = analyze_lifetimes(work.graph);
    std::set<TensorId> output_roots;
    for (const auto& output : work.graph.outputs()) output_roots.insert(*work.graph.tensors().at(output.second).base);
    for (const auto& item : life.roots) if (!item.second.external && item.second.bytes) {
        if (output_roots.count(item.first)) ++work.output_roots; else ++work.intermediate_roots;
    }
    for (std::size_t node = 0; node < life.node_count; ++node) {
        std::size_t live = 0;
        for (const auto& item : life.roots) if (!item.second.external && !output_roots.count(item.first) &&
            item.second.birth <= static_cast<std::int64_t>(node) && static_cast<std::int64_t>(node) <= item.second.last_use) live = checked_add(live, item.second.bytes);
        work.peak_intermediate = std::max(work.peak_intermediate, live);
    }
    std::size_t external_bytes = 0; std::set<const Storage*> storages;
    for (const auto& item : work.graph.tensors()) if (item.second.external && storages.insert(item.second.external->storage().get()).second)
        external_bytes = checked_add(external_bytes, item.second.external->storage()->capacity_bytes());
    if (external_bytes != work.external || work.peak_intermediate != work.peak || work.intermediate_roots != work.requests - 1 || work.output_roots != 1) throw std::runtime_error("workload lifetime accounting mismatch");
    work.external = external_bytes;
    return work;
}
std::unique_ptr<PlannedAllocationProvider> prepare(const Work& work, const std::string& policy) {
    if (policy == "dynamic") return nullptr;
    return std::make_unique<PlannedAllocationProvider>(work.graph, policy == "reuse" ? PlanPolicy::Reuse : PlanPolicy::NoReuse);
}
struct Metrics { ExecutionCounts counts; double checksum; };
Metrics once(const Work& work, PlannedAllocationProvider* provider) {
    const auto result = execute_graph(work.graph, nullptr, provider);
    if (!result.ok()) throw std::runtime_error(result.status.message);
    const auto& output = result.outputs.at("output");
    double checksum = 0;
    for (std::size_t i = 0; i < 64; ++i) {
        const auto value = output.data<float>()[i];
        if (!std::isfinite(value) || std::abs(value - work.expected[i]) > 1e-6f + 1e-6f * std::abs(work.expected[i])) throw std::runtime_error("planner workload oracle mismatch");
        checksum += value;
    }
    const auto& c = result.counts;
    if (c.allocation_requests != work.requests || c.releases != work.requests - 1 || c.peak_live_bytes != work.peak || c.live_bytes != 256 ||
        c.allocations != (provider ? 0 : work.requests) || c.frees != (provider ? 0 : work.requests - 1)) throw std::runtime_error("planner execute counter mismatch");
    return {c, checksum}; // output destruction included in per-call timed boundary
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
              << ",\"backend\":\"cpu\",\"benchmark\":\"planner-v1\"}\n";
}
void self_test() {
    for (bool diamond : {false, true}) {
        const auto work = workload(diamond);
        double checksum = 0;
        for (const auto& policy : {"dynamic", "no_reuse", "reuse"}) {
            auto provider = prepare(work, policy);
            if (provider && (provider->capacity() != (std::string(policy) == "reuse" ? work.reuse : work.no_reuse) ||
                provider->plan().reuse_count != (std::string(policy) == "reuse" ? work.reused : 0))) throw std::runtime_error("planner prepare counter mismatch");
#ifdef RUNTIME_TESTING
            const auto before = testing::cpu_allocation_counts();
#endif
            const auto m = once(work, provider.get());
#ifdef RUNTIME_TESTING
            const auto after = testing::cpu_allocation_counts();
            if (after.allocations - before.allocations != (provider ? 0 : work.requests) ||
                after.frees - before.frees != (provider ? 0 : work.requests) || after.live != before.live) throw std::runtime_error("real Storage hook mismatch including output destruction");
#endif
            if (checksum && checksum != m.checksum) throw std::runtime_error("policy checksum mismatch");
            checksum = m.checksum;
        }
    }
    std::cout << "planner benchmark workload correctness: PASS (untimed)\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--probe") { probe(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        if (std::string(kBuildType) != "Release" || kSourceDirty) throw std::runtime_error("timing requires clean Release source");
#ifdef RUNTIME_TESTING
        throw std::runtime_error("timing requires BUILD_TESTING=OFF");
#endif
        std::string policy, name; std::filesystem::path raw, prepare_raw;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument("missing argument");
            const std::string key = argv[i];
            if (key == "--policy") policy = argv[i + 1]; else if (key == "--workload") name = argv[i + 1];
            else if (key == "--raw") raw = argv[i + 1]; else if (key == "--prepare-raw") prepare_raw = argv[i + 1];
            else throw std::invalid_argument("unknown argument");
        }
        if ((policy != "dynamic" && policy != "no_reuse" && policy != "reuse") || (name != "chain" && name != "diamond") ||
            raw.empty() || prepare_raw.empty() || raw == prepare_raw || std::filesystem::exists(raw) || std::filesystem::exists(prepare_raw)) throw std::invalid_argument("invalid policy/workload/new raw files");
        const auto work = workload(name == "diamond");
        for (int warmup = 0; warmup < 3; ++warmup) { auto temporary = prepare(work, policy); }
        for (int sample = 0; sample < 10; ++sample) {
            const auto start = Clock::now(); auto temporary = prepare(work, policy); const auto stop = Clock::now();
            const auto ms = temporary ? std::chrono::duration<double, std::milli>(stop - start).count() : 0;
            append_csv(prepare_raw, {"sample", "ms", "batch", "arena_capacity", "prepared_backing_allocs", "correctness"},
                {{"sample", std::to_string(sample)}, {"ms", number(ms)}, {"batch", "1"}, {"arena_capacity", std::to_string(temporary ? temporary->capacity() : 0)}, {"prepared_backing_allocs", temporary ? "1" : "0"}, {"correctness", "passed"}});
        }
        auto provider = prepare(work, policy);
        const auto capacity = provider ? provider->capacity() : 0;
        (void)once(work, provider.get());
        for (int warmup = 0; warmup < 3; ++warmup) (void)once(work, provider.get());
        const std::vector<std::string> columns{"sample", "ms", "batch", "requests", "releases", "backing_allocs", "backing_frees", "intermediate_backing_allocs", "output_destroy_backing_frees", "peak_payload_bytes", "peak_intermediate_bytes", "output_live_bytes", "reused", "inplace", "arena_capacity", "external_bytes", "peak_resident_bytes", "prepared_backing_allocs", "checksum", "correctness"};
        for (int sample = 0; sample < 10; ++sample) {
            Metrics m{}; const auto start = Clock::now();
            for (int repeat = 0; repeat < 20; ++repeat) m = once(work, provider.get());
            const auto stop = Clock::now(); const auto& c = m.counts;
            append_csv(raw, columns, {{"sample", std::to_string(sample)}, {"ms", number(std::chrono::duration<double, std::milli>(stop - start).count() / 20)}, {"batch", "20"},
                {"requests", std::to_string(c.allocation_requests)}, {"releases", std::to_string(c.releases)}, {"backing_allocs", std::to_string(c.allocations)}, {"backing_frees", std::to_string(c.frees)},
                {"intermediate_backing_allocs", std::to_string(provider ? 0 : work.intermediate_roots)}, {"output_destroy_backing_frees", std::to_string(provider ? 0 : work.output_roots)},
                {"peak_payload_bytes", std::to_string(c.peak_live_bytes)}, {"peak_intermediate_bytes", std::to_string(work.peak_intermediate)}, {"output_live_bytes", std::to_string(c.live_bytes)},
                {"reused", std::to_string(provider ? provider->plan().reuse_count : 0)}, {"inplace", "0"}, {"arena_capacity", std::to_string(capacity)},
                {"external_bytes", std::to_string(work.external)}, {"peak_resident_bytes", std::to_string(work.external + (provider ? capacity : c.peak_live_bytes))}, {"prepared_backing_allocs", provider ? "1" : "0"},
                {"checksum", number(m.checksum)}, {"correctness", "passed"}});
        }
        (void)once(work, provider.get());
        std::cout << name << '/' << policy << ": PASS execute/prepare samples=10 warmup=3 execute batch=20\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "bench_planner: " << e.what() << '\n'; return 1; }
}
