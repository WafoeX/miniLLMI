#include "runtime/cpu_backend.hpp"
#include "runtime/planned_executor.hpp"
#include "stage0/build_info.hpp"
#include "stage0/common.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>

namespace {
using namespace runtime;
using Clock = std::chrono::steady_clock;
const CpuBackend backend(CpuMatmul::ScalarFP32V0);
constexpr int warmups = 3, samples = 10;
constexpr double kAtol = 1e-3, kRtol = 1e-3;
volatile float observable = 0;
void require(bool c, const char* m) { if (!c) throw std::runtime_error(m); }
void success(const Status& s) { if (!s.ok()) throw std::runtime_error(s.message); }
struct Work {
    stage0::Shape shape;
    Tensor a, b, out;
    std::vector<float> oracle;
    std::string hash;
    Graph graph;
    explicit Work(stage0::Shape s) : shape(s), a(Tensor::allocate_cpu({s.m, s.k})),
        b(Tensor::allocate_cpu({s.k, s.n})), out(Tensor::allocate_cpu({s.m, s.n})), oracle(out.numel()) {
        std::vector<float> av(a.numel()), bv(b.numel());
        stage0::random_inputs(av, bv, 42); hash = stage0::input_hash(av, bv);
        if (!av.empty()) std::copy(av.begin(), av.end(), a.data<float>());
        if (!bv.empty()) std::copy(bv.begin(), bv.end(), b.data<float>());
        // Stage 0 intentionally requires positive dimensions. Empty/K=0
        // correctness uses the independently known empty/zero result; all timed
        // shapes use the unchanged Stage 0 FP64 oracle, NEVER timed.
        if (s.m > 0 && s.n > 0 && s.k > 0) stage0::cpu_reference(av.data(), bv.data(), oracle.data(), s);
    }
    void make_graph() {
        graph.add_input(0, "a", a); graph.add_input(1, "b", b);
        for (TensorId id = 2; id <= 4; ++id) graph.add_tensor(id, {shape.m, shape.n});
        graph.add_node(0, OpDesc(OpCode::MATMUL, {0, 1}, {2}));
        graph.add_node(1, OpDesc(OpCode::ADD, {2, 2}, {3}));
        graph.add_node(2, OpDesc(OpCode::MUL, {3, 3}, {4}));
        graph.add_output("result", 4); success(graph.freeze());
        for (auto& value : oracle) { value = value + value; value = value * value; }
    }
};
const OpDesc mm(OpCode::MATMUL, {0, 1}, {2});
void gemm(Work& w, const TensorInputs& inputs) { success(backend.execute(mm, inputs, w.out)); }
float graph_once(const Work& w, AllocationProvider* provider) {
    const auto result = execute_graph(w.graph, nullptr, provider, &backend);
    success(result.status);
    require(result.counts.nodes_completed == 3 && result.counts.allocations == (provider ? 0u : 3u), "graph backend/provider counter mismatch");
    const auto& out = result.outputs.at("result");
    for (std::size_t i = 0; i < out.numel(); ++i) {
        const double value = out.data<float>()[i], expected = w.oracle[i];
        require(std::isfinite(value) && std::abs(value - expected) <= kAtol + kRtol * std::abs(expected), "timed graph oracle mismatch");
    }
    return out.data<float>()[0]; // full oracle scan + output destruction included in graph timer
}
std::vector<float> values(const Tensor& t) {
    std::vector<float> result(t.numel());
    if (t.numel()) std::copy_n(t.data<float>(), t.numel(), result.data());
    return result;
}
stage0::ErrorMetrics check(Work& w, const std::string& workload, AllocationProvider* provider, const TensorInputs& inputs) {
    if (workload == "gemm") {
        gemm(w, inputs);
        if (!w.out.numel()) { require(w.oracle.empty(), "empty oracle size mismatch"); return {}; }
        return stage0::compare(values(w.out), w.oracle, kAtol, kRtol);
    }
    const auto result = execute_graph(w.graph, nullptr, provider, &backend);
    success(result.status); return stage0::compare(values(result.outputs.at("result")), w.oracle, kAtol, kRtol);
}
std::string metrics(const stage0::ErrorMetrics& m) {
    return "{\"max_abs\":" + stage0::number(m.max_abs) + ",\"mean_abs\":" + stage0::number(m.mean_abs) +
        ",\"max_relative\":" + stage0::number(m.max_relative) + ",\"violations\":" + std::to_string(m.violations) +
        ",\"nonfinite\":" + std::to_string(m.nonfinite) + ",\"passed\":" + (m.passed() ? "true" : "false") + "}";
}
void probe() {
    std::cout << "{\"schema_version\":1,\"commit\":" << stage0::json_quote(stage0::kCommit)
        << ",\"source_digest\":" << stage0::json_quote(stage0::kSourceDigest) << ",\"source_dirty\":" << (stage0::kSourceDirty ? "true" : "false")
        << ",\"build_type\":" << stage0::json_quote(stage0::kBuildType) << ",\"compiler\":" << stage0::json_quote(stage0::kCompiler)
        << ",\"testing\":"
#ifdef RUNTIME_TESTING
        << "true"
#else
        << "false"
#endif
        << ",\"backend\":\"cpu\",\"benchmark\":\"cpu-v1\",\"algorithm\":\"cpu-ijk-fp32-v0\",\"threads\":1}\n";
}
void self_test() {
    for (auto s : {stage0::Shape{1,1,1}, {0,3,2}, {2,0,3}, {2,3,0}, {3,5,7}, {31,33,32}, {33,31,65}, {65,63,64}}) {
        Work w(s); const TensorInputs inputs{w.a, w.b};
        for (std::size_t i = 0; i < w.out.numel(); ++i) w.out.data<float>()[i] = std::numeric_limits<float>::quiet_NaN();
#ifdef RUNTIME_TESTING
        const auto before = testing::cpu_allocation_counts();
#endif
        require(check(w, "gemm", nullptr, inputs).passed(), "CPU scalar boundary oracle mismatch");
#ifdef RUNTIME_TESTING
        require(testing::cpu_allocation_counts().allocations == before.allocations, "GEMM dispatch allocated backing");
#endif
    }
    Work w({16,16,16}); w.make_graph(); const TensorInputs inputs{w.a, w.b};
    require(check(w, "graph", nullptr, inputs).passed(), "dynamic graph oracle mismatch");
    PlannedAllocationProvider prepared(w.graph);
    require(check(w, "graph", &prepared, inputs).passed(), "prepared graph oracle mismatch");
    (void)graph_once(w, &prepared);
    std::cout << "CPU benchmark correctness: PASS eight zero/small/non-square/boundary shapes and graph policies (untimed)\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--probe") { probe(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        require(std::string(stage0::kBuildType) == "Release" && !stage0::kSourceDirty, "timing requires committed clean Release source");
#ifdef RUNTIME_TESTING
        throw std::runtime_error("timing requires BUILD_TESTING=OFF");
#endif
        std::string workload, policy, size; std::filesystem::path raw, correctness;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument("missing value");
            const std::string key = argv[i];
            if (key == "--workload") workload = argv[i+1]; else if (key == "--policy") policy = argv[i+1];
            else if (key == "--size") size = argv[i+1]; else if (key == "--raw") raw = argv[i+1];
            else if (key == "--correctness") correctness = argv[i+1]; else throw std::invalid_argument("unknown argument");
        }
        require(!raw.empty() && !correctness.empty() && raw != correctness && !std::filesystem::exists(raw) && !std::filesystem::exists(correctness), "must specify distinct new raw/correctness files");
        const bool is_gemm = workload == "gemm";
        require(is_gemm ? (policy == "caller" && (size == "128" || size == "256" || size == "512" || size == "1024")) :
            (workload == "graph" && size == "16" && (policy == "dynamic" || policy == "reuse")), "invalid frozen workload/policy/size");
        const int n = std::stoi(size), batch = is_gemm ? 1 : 20;
        Work w({n,n,n}); if (!is_gemm) w.make_graph();
        std::unique_ptr<PlannedAllocationProvider> provider;
        if (policy == "reuse") provider = std::make_unique<PlannedAllocationProvider>(w.graph);
        const TensorInputs inputs{w.a, w.b};
        for (std::size_t i = 0; i < w.out.numel(); ++i) w.out.data<float>()[i] = std::numeric_limits<float>::quiet_NaN();
        const auto initial = check(w, workload, provider.get(), inputs);
        const std::string prefix = "{\"schema_version\":1,\"workload\":" + stage0::json_quote(workload) + ",\"policy\":" + stage0::json_quote(policy) +
            ",\"algorithm\":\"cpu-ijk-fp32-v0\",\"m\":" + size + ",\"n\":" + size + ",\"k\":" + size + ",\"seed\":42,\"input_hash\":" + stage0::json_quote(w.hash) +
            ",\"atol\":0.001,\"rtol\":0.001,\"warmup\":3,\"samples\":10,\"batch\":" + std::to_string(batch) +
            ",\"threads\":1,\"oracle\":\"unchanged-stage0-fp64-untimed\",\"initial\":" + metrics(initial);
        stage0::write_text(correctness, prefix + ",\"status\":\"initial_checked\"}\n");
        require(initial.passed(), "initial correctness failed; no timing");
        const auto once = [&] {
            if (is_gemm) { gemm(w, inputs); observable = w.out.data<float>()[0]; }
            else observable = graph_once(w, provider.get());
        };
        for (int i = 0; i < warmups; ++i) for (int r = 0; r < batch; ++r) once();
        const std::vector<std::string> columns{"sample","ms","batch","threads","m","n","k","seed","input_hash","algorithm","workload","policy","checksum"};
        for (int i = 0; i < samples; ++i) {
            const auto start = Clock::now();
            for (int r = 0; r < batch; ++r) once();
            const auto stop = Clock::now();
            const auto ms = std::chrono::duration<double, std::milli>(stop-start).count() / batch;
            require(ms > 0 && std::isfinite(ms), "invalid wall time");
            stage0::append_csv(raw, columns, {{"sample",std::to_string(i)},{"ms",stage0::number(ms)},{"batch",std::to_string(batch)},{"threads","1"},
                {"m",size},{"n",size},{"k",size},{"seed","42"},{"input_hash",w.hash},{"algorithm",backend.name()},{"workload",workload},{"policy",policy},{"checksum",stage0::number(observable)}});
        }
        // Revalidate the actual last timed output for GEMM, not a fresh kernel.
        const auto final = is_gemm ? stage0::compare(values(w.out), w.oracle, kAtol, kRtol) : check(w, workload, provider.get(), inputs);
        stage0::write_text(correctness, prefix + ",\"final\":" + metrics(final) + ",\"status\":" + stage0::json_quote(final.passed() ? "passed" : "failed_final") + "}\n");
        require(final.passed(), "final correctness failed; raw samples retained");
        std::cout << workload << '/' << policy << '/' << size << ": PASS warmup=3 samples=10 batch=" << batch << " threads=1\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "bench_cpu: " << e.what() << '\n'; return 1; }
}
