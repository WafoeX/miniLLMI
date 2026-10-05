#include "stage0/build_info.hpp"
#include "stage0/gemm_cuda.hpp"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace stage0;
namespace fs = std::filesystem;
struct Options {
    std::vector<Shape> shapes{{512,512,512},{1024,1024,1024},{2048,2048,2048},{4096,4096,4096}};
    std::string kernel = "all", experiment = "baseline", run_id;
    fs::path csv = "results/gemm/baseline.csv", raw_dir;
    int device = 0, warmup = 10, iterations = 50;
    std::uint32_t seed = 42;
    double atol = 1e-3, rtol = 1e-3;
};
long long integer(const std::string& value, long long low, long long high) {
    std::size_t used = 0; const auto x = std::stoll(value, &used);
    if (used != value.size() || x < low || x > high) throw std::invalid_argument("invalid integer: " + value);
    return x;
}
double tolerance(const std::string& value) {
    std::size_t used = 0; const double x = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(x) || x < 0) throw std::invalid_argument("invalid tolerance");
    return x;
}
std::string automatic_id() {
    std::string stamp = timestamp_utc();
    stamp.erase(std::remove_if(stamp.begin(), stamp.end(), [](char c) { return c == ':' || c == '-'; }), stamp.end());
    std::ostringstream os; os << stamp << '-' << std::hex << std::random_device{}(); return os.str();
}
Options parse(int argc, char** argv) {
    Options o; int m = 0, n = 0, k = 0; bool sizes_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string v = argv[++i];
        if (key == "--kernel") o.kernel = v;
        else if (key == "--device") o.device = static_cast<int>(integer(v, 0, INT_MAX));
        else if (key == "--warmup") o.warmup = static_cast<int>(integer(v, 10, INT_MAX));
        else if (key == "--iterations") o.iterations = static_cast<int>(integer(v, 30, INT_MAX));
        else if (key == "--seed") o.seed = static_cast<std::uint32_t>(integer(v, 0, UINT32_MAX));
        else if (key == "--m") m = static_cast<int>(integer(v, 1, INT_MAX));
        else if (key == "--n") n = static_cast<int>(integer(v, 1, INT_MAX));
        else if (key == "--k") k = static_cast<int>(integer(v, 1, INT_MAX));
        else if (key == "--atol") o.atol = tolerance(v);
        else if (key == "--rtol") o.rtol = tolerance(v);
        else if (key == "--csv") o.csv = v;
        else if (key == "--raw-dir") o.raw_dir = v;
        else if (key == "--run-id") o.run_id = v;
        else if (key == "--experiment") o.experiment = v;
        else if (key == "--sizes") {
            o.shapes.clear(); sizes_given = true; std::istringstream input(v); std::string part;
            if (v.empty() || v.back() == ',') throw std::invalid_argument("empty size");
            while (std::getline(input, part, ',')) {
                const int s = static_cast<int>(integer(part, 1, INT_MAX));
                if (std::any_of(o.shapes.begin(), o.shapes.end(), [s](Shape x) { return x.m == s; }))
                    throw std::invalid_argument("duplicate size");
                o.shapes.push_back({s,s,s});
            }
        } else throw std::invalid_argument("unknown option: " + key);
    }
    if (m || n || k) {
        if (!m || !n || !k || sizes_given) throw std::invalid_argument("provide --m --n --k together, without --sizes");
        o.shapes = {{m,n,k}};
    }
    if (o.shapes.empty()) throw std::invalid_argument("empty shapes");
    if (o.kernel != "all" && o.kernel != "naive" && o.kernel != "cublas")
        throw std::invalid_argument("kernel must be all, naive or cublas");
    if (o.experiment != "baseline" && o.experiment != "profiling") throw std::invalid_argument("invalid experiment");
    if (o.run_id.empty()) o.run_id = automatic_id();
    if (o.run_id.size() > 120 || !std::all_of(o.run_id.begin(), o.run_id.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '-'; })) throw std::invalid_argument("invalid run ID");
    if (o.raw_dir.empty()) o.raw_dir = fs::path("results/gemm/raw") / o.run_id;
    return o;
}
std::string build_info() {
    return std::string("{\"commit\":") + json_quote(kCommit) + ",\"source_digest\":" + json_quote(kSourceDigest) +
           ",\"source_dirty\":" + (kSourceDirty ? "true" : "false") + ",\"build_type\":" + json_quote(kBuildType) +
           ",\"compiler\":" + json_quote(kCompiler) + ",\"cuda_compiler\":" + json_quote(kCudaCompiler) +
           ",\"cuda_architectures\":" + json_quote(kCudaArchitectures) + "}\n";
}
void errors(Record& r, const ErrorMetrics& e) {
    r["max_error"] = number(e.max_abs); r["mean_error"] = number(e.mean_abs);
    r["relative_error"] = number(e.max_relative);
    r["violations"] = std::to_string(e.violations); r["nonfinite"] = std::to_string(e.nonfinite);
}
struct KernelResult {
    std::string name;
    Record row;
    ErrorMetrics error;
    std::vector<double> samples;
};
void run_shape(const Options& o, Shape s, const cudaDeviceProp& prop, Record metadata) {
    const fs::path dir = o.raw_dir / (std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k));
    if (!fs::create_directories(dir)) throw std::runtime_error("refusing to overwrite existing shape run: " + dir.string());
    const auto ac = elements(s.m,s.k), bc = elements(s.k,s.n), cc = elements(s.m,s.n);
    if ((static_cast<unsigned>(s.n) + kBlockX - 1) / kBlockX > static_cast<unsigned>(prop.maxGridSize[0]) ||
        (static_cast<unsigned>(s.m) + kBlockY - 1) / kBlockY > static_cast<unsigned>(prop.maxGridSize[1]) ||
        kBlockX * kBlockY > static_cast<unsigned>(prop.maxThreadsPerBlock))
        throw std::runtime_error("naive launch exceeds queried device limits");
    std::vector<float> a(ac), b(bc), reference(cc), actual(cc);
    random_inputs(a, b, o.seed);
    std::cout << "CPU reference " << s.m << 'x' << s.n << 'x' << s.k << " (full single-thread ijk; not timed)" << std::endl;
    cpu_reference(a.data(), b.data(), reference.data(), s);
    Stream stream; DeviceBuffer da(ac), db(bc), dc(cc); BlasHandle blas(stream.get());
    CUDA_CHECK(cudaMemcpyAsync(da.get(), a.data(), ac*sizeof(float), cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(db.get(), b.data(), bc*sizeof(float), cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    int blas_version = 0; CUBLAS_CHECK(cublasGetVersion(blas.get(), &blas_version));
    metadata["cublas_version"] = std::to_string(blas_version);
    metadata["m"] = std::to_string(s.m); metadata["n"] = std::to_string(s.n); metadata["k"] = std::to_string(s.k);
    metadata["input_hash"] = input_hash(a,b);
    std::vector<KernelResult> results;
    auto add = [&](const std::string& name) {
        KernelResult r; r.name = name; r.row = metadata; r.row["kernel"] = name;
        if (name == "sgemm_v0_naive") {
            r.row["block_x"] = std::to_string(kBlockX); r.row["block_y"] = std::to_string(kBlockY);
            r.row["tm"] = "1"; r.row["tn"] = "1";
        }
        results.push_back(std::move(r));
    };
    if (o.kernel == "all" || o.kernel == "naive") add("sgemm_v0_naive");
    if (o.kernel == "all" || o.kernel == "cublas") add("cublas");
    auto launch = [&](const KernelResult& r) {
        if (r.name == "sgemm_v0_naive") launch_naive(da.get(), db.get(), dc.get(), s, stream.get());
        else launch_cublas(blas.get(), da.get(), db.get(), dc.get(), s);
    };
    auto validate = [&](KernelResult& r, const std::string& phase) {
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), dc.get(), cc*sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        r.error = compare(actual, reference, o.atol, o.rtol); errors(r.row, r.error);
        auto record = r.row; record["status"] = r.error.passed() ? "passed_" + phase : "failed_correctness";
        append_csv(dir / "correctness.csv", summary_columns(), record);
        if (!r.error.passed()) {
            r.row["status"] = "failed_correctness";
            // No successful time/GFLOPS are published for incorrect results.
            append_csv(o.csv, summary_columns(), r.row);
            throw std::runtime_error(r.name + " correctness failed; inspect " + dir.string());
        }
    };
    // Validate ALL selected kernels before any timing for this shape.
    for (auto& r : results) {
        CUDA_CHECK(cudaMemsetAsync(dc.get(), 0xff, cc*sizeof(float), stream.get()));
        launch(r); validate(r, "initial");
    }
    EventTimer timer;
    auto raw_cols = summary_columns(); raw_cols.push_back("sample_index"); raw_cols.push_back("elapsed_ms");
    for (auto& r : results) {
        for (int i = 0; i < o.warmup; ++i) launch(r);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        for (int i = 0; i < o.iterations; ++i) {
            const double ms = timer.measure(stream.get(), [&] { launch(r); });
            r.samples.push_back(ms);
            auto sample = r.row; sample["status"] = "timing_sample";
            sample["sample_index"] = std::to_string(i); sample["elapsed_ms"] = number(ms);
            append_csv(dir / (r.name + "_samples.csv"), raw_cols, sample);
        }
        // Also validate the final measured output, before publishing summaries.
        validate(r, "final");
    }
    double naive = 0, cublas = 0;
    for (auto& r : results) {
        const auto st = summarize(r.samples);
        r.row["min_ms"] = number(st.min); r.row["max_ms"] = number(st.max);
        r.row["median_ms"] = number(st.median); r.row["mean_ms"] = number(st.mean);
        r.row["std_ms"] = number(st.stddev); r.row["gflops"] = number(gflops(s,st.median));
        if (r.name == "sgemm_v0_naive") naive = gflops(s,st.median);
        else cublas = gflops(s,st.median);
    }
    for (auto& r : results) {
        const double perf = std::stod(r.row.at("gflops"));
        if (naive > 0) r.row["speedup_vs_naive"] = number(perf / naive);
        if (cublas > 0) r.row["cublas_ratio"] = number(100 * perf / cublas);
        r.row["status"] = "ok";
        append_csv(dir / "summary.csv", summary_columns(), r.row);
        append_csv(o.csv, summary_columns(), r.row);
        std::cout << r.name << ": median=" << r.row["median_ms"] << " ms, " << perf
                  << " GFLOPS, max_error=" << r.error.max_abs << '\n';
    }
}
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    fs::path failure_path;
    try {
        if (argc == 2 && std::string(argv[1]) == "--build-info") { std::cout << build_info(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "bench_gemm [--kernel all|naive|cublas] [--sizes 512,1024,2048,4096]\n"
                         "  [--m M --n N --k K] [--device 0] [--warmup 10] [--iterations 50]\n"
                         "  [--seed 42] [--atol 0.001] [--rtol 0.001] [--csv PATH]\n"
                         "  [--run-id ID] [--raw-dir PATH] [--experiment baseline|profiling]\n";
            return 0;
        }
        const auto o = parse(argc,argv);
        fs::create_directories(o.raw_dir);
        if (fs::exists(o.raw_dir / "build_info.json")) throw std::runtime_error("run directory already used");
        failure_path = o.raw_dir / "failure.txt";
        write_text(o.raw_dir / "build_info.json", build_info());
        std::string command;
        for (int i = 0; i < argc; ++i) command += json_quote(argv[i]) + '\n';
        write_text(o.raw_dir / "argv.txt", command);
        if (std::string(kBuildType) != "Release") throw std::runtime_error("GPU benchmarks require Release");
        const std::string commit = kCommit;
        if (kSourceDirty || commit.size() != 40 || !std::all_of(commit.begin(),commit.end(), [](unsigned char c) {return std::isxdigit(c);}))
            throw std::runtime_error("official benchmark requires a committed, clean source build");
        int count = 0; CUDA_CHECK(cudaGetDeviceCount(&count));
        if (o.device >= count) throw std::runtime_error("selected CUDA device not available");
        CUDA_CHECK(cudaSetDevice(o.device));
        cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,o.device));
        int runtime = 0, driver = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
        Record metadata = {
            {"timestamp",timestamp_utc()},{"run_id",o.run_id},{"commit",kCommit},{"source_digest",kSourceDigest},
            {"source_dirty","0"},{"experiment",o.experiment},{"gpu",p.name},{"gpu_uuid",uuid_string(p.uuid)},
            {"device",std::to_string(o.device)},{"compute_capability",std::to_string(p.major)+"."+std::to_string(p.minor)},
            {"cuda",std::to_string(runtime)},{"cuda_driver",std::to_string(driver)},
            {"compiler",kCompiler},{"cuda_compiler",kCudaCompiler},{"cuda_architectures",kCudaArchitectures},
            {"build_type",kBuildType},{"dtype","fp32"},{"layout","row-major"},{"math_mode","fp32_pedantic"},
            {"alpha","1"},{"beta","0"},{"warmup",std::to_string(o.warmup)},{"iterations",std::to_string(o.iterations)},
            {"seed",std::to_string(o.seed)},{"atol",number(o.atol)},{"rtol",number(o.rtol)}
        };
        for (auto s : o.shapes) run_shape(o,s,p,metadata);
        write_text(o.raw_dir / "completed.txt", "all requested shapes passed correctness and timing\n");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "bench_gemm: " << e.what() << '\n';
        if (!failure_path.empty() && !fs::exists(failure_path)) {
            try { stage0::write_text(failure_path, e.what() + std::string("\n")); } catch (...) {}
        }
        return 1;
    }
}
