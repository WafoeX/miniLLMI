// bench_gemm is a CUDA-only target.  This guard keeps host-only editor analysis
// independent of an unavailable local CUDA toolkit; CMake defines it for the
// actual CUDA target.
#if defined(STAGE0_CUDA_BUILD)
#include "stage0/build_info.hpp"
#include "stage0/gemm_cuda.hpp"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
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
    std::string kernel = "all", order, experiment = "baseline", run_id;
    fs::path csv = "results/gemm/baseline.csv", raw_dir, reference_cache_dir;
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
        else if (key == "--order") o.order = v;
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
        else if (key == "--reference-cache-dir") o.reference_cache_dir = v;
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
    if (!gemm_kernel_from_name(o.kernel) && o.kernel != "all")
        throw std::invalid_argument("kernel must be all, naive, tiled or cublas");
    if (o.experiment != "baseline" && o.experiment != "profiling" && o.experiment != "stage9")
        throw std::invalid_argument("invalid experiment");
    if (o.experiment != "stage9" && (o.kernel == "tiled" || o.kernel == "v1" || o.kernel == "sgemm_v1_tiled"))
        throw std::invalid_argument("tiled is available only in experiment stage9");
    if (!o.reference_cache_dir.empty() && o.experiment != "stage9")
        throw std::invalid_argument("--reference-cache-dir is supported only for experiment stage9");
    if (o.experiment == "stage9" && o.csv == fs::path("results/gemm/baseline.csv")) o.csv = "results/gemm/stage9.csv";
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
const std::vector<std::string>& stage9_columns() {
    static const std::vector<std::string> columns = [] {
        auto result = summary_columns();
        result.insert(result.begin(), "schema_version");
        result.insert(result.end(), {"threads", "vector", "shared_bytes", "kernel_order"});
        return result;
    }();
    return columns;
}
std::vector<GemmKernel> selected_kernels(const Options& o) {
    if (o.experiment == "stage9" && o.kernel != "all")
        throw std::invalid_argument("stage9 requires --kernel all for paired v0/v1/cuBLAS evidence");
    if (o.order.empty()) {
        if (o.kernel == "all") return o.experiment == "stage9"
            ? std::vector<GemmKernel>{GemmKernel::V0Naive, GemmKernel::V1Tiled, GemmKernel::CuBlas}
            : std::vector<GemmKernel>{GemmKernel::V0Naive, GemmKernel::CuBlas};
        return {*gemm_kernel_from_name(o.kernel)};
    }
    std::vector<GemmKernel> result;
    std::istringstream input(o.order); std::string part;
    while (std::getline(input, part, ',')) {
        const auto kernel = gemm_kernel_from_name(part);
        if (!kernel || std::find(result.begin(), result.end(), *kernel) != result.end())
            throw std::invalid_argument("invalid or duplicate kernel order");
        result.push_back(*kernel);
    }
    const std::vector<GemmKernel> required{GemmKernel::V0Naive, GemmKernel::V1Tiled, GemmKernel::CuBlas};
    if (o.experiment != "stage9" || result.size() != required.size() ||
        !std::all_of(required.begin(), required.end(), [&](GemmKernel kernel) {
            return std::find(result.begin(), result.end(), kernel) != result.end(); }))
        throw std::invalid_argument("--order is supported only for a complete stage9 v0,v1,cublas sequence");
    return result;
}
struct KernelResult {
    GemmKernel kernel;
    std::string name;
    Record row;
    ErrorMetrics error;
    std::vector<double> samples;
};
void prepare_reference(const Options& o, Shape s, const std::vector<float>& a, const std::vector<float>& b,
                       std::vector<float>& output) {
    if (o.reference_cache_dir.empty()) {
        std::cout << "CPU reference " << s.m << 'x' << s.n << 'x' << s.k
                  << " (full single-thread ijk; not timed)" << std::endl;
        cpu_reference(a.data(), b.data(), output.data(), s);
        return;
    }
    const auto name = std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k) +
                      "-seed" + std::to_string(o.seed) + "-" + input_hash(a, b) + ".f32";
    const auto path = o.reference_cache_dir / name;
    const auto bytes = output.size() * sizeof(float);
    if (fs::exists(path)) {
        if (fs::file_size(path) != bytes) throw std::runtime_error("invalid CPU reference cache: " + path.string());
        std::ifstream input(path, std::ios::binary);
        input.read(reinterpret_cast<char*>(output.data()), static_cast<std::streamsize>(bytes));
        if (!input || input.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("cannot read CPU reference cache: " + path.string());
        std::cout << "CPU reference cache hit " << s.m << 'x' << s.n << 'x' << s.k
                  << " (full FP64 oracle generated by this Stage 9 run)" << std::endl;
        return;
    }
    std::cout << "CPU reference cache miss " << s.m << 'x' << s.n << 'x' << s.k
              << " (full single-thread ijk; not timed)" << std::endl;
    cpu_reference(a.data(), b.data(), output.data(), s);
    fs::create_directories(o.reference_cache_dir);
    const auto temporary = path.string() + "." + o.run_id + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(bytes));
        stream.flush();
        if (!stream) throw std::runtime_error("cannot write CPU reference cache: " + path.string());
    }
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        fs::remove(temporary);
        throw std::runtime_error("cannot finalize CPU reference cache: " + path.string());
    }
}
void run_shape(const Options& o, Shape s, const cudaDeviceProp& prop, Record metadata) {
    const auto& columns = o.experiment == "stage9" ? stage9_columns() : summary_columns();
    const auto kernels = selected_kernels(o);
    const fs::path dir = o.raw_dir / (std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k));
    if (!fs::create_directories(dir)) throw std::runtime_error("refusing to overwrite existing shape run: " + dir.string());
    const auto ac = elements(s.m,s.k), bc = elements(s.k,s.n), cc = elements(s.m,s.n);
    if ((static_cast<unsigned>(s.n) + kBlockX - 1) / kBlockX > static_cast<unsigned>(prop.maxGridSize[0]) ||
        (static_cast<unsigned>(s.m) + kBlockY - 1) / kBlockY > static_cast<unsigned>(prop.maxGridSize[1]) ||
        kBlockX * kBlockY > static_cast<unsigned>(prop.maxThreadsPerBlock))
        throw std::runtime_error("naive launch exceeds queried device limits");
    std::vector<float> a(ac), b(bc), reference(cc), actual(cc);
    random_inputs(a, b, o.seed);
    prepare_reference(o, s, a, b, reference);
    Stream stream; DeviceBuffer da(ac), db(bc), dc(cc); BlasHandle blas(stream.get());
    CUDA_CHECK(cudaMemcpyAsync(da.get(), a.data(), ac*sizeof(float), cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(db.get(), b.data(), bc*sizeof(float), cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    int blas_version = 0; CUBLAS_CHECK(cublasGetVersion(blas.get(), &blas_version));
    metadata["cublas_version"] = std::to_string(blas_version);
    metadata["m"] = std::to_string(s.m); metadata["n"] = std::to_string(s.n); metadata["k"] = std::to_string(s.k);
    metadata["input_hash"] = input_hash(a,b);
    std::vector<KernelResult> results;
    for (const auto kernel : kernels) {
        const auto& config = gemm_kernel_config(kernel);
        if (config.threads && (config.threads > static_cast<unsigned>(prop.maxThreadsPerBlock) ||
                               config.shared_bytes > static_cast<std::size_t>(prop.sharedMemPerBlock)))
            throw std::runtime_error(std::string(config.name) + " exceeds queried device launch resources");
        KernelResult result{kernel, config.name, metadata, {}, {}};
        result.row["kernel"] = config.name;
        result.row["block_x"] = std::to_string(config.block_x); result.row["block_y"] = std::to_string(config.block_y);
        result.row["bm"] = std::to_string(config.bm); result.row["bn"] = std::to_string(config.bn);
        result.row["bk"] = std::to_string(config.bk); result.row["tm"] = std::to_string(config.tm);
        result.row["tn"] = std::to_string(config.tn);
        if (o.experiment == "stage9") {
            result.row["schema_version"] = "stage9-gemm-v1";
            result.row["threads"] = std::to_string(config.threads);
            result.row["vector"] = std::to_string(config.vector_width);
            result.row["shared_bytes"] = std::to_string(config.shared_bytes);
            result.row["kernel_order"] = o.order.empty() ? "v0,v1,cublas" : o.order;
        }
        results.push_back(std::move(result));
    }
    auto launch = [&](const KernelResult& r) {
        launch_gemm(r.kernel, blas.get(), da.get(), db.get(), dc.get(), s, stream.get());
    };
    auto validate = [&](KernelResult& r, const std::string& phase) {
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), dc.get(), cc*sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        r.error = compare(actual, reference, o.atol, o.rtol); errors(r.row, r.error);
        auto record = r.row; record["status"] = r.error.passed() ? "passed_" + phase : "failed_correctness";
        append_csv(dir / "correctness.csv", columns, record);
        if (!r.error.passed()) {
            r.row["status"] = "failed_correctness";
            // No successful time/GFLOPS are published for incorrect results.
            append_csv(o.csv, columns, r.row);
            throw std::runtime_error(r.name + " correctness failed; inspect " + dir.string());
        }
    };
    // Validate ALL selected kernels before any timing for this shape.
    for (auto& r : results) {
        CUDA_CHECK(cudaMemsetAsync(dc.get(), 0xff, cc*sizeof(float), stream.get()));
        launch(r); validate(r, "initial");
    }
    EventTimer timer;
    auto raw_cols = columns; raw_cols.push_back("sample_index"); raw_cols.push_back("elapsed_ms");
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
        if (r.kernel == GemmKernel::V0Naive) naive = gflops(s,st.median);
        if (r.kernel == GemmKernel::CuBlas) cublas = gflops(s,st.median);
    }
    for (auto& r : results) {
        const double perf = std::stod(r.row.at("gflops"));
        if (naive > 0) r.row["speedup_vs_naive"] = number(perf / naive);
        if (cublas > 0) r.row["cublas_ratio"] = number(100 * perf / cublas);
        r.row["status"] = "ok";
        append_csv(dir / "summary.csv", columns, r.row);
        append_csv(o.csv, columns, r.row);
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
            std::cout << "bench_gemm [--kernel all|naive|tiled|cublas] [--sizes 512,1024,2048,4096]\n"
                         "  [--m M --n N --k K] [--device 0] [--warmup 10] [--iterations 50]\n"
                         "  [--seed 42] [--atol 0.001] [--rtol 0.001] [--csv PATH]\n"
                         "  [--run-id ID] [--raw-dir PATH] [--reference-cache-dir PATH] [--experiment baseline|profiling|stage9]\n"
                         "  [--order v0,v1,cublas] (stage9 paired runs only)\n";
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
#endif // STAGE0_CUDA_BUILD
