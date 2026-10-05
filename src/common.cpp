#include "stage0/common.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace stage0 {
std::size_t elements(int rows, int cols) {
    if (rows <= 0 || cols <= 0) throw std::invalid_argument("shape must be positive");
    const auto r = static_cast<std::size_t>(rows), c = static_cast<std::size_t>(cols);
    if (r > std::numeric_limits<std::size_t>::max() / c / sizeof(float))
        throw std::overflow_error("matrix byte size overflow");
    return r * c;
}

void cpu_reference(const float* a, const float* b, float* c, Shape s) {
    elements(s.m, s.k); elements(s.k, s.n); elements(s.m, s.n);
    if (!a || !b || !c) throw std::invalid_argument("null matrix");
    // Single-thread ijk. FP64 accumulation is a numerical oracle, never timed.
    for (int i = 0; i < s.m; ++i) {
        for (int j = 0; j < s.n; ++j) {
            double acc = 0;
            for (int k = 0; k < s.k; ++k)
                acc += static_cast<double>(a[static_cast<std::size_t>(i) * s.k + k]) *
                       static_cast<double>(b[static_cast<std::size_t>(k) * s.n + j]);
            c[static_cast<std::size_t>(i) * s.n + j] = static_cast<float>(acc);
        }
    }
}

void random_inputs(std::vector<float>& a, std::vector<float>& b, std::uint32_t seed) {
    std::mt19937 rng(seed);
    // Exact power-of-two scaling, independent of std::uniform_real_distribution implementations.
    auto fill = [&rng](std::vector<float>& v) {
        for (auto& x : v) x = static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f) - 0.5f;
    };
    fill(a); fill(b);
}
std::string input_hash(const std::vector<float>& a, const std::vector<float>& b) {
    std::uint64_t hash = 14695981039346656037ULL;
    auto update = [&hash](const std::vector<float>& v) {
        for (float x : v) {
            std::uint32_t bits;
            static_assert(sizeof(bits) == sizeof(x), "FP32 required");
            std::memcpy(&bits, &x, sizeof(bits));
            for (unsigned shift = 0; shift < 32; shift += 8) {
                hash ^= (bits >> shift) & 0xffu; hash *= 1099511628211ULL;
            }
        }
    };
    update(a); update(b);
    std::ostringstream os; os << std::hex << std::setw(16) << std::setfill('0') << hash;
    return os.str();
}
ErrorMetrics compare(const std::vector<float>& actual, const std::vector<float>& reference,
                     double atol, double rtol) {
    if (actual.empty() || actual.size() != reference.size())
        throw std::invalid_argument("nonempty equally sized outputs required");
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0)
        throw std::invalid_argument("invalid tolerance");
    ErrorMetrics e;
    double total = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double x = actual[i], ref = reference[i];
        if (!std::isfinite(x) || !std::isfinite(ref)) { ++e.nonfinite; continue; }
        const double delta = std::abs(x - ref);
        total += delta;
        e.max_abs = std::max(e.max_abs, delta);
        e.max_relative = std::max(e.max_relative, delta / std::max(std::abs(ref), 1e-12));
        if (delta > atol + rtol * std::abs(ref)) ++e.violations;
    }
    e.mean_abs = total / static_cast<double>(actual.size());
    if (e.nonfinite) e.max_abs = e.mean_abs = e.max_relative = std::numeric_limits<double>::infinity();
    return e;
}
Statistics summarize(const std::vector<double>& ms) {
    if (ms.empty()) throw std::invalid_argument("empty timing samples");
    for (double x : ms)
        if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("nonpositive/nonfinite time");
    auto sorted = ms; std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
    double variance = 0;
    for (double x : ms) variance += (x - mean) * (x - mean);
    const std::size_t mid = sorted.size() / 2;
    const double median = sorted.size() % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    return {sorted.front(), sorted.back(), mean, median, std::sqrt(variance / ms.size())};
}
double gflops(Shape s, double ms) {
    elements(s.m, s.k); elements(s.k, s.n); elements(s.m, s.n);
    if (!std::isfinite(ms) || ms <= 0) throw std::invalid_argument("invalid GEMM time");
    return (2.0 * s.m * s.n * s.k) / (ms * 1e6);
}
std::string number(double x) {
    std::ostringstream os; os << std::setprecision(17) << x; return os.str();
}
std::string timestamp_utc() {
    const auto now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream os; os << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ"); return os.str();
}
std::string json_quote(const std::string& s) {
    std::ostringstream os; os << '"';
    for (unsigned char c : s) {
        switch (c) {
        case '"': os << "\\\""; break;
        case '\\': os << "\\\\"; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default:
            if (c < 32) os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
            else os << c;
        }
    }
    os << '"'; return os.str();
}
const std::vector<std::string>& summary_columns() {
    static const std::vector<std::string> columns = {
        "timestamp", "run_id", "commit", "source_digest", "source_dirty", "experiment",
        "gpu", "gpu_uuid", "device", "compute_capability", "cuda", "cuda_driver", "cublas_version",
        "compiler", "cuda_compiler", "cuda_architectures", "build_type", "kernel",
        "m", "n", "k", "dtype", "layout", "math_mode", "alpha", "beta", "warmup", "iterations",
        "seed", "input_hash", "block_x", "block_y", "bm", "bn", "bk", "tm", "tn",
        "atol", "rtol", "min_ms", "max_ms", "median_ms", "mean_ms", "std_ms", "gflops",
        "speedup_vs_naive", "cublas_ratio", "max_error", "mean_error", "relative_error",
        "violations", "nonfinite", "status"
    };
    return columns;
}
namespace {
std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    return out + '"';
}
std::string csv_line(const std::vector<std::string>& cols, const Record* row) {
    std::string line;
    for (const auto& col : cols) {
        if (!line.empty()) line += ',';
        if (!row) line += csv_quote(col);
        else {
            auto it = row->find(col);
            line += csv_quote(it == row->end() ? "" : it->second);
        }
    }
    return line;
}
}
void append_csv(const std::filesystem::path& path, const std::vector<std::string>& cols,
                const Record& row) {
    if (cols.empty()) throw std::invalid_argument("empty CSV schema");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const auto header = csv_line(cols, nullptr);
    const bool existing = std::filesystem::exists(path) && std::filesystem::file_size(path) != 0;
    if (existing) {
        std::ifstream in(path); std::string first; std::getline(in, first);
        if (!in || first != header) throw std::runtime_error("CSV schema mismatch: " + path.string());
    }
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot open CSV: " + path.string());
    if (!existing) out << header << '\n';
    out << csv_line(cols, &row) << '\n';
    out.flush();
    if (!out) throw std::runtime_error("CSV write failed: " + path.string());
}
void write_text(const std::filesystem::path& path, const std::string& text) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path); out << text; out.flush();
    if (!out) throw std::runtime_error("write failed: " + path.string());
}
}
