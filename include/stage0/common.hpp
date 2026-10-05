#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace stage0 {
struct Shape { int m, n, k; };
std::size_t elements(int rows, int cols);
void cpu_reference(const float* a, const float* b, float* c, Shape s);
void random_inputs(std::vector<float>& a, std::vector<float>& b, std::uint32_t seed);
std::string input_hash(const std::vector<float>& a, const std::vector<float>& b);

struct ErrorMetrics {
    double max_abs = 0, mean_abs = 0, max_relative = 0;
    std::size_t violations = 0, nonfinite = 0;
    bool passed() const { return violations == 0 && nonfinite == 0; }
};
ErrorMetrics compare(const std::vector<float>& actual, const std::vector<float>& reference,
                     double atol, double rtol);
struct Statistics { double min, max, mean, median, stddev; };
Statistics summarize(const std::vector<double>& milliseconds);
double gflops(Shape s, double milliseconds);
std::string number(double x);
std::string timestamp_utc();
std::string json_quote(const std::string& s);

using Record = std::map<std::string, std::string>;
const std::vector<std::string>& summary_columns();
// Append only, check existing header; callers must serialize writers.
void append_csv(const std::filesystem::path& path, const std::vector<std::string>& columns,
                const Record& row);
void write_text(const std::filesystem::path& path, const std::string& text);
}
