#include "stage0/common.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void throws(F&& fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    require(caught, "expected exception");
}
}
int main() {
    std::filesystem::path temp;
    try {
        using namespace stage0;
        std::vector<float> a{1,2,3,4,5,6}, b{7,8,9,10,11,12}, c(4);
        cpu_reference(a.data(),b.data(),c.data(),{2,2,3});
        require(c == std::vector<float>({58,64,139,154}), "rectangular ijk GEMM");
        c.assign(4,123); cpu_reference(a.data(),b.data(),c.data(),{2,2,3});
        require(c[0] == 58, "reference must overwrite, not accumulate old C");
        std::vector<float> za(6,0), zb(6,0);
        cpu_reference(za.data(),zb.data(),c.data(),{2,2,3});
        require(c == std::vector<float>(4,0), "zero inputs");
        throws([] { elements(0,2); }); throws([&] { cpu_reference(nullptr,b.data(),c.data(),{2,2,3}); });
        std::vector<float> x(32), y(17), x2(32), y2(17);
        random_inputs(x,y,42); random_inputs(x2,y2,42);
        require(x == x2 && y == y2 && input_hash(x,y) == input_hash(x2,y2), "seed reproducibility");
        for (float v : x) require(v >= -0.5f && v < 0.5f, "input range");
        y2[0] += 1; require(input_hash(x,y) != input_hash(x2,y2), "input hash");
        const auto ok = compare({1.001f,0.00001f},{1,0},0.0001,0.002);
        require(ok.passed() && ok.max_abs > 0, "mixed tolerance");
        require(!compare({1.01f},{1},0.001,0.001).passed(), "incorrect data must fail");
        const auto nan = compare({std::numeric_limits<float>::quiet_NaN()},{0},1,1);
        require(!nan.passed() && nan.nonfinite == 1 && std::isinf(nan.max_abs), "NaN must fail");
        require(!compare({std::numeric_limits<float>::infinity()},{0},1,1).passed(), "Inf must fail");
        throws([] { compare({}, {}, 0,0); }); throws([] { compare({0},{0},-1,0); });
        auto stats = summarize({4,1,2,3});
        require(stats.min == 1 && stats.max == 4 && stats.median == 2.5 && stats.mean == 2.5,
                "min/max/mean/even median");
        require(std::abs(stats.stddev - std::sqrt(1.25)) < 1e-12, "population stddev");
        require(summarize({3,1,2}).median == 2, "odd median");
        throws([] { summarize({0}); }); throws([] { summarize({}); });
        require(gflops({100,100,100},2) == 1, "GFLOPS formula");
        throws([] { gflops({1,1,1},0); });
        require(json_quote("a\n\"\\") == "\"a\\n\\\"\\\\\"", "JSON escaping");
        temp = std::filesystem::temp_directory_path() / ("stage0-test-" + std::to_string(std::random_device{}()));
        const auto csv = temp / "test.csv";
        append_csv(csv,{"name","value"},{{"name","a,\"b"},{"value","2"}});
        append_csv(csv,{"name","value"},{{"name","second"}});
        std::ifstream in(csv); std::string header, row; std::getline(in,header); std::getline(in,row);
        require(header == "\"name\",\"value\"", "CSV header");
        require(row == "\"a,\"\"b\",\"2\"", "CSV escaping");
        throws([&] { append_csv(csv,{"different"},{}); });
        std::filesystem::remove_all(temp);
        std::cout << "CPU reference, validation, statistics, GFLOPS, CSV: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        if (!temp.empty()) std::filesystem::remove_all(temp);
        std::cerr << "test_common: " << e.what() << '\n'; return 1;
    }
}
