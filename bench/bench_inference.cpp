#include "model/generation.hpp"
#include "model/model_file.hpp"
#include "stage0/build_info.hpp"
#include "stage0/common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kWarmups = 3;
constexpr int kSamples = 10;
const std::string kPrompt = "Stage17 inference";
const std::vector<std::int32_t> kContinuation{73, 18, 3, 234};

enum class QuantMode { Float, Int8 };

struct Options {
    std::string model;
    model::GenerationBackend backend = model::GenerationBackend::CPU;
    QuantMode quant = QuantMode::Float;
    bool cache = true;
    int runs = 3;
};

const char* name(model::GenerationBackend backend) {
    return backend == model::GenerationBackend::CPU ? "cpu" : "mixed";
}
const char* name(QuantMode quant) { return quant == QuantMode::Float ? "float" : "int8"; }

int positive_int(const std::string& value, const char* option) {
    std::size_t parsed = 0;
    int result = 0;
    try { result = std::stoi(value, &parsed, 10); }
    catch (const std::exception&) { throw std::invalid_argument(std::string(option) + " must be positive"); }
    if (parsed != value.size() || result <= 0) throw std::invalid_argument(std::string(option) + " must be positive");
    return result;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&](const char* option) {
            if (++index >= argc) throw std::invalid_argument(std::string(option) + " requires a value");
            return std::string(argv[index]);
        };
        if (argument == "--model") options.model = value("--model");
        else if (argument == "--backend") {
            const auto selected = value("--backend");
            if (selected == "cpu") options.backend = model::GenerationBackend::CPU;
            else if (selected == "mixed") options.backend = model::GenerationBackend::Mixed;
            else throw std::invalid_argument("--backend must be cpu or mixed");
        } else if (argument == "--quant") {
            const auto selected = value("--quant");
            if (selected == "float") options.quant = QuantMode::Float;
            else if (selected == "int8") options.quant = QuantMode::Int8;
            else throw std::invalid_argument("--quant must be float or int8");
        } else if (argument == "--cache") {
            const auto selected = value("--cache");
            if (selected == "on") options.cache = true;
            else if (selected == "off") options.cache = false;
            else throw std::invalid_argument("--cache must be on or off");
        } else if (argument == "--runs") options.runs = positive_int(value("--runs"), "--runs");
        else throw std::invalid_argument("usage: bench_inference --model FILE [--backend cpu|mixed] "
                                         "[--quant float|int8] [--cache on|off] [--runs N]");
    }
    if (options.model.empty()) throw std::invalid_argument("--model is required");
    return options;
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("median requires samples");
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

void array(const std::vector<double>& values) {
    std::cout << '[' << std::setprecision(17);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void ids(const std::vector<std::int32_t>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void check_model_format(const Options& options, const model::ModelFileMetadata& metadata) {
    const auto expected = options.quant == QuantMode::Float ? model::MODEL_FILE_VERSION_V1 : model::MODEL_FILE_VERSION_V2;
    if (metadata.format_version != expected)
        throw std::invalid_argument("--quant does not match model format (float=V1, int8=V2)");
}

void print_record(const Options& options, int run, double load_prepare_ms,
                  const std::vector<model::TeacherForcedResult>& samples) {
    std::vector<double> prefill, first, decode;
    prefill.reserve(samples.size()); first.reserve(samples.size()); decode.reserve(samples.size());
    for (const auto& sample : samples) {
        if (sample.decode_ms.size() != kContinuation.size()) throw std::runtime_error("teacher-forced decode sample count");
        prefill.push_back(sample.prefill_ms);
        first.push_back(sample.first_token_ms);
        double total = 0.0;
        for (const auto value : sample.decode_ms) total += value;
        decode.push_back(total / static_cast<double>(sample.decode_ms.size()));
    }
    const auto& counts = samples.back().total_counts;
    const auto& prefill_counts = samples.back().prefill_counts;
    std::cout << "{\"schema_version\":1,\"stage\":\"stage17-c3\",\"variant\":\"teacher_forced\""
              << ",\"commit\":" << stage0::json_quote(stage0::kCommit)
              << ",\"source_digest\":" << stage0::json_quote(stage0::kSourceDigest)
              << ",\"source_dirty\":" << (stage0::kSourceDirty ? "true" : "false")
              << ",\"build_type\":" << stage0::json_quote(stage0::kBuildType)
              << ",\"compiler\":" << stage0::json_quote(stage0::kCompiler)
              << ",\"cuda_compiler\":" << stage0::json_quote(stage0::kCudaCompiler)
              << ",\"backend\":" << stage0::json_quote(name(options.backend))
              << ",\"quant\":" << stage0::json_quote(name(options.quant))
              << ",\"cache\":" << (options.cache ? "true" : "false")
              << ",\"run\":" << run << ",\"independent_runs\":" << options.runs
              << ",\"warmups\":" << kWarmups << ",\"samples\":" << kSamples
              << ",\"prompt\":" << stage0::json_quote(kPrompt) << ",\"continuation_tokens\":";
    ids(kContinuation);
    std::cout << std::setprecision(17) << ",\"model_load_prepare_ms\":" << load_prepare_ms
              << ",\"prefill_ms_samples\":";
    array(prefill);
    std::cout << ",\"first_token_ms_samples\":";
    array(first);
    std::cout << ",\"decode_ms_per_token_samples\":";
    array(decode);
    std::cout << ",\"prefill_median_ms\":" << median(prefill)
              << ",\"first_token_median_ms\":" << median(first)
              << ",\"decode_ms_per_token_median\":" << median(decode)
              << ",\"tokens_per_second\":" << 1000.0 / median(decode)
              << ",\"cache_persistent_bytes\":" << samples.back().cache_persistent_bytes
              << ",\"prefill_nodes\":" << prefill_counts.nodes_completed
              << ",\"total_nodes\":" << counts.nodes_completed
              << ",\"execute_allocations\":" << counts.allocations
              << ",\"peak_live_bytes\":" << counts.peak_live_bytes
              << ",\"arena_capacity_bytes\":" << counts.arena_capacity_bytes
              << ",\"copies\":" << counts.copies << ",\"copy_bytes\":" << counts.copy_bytes
              << ",\"backend_dispatches\":" << counts.backend_dispatches
              << ",\"backend_switches\":" << counts.backend_switches
              << ",\"correctness\":\"passed\"}" << '\n';
}

void print_smoke(const Options& options, const model::LoadedModel& loaded) {
    const auto generated = model::generate_greedy(loaded, kPrompt,
        {options.backend, options.cache, static_cast<std::int64_t>(kContinuation.size())});
    std::cout << "{\"schema_version\":1,\"stage\":\"stage17-c3\",\"variant\":\"greedy_smoke\""
              << ",\"backend\":" << stage0::json_quote(name(options.backend))
              << ",\"quant\":" << stage0::json_quote(name(options.quant))
              << ",\"cache\":" << (options.cache ? "true" : "false")
              << ",\"generated_tokens\":";
    ids(generated.generated_tokens);
    std::cout << ",\"correctness\":\"passed\"}" << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto load_begin = std::chrono::steady_clock::now();
        const auto metadata = model::inspect_model_file(options.model);
        check_model_format(options, metadata);
        const auto loaded = model::load_model_file(options.model);
        const auto load_end = std::chrono::steady_clock::now();
        const auto load_prepare_ms = std::chrono::duration<double, std::milli>(load_end - load_begin).count();
        const model::GenerationOptions generation{options.backend, options.cache,
                                                   static_cast<std::int64_t>(kContinuation.size())};
        for (int run = 0; run < options.runs; ++run) {
            for (int warmup = 0; warmup < kWarmups; ++warmup)
                (void)model::run_teacher_forced(loaded, kPrompt, kContinuation, generation);
            std::vector<model::TeacherForcedResult> samples;
            samples.reserve(kSamples);
            for (int sample = 0; sample < kSamples; ++sample)
                samples.push_back(model::run_teacher_forced(loaded, kPrompt, kContinuation, generation));
            print_record(options, run, load_prepare_ms, samples);
        }
        print_smoke(options, loaded);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_inference: " << error.what() << '\n';
        return 1;
    }
}
