#include "model/generation.hpp"
#include "model/model_file.hpp"
#include "model/quantization.hpp"
#include "stage0/common.hpp"
#include "stage0/provenance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::string_view kPrompt = "Stage17 inference";
const std::vector<std::int32_t> kContinuation{73, 18, 3, 234};

struct Options {
    std::string float_model;
    std::string int8_model;
    model::GenerationBackend backend = model::GenerationBackend::CPU;
    bool cache = true;
};

struct Accounting {
    std::size_t file_bytes = 0;
    std::size_t payload_bytes = 0;
    std::size_t eligible_serialized_bytes = 0;
    std::size_t eligible_fp32_bytes = 0;
    std::size_t resident_parameter_bytes = 0;
    std::size_t prepare_dequant_bytes = 0;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&](const char* option) {
            if (++index >= argc) throw std::invalid_argument(std::string(option) + " requires a value");
            return std::string(argv[index]);
        };
        if (argument == "--float-model") options.float_model = value("--float-model");
        else if (argument == "--int8-model") options.int8_model = value("--int8-model");
        else if (argument == "--backend") {
            const auto selected = value("--backend");
            if (selected == "cpu") options.backend = model::GenerationBackend::CPU;
            else if (selected == "mixed") options.backend = model::GenerationBackend::Mixed;
            else throw std::invalid_argument("--backend must be cpu or mixed");
        } else if (argument == "--cache") {
            const auto selected = value("--cache");
            if (selected == "on") options.cache = true;
            else if (selected == "off") options.cache = false;
            else throw std::invalid_argument("--cache must be on or off");
        } else throw std::invalid_argument("usage: bench_quantization --float-model FILE --int8-model FILE "
                                           "[--backend cpu|mixed] [--cache on|off]");
    }
    if (options.float_model.empty() || options.int8_model.empty())
        throw std::invalid_argument("--float-model and --int8-model are required");
    return options;
}

Accounting accounting(const model::ModelFileMetadata& metadata, const model::LoadedModel& loaded) {
    Accounting result{metadata.file_bytes, metadata.payload_bytes};
    for (const auto& descriptor : metadata.tensors) {
        const model::ParameterSpec spec{descriptor.name, descriptor.shape};
        if (!model::is_weight_only_int8_eligible(spec)) continue;
        result.eligible_fp32_bytes += runtime::nbytes(descriptor.shape, runtime::DType::FP32);
        result.eligible_serialized_bytes += descriptor.nbytes;
        if (descriptor.quantization) result.eligible_serialized_bytes += descriptor.quantization->scales_nbytes;
    }
    for (const auto& item : loaded.parameters.tensors()) result.resident_parameter_bytes += item.second.nbytes();
    for (const auto& item : loaded.prepare_dequant_weights) {
        result.resident_parameter_bytes += item.source.values.nbytes() + item.source.scales.nbytes();
        result.prepare_dequant_bytes += loaded.parameters.at(item.name).nbytes();
    }
    return result;
}

void ids(const std::vector<std::int32_t>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

const char* backend_name(model::GenerationBackend backend) {
    return backend == model::GenerationBackend::CPU ? "cpu" : "mixed";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto float_metadata = model::inspect_model_file(options.float_model);
        const auto int8_metadata = model::inspect_model_file(options.int8_model);
        if (float_metadata.format_version != model::MODEL_FILE_VERSION_V1 || int8_metadata.format_version != model::MODEL_FILE_VERSION_V2)
            throw std::invalid_argument("quant benchmark requires V1 float and V2 INT8 models");
        const auto float_model = model::load_model_file(options.float_model);
        const auto int8_model = model::load_model_file(options.int8_model);
        const auto float_accounting = accounting(float_metadata, float_model);
        const auto int8_accounting = accounting(int8_metadata, int8_model);
        if (float_accounting.eligible_fp32_bytes != int8_accounting.eligible_fp32_bytes ||
            float_accounting.eligible_serialized_bytes != float_accounting.eligible_fp32_bytes)
            throw std::runtime_error("float/INT8 eligible parameter accounting mismatch");
        if (int8_accounting.eligible_serialized_bytes * 100U > int8_accounting.eligible_fp32_bytes * 35U)
            throw std::runtime_error("INT8 eligible payload including scales exceeds 35 percent gate");

        const model::GenerationOptions generation{options.backend, options.cache,
                                                   static_cast<std::int64_t>(kContinuation.size())};
        const auto float_run = model::run_teacher_forced(float_model, std::string(kPrompt), kContinuation, generation);
        const auto int8_run = model::run_teacher_forced(int8_model, std::string(kPrompt), kContinuation, generation);
        if (float_run.final_logits.shape() != int8_run.final_logits.shape())
            throw std::runtime_error("float/INT8 teacher-forced logit shape mismatch");
        double sum_absolute_error = 0.0;
        double max_absolute_error = 0.0;
        for (std::size_t index = 0; index < float_run.final_logits.numel(); ++index) {
            const auto reference = float_run.final_logits.data<float>()[index];
            const auto actual = int8_run.final_logits.data<float>()[index];
            const auto error = std::fabs(static_cast<double>(actual) - reference);
            if (!std::isfinite(actual) || error > 1e-2 + 1e-2 * std::fabs(reference))
                throw std::runtime_error("INT8 teacher-forced logit error exceeds frozen tolerance");
            sum_absolute_error += error;
            max_absolute_error = std::max(max_absolute_error, error);
        }
        const auto float_greedy = model::generate_greedy(float_model, std::string(kPrompt), generation);
        const auto int8_greedy = model::generate_greedy(int8_model, std::string(kPrompt), generation);
        if (float_greedy.generated_tokens.empty() || int8_greedy.generated_tokens.empty())
            throw std::runtime_error("greedy smoke returned no tokens");

        std::cout << "{\"schema_version\":1,\"stage\":\"stage16-c4\",\"commit\":"
                  << stage0::json_quote(stage0::kCommit) << ",\"source_digest\":"
                  << stage0::json_quote(stage0::kSourceDigest) << ",\"source_dirty\":"
                  << (stage0::kSourceDirty ? "true" : "false") << ",\"backend\":"
                  << stage0::json_quote(backend_name(options.backend)) << ",\"cache\":"
                  << (options.cache ? "true" : "false") << ",\"eligible_fp32_bytes\":"
                  << int8_accounting.eligible_fp32_bytes << ",\"eligible_int8_scale_bytes\":"
                  << int8_accounting.eligible_serialized_bytes << ",\"eligible_int8_ratio\":"
                  << static_cast<double>(int8_accounting.eligible_serialized_bytes) /
                         static_cast<double>(int8_accounting.eligible_fp32_bytes)
                  << ",\"float_file_bytes\":" << float_accounting.file_bytes
                  << ",\"int8_file_bytes\":" << int8_accounting.file_bytes
                  << ",\"float_payload_bytes\":" << float_accounting.payload_bytes
                  << ",\"int8_payload_bytes\":" << int8_accounting.payload_bytes
                  << ",\"float_resident_parameter_bytes\":" << float_accounting.resident_parameter_bytes
                  << ",\"int8_resident_parameter_bytes\":" << int8_accounting.resident_parameter_bytes
                  << ",\"int8_prepare_dequant_bytes\":" << int8_accounting.prepare_dequant_bytes
                  << ",\"float_peak_live_bytes\":" << float_run.total_counts.peak_live_bytes
                  << ",\"int8_peak_live_bytes\":" << int8_run.total_counts.peak_live_bytes
                  << ",\"float_copies\":" << float_run.total_counts.copies
                  << ",\"int8_copies\":" << int8_run.total_counts.copies
                  << ",\"logit_mae\":" << sum_absolute_error / static_cast<double>(float_run.final_logits.numel())
                  << ",\"logit_max_abs_error\":" << max_absolute_error << ",\"float_generated_tokens\":";
        ids(float_greedy.generated_tokens);
        std::cout << ",\"int8_generated_tokens\":";
        ids(int8_greedy.generated_tokens);
        std::cout << ",\"correctness\":\"passed\"}" << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_quantization: " << error.what() << '\n';
        return 1;
    }
}
