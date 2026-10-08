#include "model/model_file.hpp"
#include "model/quantization.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace model;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void put_u8(std::vector<unsigned char>& bytes, std::uint8_t value) { bytes.push_back(value); }
void put_u32(std::vector<unsigned char>& bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) put_u8(bytes, static_cast<std::uint8_t>(value >> shift));
}
void put_u64(std::vector<unsigned char>& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) put_u8(bytes, static_cast<std::uint8_t>(value >> shift));
}
void put_i64(std::vector<unsigned char>& bytes, std::int64_t value) {
    put_u64(bytes, static_cast<std::uint64_t>(value));
}
void put_f64(std::vector<unsigned char>& bytes, double value) {
    std::uint64_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    put_u64(bytes, raw);
}

void write_container(const std::filesystem::path& path, bool overlap = false, bool duplicate = false) {
    const auto config = DecoderConfig::tiny();
    const auto specs = parameter_specs(config);
    std::vector<std::string> names;
    names.reserve(specs.size());
    std::size_t metadata_bytes = 0;
    for (const auto& spec : specs) {
        names.push_back(spec.name);
        metadata_bytes += 4 + spec.name.size() + 4 + 4 + 8 + 8 + 8 * spec.shape.rank();
    }
    if (duplicate) names.at(1) = names.at(0);
    const std::size_t fixed_bytes = 8 + 4 + 4 + 4 + 9 * 8 + 2 * 8 + 1 + 7;
    const auto payload_start = fixed_bytes + metadata_bytes;
    std::vector<std::size_t> offsets;
    offsets.reserve(specs.size());
    std::size_t cursor = payload_start;
    for (const auto& spec : specs) {
        offsets.push_back(cursor);
        cursor += runtime::nbytes(spec.shape, runtime::DType::FP32);
    }
    if (overlap) offsets.at(1) = offsets.at(0);

    std::vector<unsigned char> bytes;
    for (const auto value : std::array<unsigned char, 8>{{'M', 'L', 'L', 'M', 'R', 'T', 'F', 0}}) put_u8(bytes, value);
    put_u32(bytes, MODEL_FILE_VERSION);
    put_u32(bytes, BYTE_VOCABULARY_VERSION);
    put_u32(bytes, static_cast<std::uint32_t>(specs.size()));
    for (const auto value : {config.batch, config.layers, config.hidden, config.heads, config.kv_heads,
                             config.head_dim, config.ffn, config.vocab, config.max_seq})
        put_i64(bytes, value);
    put_f64(bytes, config.rms_epsilon);
    put_f64(bytes, config.rope_base);
    put_u8(bytes, 0);
    for (int index = 0; index < 7; ++index) put_u8(bytes, 0);
    for (std::size_t index = 0; index < specs.size(); ++index) {
        const auto& spec = specs[index];
        put_u32(bytes, static_cast<std::uint32_t>(names[index].size()));
        bytes.insert(bytes.end(), names[index].begin(), names[index].end());
        put_u32(bytes, 1);
        put_u32(bytes, static_cast<std::uint32_t>(spec.shape.rank()));
        put_u64(bytes, offsets[index]);
        put_u64(bytes, runtime::nbytes(spec.shape, runtime::DType::FP32));
        for (const auto dimension : spec.shape.values()) put_i64(bytes, dimension);
    }
    for (const auto& spec : specs)
        bytes.insert(bytes.end(), runtime::nbytes(spec.shape, runtime::DType::FP32), 0);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("cannot write test model container");
}

template<class Action>
void require_rejected(Action&& action, const char* message) {
    try {
        action();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

ParameterTable deterministic_parameters(const DecoderConfig& config) {
    std::map<std::string, runtime::Tensor> tensors;
    for (const auto& spec : parameter_specs(config)) {
        auto tensor = runtime::Tensor::allocate_cpu(spec.shape);
        for (std::size_t index = 0; index < tensor.numel(); ++index)
            tensor.data<float>()[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 29.F;
        tensors.emplace(spec.name, std::move(tensor));
    }
    return ParameterTable(config, std::move(tensors));
}

void test_v2_quantized_container(const std::filesystem::path& root) {
    const auto path = root / "quantized-v2.mllm";
    const auto parameters = deterministic_parameters(DecoderConfig::tiny());
    write_quantized_model_file(path.string(), parameters);
    const auto metadata = inspect_model_file(path.string());
    require(metadata.format_version == MODEL_FILE_VERSION_V2, "V2 format version");
    std::size_t eligible_fp32_bytes = 0;
    std::size_t quantized_payload_bytes = 0;
    std::size_t quantized_count = 0;
    for (const auto& descriptor : metadata.tensors) {
        const auto spec = ParameterSpec{descriptor.name, descriptor.shape};
        if (is_weight_only_int8_eligible(spec)) {
            require(descriptor.dtype == runtime::DType::INT8 && descriptor.quantization &&
                        descriptor.quantization->version == INT8_QUANTIZATION_VERSION &&
                        descriptor.quantization->output_axis == 1 &&
                        descriptor.quantization->scales_dtype == runtime::DType::FP32 &&
                        descriptor.quantization->scales_shape == runtime::Shape({descriptor.shape[1]}),
                    "eligible V2 INT8 descriptor");
            eligible_fp32_bytes += runtime::nbytes(descriptor.shape, runtime::DType::FP32);
            quantized_payload_bytes += descriptor.nbytes + descriptor.quantization->scales_nbytes;
            ++quantized_count;
        } else {
            require(descriptor.dtype == runtime::DType::FP32 && !descriptor.quantization,
                    "ineligible tensor must remain FP32");
        }
    }
    require(quantized_count != 0 && quantized_payload_bytes * 100 <= eligible_fp32_bytes * 35,
            "INT8 plus scale payload compression gate");

    const auto corrupted = root / "quantized-corrupt.mllm";
    std::filesystem::copy_file(path, corrupted, std::filesystem::copy_options::overwrite_existing);
    const auto first_quantized = std::find_if(metadata.tensors.begin(), metadata.tensors.end(),
        [](const ModelTensorMetadata& descriptor) { return descriptor.quantization.has_value(); });
    require(first_quantized != metadata.tensors.end(), "V2 needs at least one quantized tensor");
    {
        std::fstream output(corrupted, std::ios::binary | std::ios::in | std::ios::out);
        output.seekg(static_cast<std::streamoff>(first_quantized->offset_bytes));
        char value = 0;
        output.read(&value, 1);
        output.seekp(static_cast<std::streamoff>(first_quantized->offset_bytes));
        value ^= 1;
        output.write(&value, 1);
    }
    require_rejected([&] { (void)inspect_model_file(corrupted.string()); }, "V2 checksum mismatch accepted");
}
} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() / "mini_llm_stage15_model_file";
    try {
        std::filesystem::create_directories(root);
        const auto valid = root / "valid.mllm";
        write_container(valid);
        const auto metadata = inspect_model_file(valid.string());
        require(metadata.config.vocab == 258 && metadata.vocabulary_version == BYTE_VOCABULARY_VERSION,
                "valid header metadata");
        require(metadata.format_version == MODEL_FILE_VERSION_V1 &&
                    metadata.tensors.size() == parameter_specs(metadata.config).size() &&
                    metadata.payload_bytes == parameter_bytes(metadata.config),
                "valid canonical tensor metadata");
        test_v2_quantized_container(root);

        const auto truncated = root / "truncated.mllm";
        write_container(truncated);
        std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 1);
        require_rejected([&] { (void)inspect_model_file(truncated.string()); }, "truncated payload accepted");

        const auto overlap = root / "overlap.mllm";
        write_container(overlap, true);
        require_rejected([&] { (void)inspect_model_file(overlap.string()); }, "overlapping payload accepted");

        const auto duplicate = root / "duplicate.mllm";
        write_container(duplicate, false, true);
        require_rejected([&] { (void)inspect_model_file(duplicate.string()); }, "duplicate name accepted");

        const auto version = root / "version.mllm";
        write_container(version);
        {
            std::fstream output(version, std::ios::binary | std::ios::in | std::ios::out);
            output.seekp(8);
            const unsigned char unsupported[] = {2, 0, 0, 0};
            output.write(reinterpret_cast<const char*>(unsupported), sizeof(unsupported));
        }
        require_rejected([&] { (void)inspect_model_file(version.string()); }, "unsupported version accepted");
        require_rejected([&] { (void)inspect_model_file(valid.string(), {1024}); }, "allocation budget ignored");

        std::filesystem::remove_all(root);
        std::cout << "Stage 15 C1 and Stage 16 C2 versioned model-file validation: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << "test_model_file: " << error.what() << '\n';
        return 1;
    }
}
