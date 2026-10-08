#pragma once

#include "model/config.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace model {

// Little-endian on-disk containers use one fixed byte vocabulary: token IDs
// 0..255 plus BOS=256 and EOS=257. V1 is preserved FP32-only; V2 adds explicit
// Stage 16 weight-only INT8 descriptors without changing the runtime graph.
inline constexpr std::uint32_t MODEL_FILE_VERSION_V1 = 1;
inline constexpr std::uint32_t MODEL_FILE_VERSION_V2 = 2;
inline constexpr std::uint32_t MODEL_FILE_VERSION = MODEL_FILE_VERSION_V1;
inline constexpr std::uint32_t BYTE_VOCABULARY_VERSION = 1;

struct ModelFileLimits {
    // Applies to the sum of all serialized payloads, before runtime allocation.
    std::size_t max_allocation_bytes = 64U * 1024U * 1024U;
};

struct ModelQuantizationMetadata {
    std::uint32_t version = 0;
    std::size_t output_axis = 0;
    runtime::DType scales_dtype = runtime::DType::FP32;
    runtime::Shape scales_shape;
    std::size_t scales_offset_bytes = 0;
    std::size_t scales_nbytes = 0;
};

struct ModelTensorMetadata {
    std::string name;
    runtime::DType dtype;
    runtime::Shape shape;
    std::size_t offset_bytes = 0;
    std::size_t nbytes = 0;
    std::uint64_t checksum = 0; // V2 FNV-1a over value payload then scales.
    std::optional<ModelQuantizationMetadata> quantization;
};

struct ModelFileMetadata {
    DecoderConfig config;
    std::uint32_t format_version = 0;
    std::uint32_t vocabulary_version = 0;
    std::vector<ModelTensorMetadata> tensors;
    std::size_t payload_bytes = 0;
    std::size_t file_bytes = 0;
};

// Parses and fully validates a V1 or V2 container without allocating runtime
// tensors. V2 checks payload checksums and accepts INT8 only with a finite,
// positive FP32 scale tensor on output axis 1.
ModelFileMetadata inspect_model_file(const std::string& path, ModelFileLimits limits = {});

struct LoadedModel {
    ParameterTable parameters;
    std::uint32_t vocabulary_version = 0;

    const DecoderConfig& config() const noexcept { return parameters.config(); }
};

// Deterministic offline FP32 V1 writer and checked loader. The loader allocates
// only after inspect_model_file has enforced its budget.
void write_model_file(const std::string& path, const ParameterTable& parameters,
                      std::uint32_t vocabulary_version = BYTE_VOCABULARY_VERSION);

// Deterministic offline V2 writer. Eligible projection W[in,out] tensors are
// quantized per output column; embeddings and norm weights remain FP32.
void write_quantized_model_file(const std::string& path, const ParameterTable& parameters,
                                std::uint32_t vocabulary_version = BYTE_VOCABULARY_VERSION);
LoadedModel load_model_file(const std::string& path, ModelFileLimits limits = {});

} // namespace model
