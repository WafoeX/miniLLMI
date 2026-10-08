#pragma once

#include "model/config.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace model {

// Little-endian on-disk container version. Version 1 has one fixed byte
// vocabulary: token IDs 0..255 plus BOS=256 and EOS=257.
inline constexpr std::uint32_t MODEL_FILE_VERSION = 1;
inline constexpr std::uint32_t BYTE_VOCABULARY_VERSION = 1;

struct ModelFileLimits {
    // Applies to the sum of tensor payload bytes, before any tensor allocation.
    std::size_t max_allocation_bytes = 64U * 1024U * 1024U;
};

struct ModelTensorMetadata {
    std::string name;
    runtime::DType dtype;
    runtime::Shape shape;
    std::size_t offset_bytes = 0;
    std::size_t nbytes = 0;
};

struct ModelFileMetadata {
    DecoderConfig config;
    std::uint32_t vocabulary_version = 0;
    std::vector<ModelTensorMetadata> tensors;
    std::size_t payload_bytes = 0;
    std::size_t file_bytes = 0;
};

// Parses and fully validates a version-1 container without allocating runtime
// tensors. Payloads must be canonical FP32 decoder parameters, non-overlapping,
// contiguous, and exactly cover the file suffix.
ModelFileMetadata inspect_model_file(const std::string& path, ModelFileLimits limits = {});

struct LoadedModel {
    ParameterTable parameters;
    std::uint32_t vocabulary_version = 0;

    const DecoderConfig& config() const noexcept { return parameters.config(); }
};

// Deterministic offline writer and checked loader for the version-1 container.
// The loader allocates only after inspect_model_file has enforced its budget.
void write_model_file(const std::string& path, const ParameterTable& parameters,
                      std::uint32_t vocabulary_version = BYTE_VOCABULARY_VERSION);
LoadedModel load_model_file(const std::string& path, ModelFileLimits limits = {});

} // namespace model
