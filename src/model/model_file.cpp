#include "model/model_file.hpp"

#include "model/quantization.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace model {
namespace {

constexpr std::array<unsigned char, 8> MAGIC{{'M', 'L', 'L', 'M', 'R', 'T', 'F', 0}};
constexpr std::size_t MAX_NAME_BYTES = 128;
constexpr std::uint64_t FNV_OFFSET = 1469598103934665603ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("model file: " + message);
}

std::size_t as_size(std::uint64_t value, const char* field) {
    if (value > std::numeric_limits<std::size_t>::max()) invalid(std::string(field) + " exceeds size_t");
    return static_cast<std::size_t>(value);
}

std::uint8_t read_u8(std::istream& input, const char* field) {
    const auto value = input.get();
    if (value == std::char_traits<char>::eof()) invalid(std::string("truncated ") + field);
    return static_cast<std::uint8_t>(value);
}

std::uint32_t read_u32(std::istream& input, const char* field) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<std::uint32_t>(read_u8(input, field)) << shift;
    return value;
}

std::uint64_t read_u64(std::istream& input, const char* field) {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<std::uint64_t>(read_u8(input, field)) << shift;
    return value;
}

std::int64_t read_i64(std::istream& input, const char* field) {
    const auto value = read_u64(input, field);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        invalid(std::string(field) + " is outside int64 range");
    return static_cast<std::int64_t>(value);
}

double read_f64(std::istream& input, const char* field) {
    const auto bits = read_u64(input, field);
    double value = 0;
    static_assert(sizeof(value) == sizeof(bits), "double must be IEEE-754-sized");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::size_t stream_position(std::istream& input) {
    const auto position = input.tellg();
    if (position < 0) invalid("stream position is unavailable");
    return as_size(static_cast<std::uint64_t>(position), "stream position");
}

std::size_t file_size(std::ifstream& input) {
    input.seekg(0, std::ios::end);
    const auto size = stream_position(input);
    input.seekg(0, std::ios::beg);
    return size;
}

runtime::DType dtype_from_tag(std::uint32_t tag) {
    if (tag == 1) return runtime::DType::FP32;
    if (tag == 2) return runtime::DType::INT8;
    invalid("unsupported tensor dtype");
}

std::uint32_t dtype_tag(runtime::DType dtype) {
    if (dtype == runtime::DType::FP32) return 1;
    if (dtype == runtime::DType::INT8) return 2;
    throw std::invalid_argument("model file: unsupported serialized dtype");
}

DecoderConfig read_config(std::istream& input) {
    DecoderConfig config;
    config.batch = read_i64(input, "config.batch");
    config.layers = read_i64(input, "config.layers");
    config.hidden = read_i64(input, "config.hidden");
    config.heads = read_i64(input, "config.heads");
    config.kv_heads = read_i64(input, "config.kv_heads");
    config.head_dim = read_i64(input, "config.head_dim");
    config.ffn = read_i64(input, "config.ffn");
    config.vocab = read_i64(input, "config.vocab");
    config.max_seq = read_i64(input, "config.max_seq");
    config.rms_epsilon = read_f64(input, "config.rms_epsilon");
    config.rope_base = read_f64(input, "config.rope_base");
    const auto bias = read_u8(input, "config.bias");
    if (bias > 1) invalid("config.bias is not boolean");
    config.bias = bias != 0;
    for (std::size_t index = 0; index < 7; ++index)
        if (read_u8(input, "config reserved bytes") != 0) invalid("config reserved bytes are nonzero");
    const auto status = config.validate();
    if (!status.ok()) invalid(status.message);
    return config;
}

ModelTensorMetadata read_tensor_metadata(std::istream& input, bool v2) {
    const auto name_bytes = read_u32(input, "tensor name length");
    if (name_bytes == 0 || name_bytes > MAX_NAME_BYTES) invalid("tensor name length is invalid");
    std::string name(name_bytes, '\0');
    input.read(name.data(), static_cast<std::streamsize>(name_bytes));
    if (!input) invalid("truncated tensor name");
    const auto dtype = dtype_from_tag(read_u32(input, "tensor dtype"));
    const auto rank = read_u32(input, "tensor rank");
    if (rank > runtime::MAX_RANK) invalid("tensor rank exceeds runtime maximum");
    const auto offset_bytes = as_size(read_u64(input, "tensor offset"), "tensor offset");
    const auto nbytes = as_size(read_u64(input, "tensor byte length"), "tensor byte length");
    std::vector<std::int64_t> dimensions;
    dimensions.reserve(rank);
    for (std::uint32_t axis = 0; axis < rank; ++axis)
        dimensions.push_back(read_i64(input, "tensor dimension"));

    ModelTensorMetadata result{std::move(name), dtype, runtime::Shape(std::move(dimensions)), offset_bytes, nbytes,
                               0, std::nullopt};
    if (!v2) return result;

    const auto quantization_version = read_u32(input, "quantization version");
    if (quantization_version == 0) {
        if (dtype != runtime::DType::FP32) invalid("INT8 tensor is missing quantization metadata");
    } else if (quantization_version == INT8_QUANTIZATION_VERSION) {
        if (dtype != runtime::DType::INT8) invalid("quantization metadata requires INT8 value payload");
        const auto output_axis = read_i64(input, "quantization output axis");
        if (output_axis < 0) invalid("quantization output axis is negative");
        const auto scales_dtype = dtype_from_tag(read_u32(input, "quantization scales dtype"));
        const auto scales_rank = read_u32(input, "quantization scales rank");
        if (scales_rank > runtime::MAX_RANK) invalid("quantization scales rank exceeds runtime maximum");
        const auto scales_offset = as_size(read_u64(input, "quantization scales offset"), "quantization scales offset");
        const auto scales_nbytes = as_size(read_u64(input, "quantization scales byte length"),
                                           "quantization scales byte length");
        std::vector<std::int64_t> scales_shape;
        scales_shape.reserve(scales_rank);
        for (std::uint32_t axis = 0; axis < scales_rank; ++axis)
            scales_shape.push_back(read_i64(input, "quantization scales dimension"));
        result.quantization = ModelQuantizationMetadata{quantization_version,
            static_cast<std::size_t>(output_axis), scales_dtype, runtime::Shape(std::move(scales_shape)),
            scales_offset, scales_nbytes};
    } else {
        invalid("unsupported quantization version");
    }
    result.checksum = read_u64(input, "tensor checksum");
    return result;
}

struct PayloadInterval {
    std::size_t offset;
    std::size_t nbytes;
};

std::uint64_t checksum_bytes(std::uint64_t hash, const void* data, std::size_t nbytes) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < nbytes; ++index) {
        hash ^= bytes[index];
        hash *= FNV_PRIME;
    }
    return hash;
}

std::uint64_t checksum_payloads(std::istream& input, const std::vector<PayloadInterval>& payloads) {
    std::array<unsigned char, 4096> buffer{};
    auto hash = FNV_OFFSET;
    for (const auto& payload : payloads) {
        input.seekg(static_cast<std::streamoff>(payload.offset), std::ios::beg);
        if (!input) invalid("cannot seek to payload for checksum");
        std::size_t remaining = payload.nbytes;
        while (remaining != 0) {
            const auto chunk = std::min(remaining, buffer.size());
            input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(chunk));
            if (!input) invalid("truncated payload during checksum");
            hash = checksum_bytes(hash, buffer.data(), chunk);
            remaining -= chunk;
        }
    }
    return hash;
}

void validate_canonical_metadata(const std::vector<ModelTensorMetadata>& tensors, const DecoderConfig& config,
                                 std::size_t metadata_end, std::size_t total_file_bytes, ModelFileLimits limits,
                                 std::size_t& payload_bytes, bool v2, std::istream* input) {
    const auto expected = parameter_specs(config);
    if (tensors.size() != expected.size()) invalid("tensor count does not match decoder configuration");

    std::set<std::string> names;
    std::vector<PayloadInterval> intervals;
    payload_bytes = 0;
    for (const auto& tensor : tensors) {
        if (!names.insert(tensor.name).second) invalid("duplicate tensor name '" + tensor.name + "'");
        const auto spec = std::find_if(expected.begin(), expected.end(), [&](const ParameterSpec& candidate) {
            return candidate.name == tensor.name;
        });
        if (spec == expected.end()) invalid("unknown tensor name '" + tensor.name + "'");
        if (tensor.shape != spec->shape) invalid("tensor '" + tensor.name + "' has incompatible shape");
        if (!v2 && tensor.dtype != runtime::DType::FP32) invalid("V1 tensor '" + tensor.name + "' is not FP32");
        if (tensor.dtype != runtime::DType::FP32 && tensor.dtype != runtime::DType::INT8)
            invalid("tensor '" + tensor.name + "' has unsupported dtype");
        const auto expected_bytes = runtime::nbytes(spec->shape, tensor.dtype);
        if (tensor.nbytes != expected_bytes) invalid("tensor '" + tensor.name + "' has incompatible byte length");
        if (tensor.offset_bytes < metadata_end || tensor.offset_bytes > total_file_bytes ||
            tensor.nbytes > total_file_bytes - tensor.offset_bytes)
            invalid("tensor '" + tensor.name + "' payload is outside file bounds");
        try {
            payload_bytes = runtime::checked_add(payload_bytes, tensor.nbytes);
        } catch (const std::overflow_error&) {
            invalid("total payload byte count overflows");
        }
        intervals.push_back({tensor.offset_bytes, tensor.nbytes});

        if (tensor.dtype == runtime::DType::FP32) {
            if (tensor.quantization) invalid("FP32 tensor '" + tensor.name + "' has quantization metadata");
        } else {
            if (!tensor.quantization) invalid("INT8 tensor '" + tensor.name + "' is missing scales");
            if (!is_weight_only_int8_eligible(*spec)) invalid("INT8 tensor '" + tensor.name + "' is ineligible");
            const auto& quantization = *tensor.quantization;
            if (quantization.version != INT8_QUANTIZATION_VERSION || quantization.output_axis != 1 ||
                quantization.scales_dtype != runtime::DType::FP32 ||
                quantization.scales_shape != runtime::Shape({spec->shape[1]}))
                invalid("INT8 tensor '" + tensor.name + "' has incompatible scale metadata");
            if (quantization.scales_nbytes != runtime::nbytes(quantization.scales_shape, runtime::DType::FP32))
                invalid("INT8 tensor '" + tensor.name + "' has incompatible scale byte length");
            const auto expected_scale_offset = runtime::checked_add(tensor.offset_bytes, tensor.nbytes);
            if (quantization.scales_offset_bytes != expected_scale_offset ||
                quantization.scales_offset_bytes > total_file_bytes ||
                quantization.scales_nbytes > total_file_bytes - quantization.scales_offset_bytes)
                invalid("INT8 tensor '" + tensor.name + "' scale payload is outside file bounds");
            try {
                payload_bytes = runtime::checked_add(payload_bytes, quantization.scales_nbytes);
            } catch (const std::overflow_error&) {
                invalid("total payload byte count overflows");
            }
            intervals.push_back({quantization.scales_offset_bytes, quantization.scales_nbytes});
        }
        if (payload_bytes > limits.max_allocation_bytes) invalid("payload exceeds configured allocation budget");
    }
    for (const auto& spec : expected)
        if (names.find(spec.name) == names.end()) invalid("missing tensor name '" + spec.name + "'");

    std::sort(intervals.begin(), intervals.end(), [](const PayloadInterval& left, const PayloadInterval& right) {
        return left.offset < right.offset;
    });
    std::size_t cursor = metadata_end;
    for (const auto& interval : intervals) {
        if (interval.offset != cursor) invalid("payloads overlap or leave a gap");
        try {
            cursor = runtime::checked_add(cursor, interval.nbytes);
        } catch (const std::overflow_error&) {
            invalid("payload end overflows");
        }
    }
    if (cursor != total_file_bytes) invalid("trailing bytes after payloads");

    if (v2) {
        if (!input) invalid("V2 checksum input is unavailable");
        for (const auto& tensor : tensors) {
            std::vector<PayloadInterval> payloads{{tensor.offset_bytes, tensor.nbytes}};
            if (tensor.quantization)
                payloads.push_back({tensor.quantization->scales_offset_bytes, tensor.quantization->scales_nbytes});
            if (checksum_payloads(*input, payloads) != tensor.checksum)
                invalid("tensor '" + tensor.name + "' checksum mismatch");
        }
    }
}

void write_u8(std::ostream& output, std::uint8_t value) {
    output.put(static_cast<char>(value));
}

void write_u32(std::ostream& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) write_u8(output, static_cast<std::uint8_t>(value >> shift));
}

void write_u64(std::ostream& output, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) write_u8(output, static_cast<std::uint8_t>(value >> shift));
}

void write_i64(std::ostream& output, std::int64_t value) {
    write_u64(output, static_cast<std::uint64_t>(value));
}

void write_f64(std::ostream& output, double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64(output, bits);
}

void write_config(std::ostream& output, const DecoderConfig& config) {
    for (const auto value : {config.batch, config.layers, config.hidden, config.heads, config.kv_heads,
                             config.head_dim, config.ffn, config.vocab, config.max_seq})
        write_i64(output, value);
    write_f64(output, config.rms_epsilon);
    write_f64(output, config.rope_base);
    write_u8(output, config.bias ? 1 : 0);
    for (std::size_t index = 0; index < 7; ++index) write_u8(output, 0);
}

std::size_t descriptor_bytes(const ModelTensorMetadata& tensor, bool v2) {
    std::size_t bytes = 4;
    bytes = runtime::checked_add(bytes, tensor.name.size());
    bytes = runtime::checked_add(bytes, 4 + 4 + 8 + 8);
    bytes = runtime::checked_add(bytes, runtime::checked_mul(tensor.shape.rank(), 8));
    if (!v2) return bytes;
    bytes = runtime::checked_add(bytes, 4); // quantization version
    if (tensor.quantization) {
        bytes = runtime::checked_add(bytes, 8 + 4 + 4 + 8 + 8);
        bytes = runtime::checked_add(bytes, runtime::checked_mul(tensor.quantization->scales_shape.rank(), 8));
    }
    return runtime::checked_add(bytes, 8); // checksum
}

std::streamsize as_streamsize(std::size_t value, const char* field) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error(std::string(field) + " exceeds streamsize");
    return static_cast<std::streamsize>(value);
}

void write_metadata(std::ostream& output, const ModelTensorMetadata& tensor, bool v2) {
    write_u32(output, static_cast<std::uint32_t>(tensor.name.size()));
    output.write(tensor.name.data(), as_streamsize(tensor.name.size(), "tensor name"));
    write_u32(output, dtype_tag(tensor.dtype));
    write_u32(output, static_cast<std::uint32_t>(tensor.shape.rank()));
    write_u64(output, tensor.offset_bytes);
    write_u64(output, tensor.nbytes);
    for (const auto dimension : tensor.shape.values()) write_i64(output, dimension);
    if (!v2) return;
    write_u32(output, tensor.quantization ? tensor.quantization->version : 0);
    if (tensor.quantization) {
        const auto& quantization = *tensor.quantization;
        write_i64(output, static_cast<std::int64_t>(quantization.output_axis));
        write_u32(output, dtype_tag(quantization.scales_dtype));
        write_u32(output, static_cast<std::uint32_t>(quantization.scales_shape.rank()));
        write_u64(output, quantization.scales_offset_bytes);
        write_u64(output, quantization.scales_nbytes);
        for (const auto dimension : quantization.scales_shape.values()) write_i64(output, dimension);
    }
    write_u64(output, tensor.checksum);
}

void write_header(std::ostream& output, std::uint32_t version, std::uint32_t vocabulary_version,
                  const DecoderConfig& config, const std::vector<ModelTensorMetadata>& metadata) {
    for (const auto value : MAGIC) write_u8(output, value);
    write_u32(output, version);
    write_u32(output, vocabulary_version);
    write_u32(output, static_cast<std::uint32_t>(metadata.size()));
    write_config(output, config);
    for (const auto& tensor : metadata) write_metadata(output, tensor, version == MODEL_FILE_VERSION_V2);
}

std::uint64_t checksum_tensor_payload(const runtime::Tensor& value,
                                      const std::optional<QuantizedTensor>& quantized) {
    auto hash = FNV_OFFSET;
    if (quantized) {
        hash = checksum_bytes(hash, quantized->values.data<std::int8_t>(), quantized->values.nbytes());
        return checksum_bytes(hash, quantized->scales.data<float>(), quantized->scales.nbytes());
    }
    return checksum_bytes(hash, value.data<float>(), value.nbytes());
}

} // namespace

ModelFileMetadata inspect_model_file(const std::string& path, ModelFileLimits limits) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("model file: cannot open '" + path + "'");
    const auto total_file_bytes = file_size(input);
    for (const auto expected : MAGIC)
        if (read_u8(input, "magic") != expected) invalid("bad magic");
    const auto format_version = read_u32(input, "format version");
    if (format_version != MODEL_FILE_VERSION_V1 && format_version != MODEL_FILE_VERSION_V2)
        invalid("unsupported format version");
    const auto vocabulary_version = read_u32(input, "vocabulary version");
    if (vocabulary_version != BYTE_VOCABULARY_VERSION) invalid("unsupported vocabulary version");
    const auto tensor_count = read_u32(input, "tensor count");
    const auto config = read_config(input);
    if (config.vocab != 258) invalid("byte vocabulary version requires vocab=258");
    const auto expected = parameter_specs(config);
    if (tensor_count != expected.size()) invalid("tensor count does not match decoder configuration");

    std::vector<ModelTensorMetadata> tensors;
    tensors.reserve(tensor_count);
    try {
        for (std::uint32_t index = 0; index < tensor_count; ++index)
            tensors.push_back(read_tensor_metadata(input, format_version == MODEL_FILE_VERSION_V2));
    } catch (const std::exception& error) {
        invalid(error.what());
    }

    const auto metadata_end = stream_position(input);
    std::size_t payload_bytes = 0;
    validate_canonical_metadata(tensors, config, metadata_end, total_file_bytes, limits, payload_bytes,
                                format_version == MODEL_FILE_VERSION_V2, &input);
    return {config, format_version, vocabulary_version, std::move(tensors), payload_bytes, total_file_bytes};
}

void write_model_file(const std::string& path, const ParameterTable& parameters,
                      std::uint32_t vocabulary_version) {
    const auto& config = parameters.config();
    if (vocabulary_version != BYTE_VOCABULARY_VERSION)
        throw std::invalid_argument("model file: unsupported vocabulary version for writer");
    if (config.vocab != 258) throw std::invalid_argument("model file: byte vocabulary requires vocab=258");
    const auto specs = parameter_specs(config);
    constexpr std::size_t fixed_header_bytes = 8 + 4 + 4 + 4 + 9 * 8 + 2 * 8 + 1 + 7;
    std::size_t metadata_bytes = 0;
    for (const auto& spec : specs)
        metadata_bytes = runtime::checked_add(metadata_bytes,
            descriptor_bytes({spec.name, runtime::DType::FP32, spec.shape, 0, 0, 0, std::nullopt}, false));
    auto payload_offset = runtime::checked_add(fixed_header_bytes, metadata_bytes);

    std::vector<ModelTensorMetadata> metadata;
    metadata.reserve(specs.size());
    for (const auto& spec : specs) {
        const auto bytes = runtime::nbytes(spec.shape, runtime::DType::FP32);
        metadata.push_back({spec.name, runtime::DType::FP32, spec.shape, payload_offset, bytes, 0, std::nullopt});
        payload_offset = runtime::checked_add(payload_offset, bytes);
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("model file: cannot create '" + path + "'");
    write_header(output, MODEL_FILE_VERSION_V1, vocabulary_version, config, metadata);
    for (const auto& spec : specs) {
        const auto& tensor = parameters.at(spec.name);
        output.write(reinterpret_cast<const char*>(tensor.data<float>()),
                     as_streamsize(tensor.nbytes(), "tensor payload"));
    }
    if (!output) throw std::runtime_error("model file: failed while writing '" + path + "'");
}

void write_quantized_model_file(const std::string& path, const ParameterTable& parameters,
                                std::uint32_t vocabulary_version) {
    const auto& config = parameters.config();
    if (vocabulary_version != BYTE_VOCABULARY_VERSION)
        throw std::invalid_argument("model file: unsupported vocabulary version for writer");
    if (config.vocab != 258) throw std::invalid_argument("model file: byte vocabulary requires vocab=258");
    const auto specs = parameter_specs(config);

    struct PendingTensor {
        ParameterSpec spec;
        const runtime::Tensor* source;
        std::optional<QuantizedTensor> quantized;
    };
    std::vector<PendingTensor> pending;
    pending.reserve(specs.size());
    for (const auto& spec : specs) {
        const auto& source = parameters.at(spec.name);
        PendingTensor entry{spec, &source, std::nullopt};
        if (is_weight_only_int8_eligible(spec)) entry.quantized = quantize_per_output_channel(source);
        pending.push_back(std::move(entry));
    }

    constexpr std::size_t fixed_header_bytes = 8 + 4 + 4 + 4 + 9 * 8 + 2 * 8 + 1 + 7;
    std::size_t metadata_bytes = 0;
    for (const auto& entry : pending) {
        ModelTensorMetadata descriptor{entry.spec.name,
            entry.quantized ? runtime::DType::INT8 : runtime::DType::FP32, entry.spec.shape, 0, 0, 0, std::nullopt};
        if (entry.quantized) {
            descriptor.quantization = {INT8_QUANTIZATION_VERSION, 1, runtime::DType::FP32,
                                       entry.quantized->scales.shape()};
        }
        metadata_bytes = runtime::checked_add(metadata_bytes, descriptor_bytes(descriptor, true));
    }
    auto payload_offset = runtime::checked_add(fixed_header_bytes, metadata_bytes);

    std::vector<ModelTensorMetadata> metadata;
    metadata.reserve(pending.size());
    for (const auto& entry : pending) {
        ModelTensorMetadata descriptor{entry.spec.name,
            entry.quantized ? runtime::DType::INT8 : runtime::DType::FP32, entry.spec.shape, 0, 0, 0, std::nullopt};
        descriptor.offset_bytes = payload_offset;
        descriptor.nbytes = entry.quantized ? entry.quantized->values.nbytes() : entry.source->nbytes();
        payload_offset = runtime::checked_add(payload_offset, descriptor.nbytes);
        if (entry.quantized) {
            ModelQuantizationMetadata quantization{INT8_QUANTIZATION_VERSION, 1, runtime::DType::FP32,
                entry.quantized->scales.shape(), payload_offset, entry.quantized->scales.nbytes()};
            descriptor.quantization = std::move(quantization);
            payload_offset = runtime::checked_add(payload_offset, descriptor.quantization->scales_nbytes);
        }
        descriptor.checksum = checksum_tensor_payload(*entry.source, entry.quantized);
        metadata.push_back(std::move(descriptor));
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("model file: cannot create '" + path + "'");
    write_header(output, MODEL_FILE_VERSION_V2, vocabulary_version, config, metadata);
    for (const auto& entry : pending) {
        if (entry.quantized) {
            output.write(reinterpret_cast<const char*>(entry.quantized->values.data<std::int8_t>()),
                         as_streamsize(entry.quantized->values.nbytes(), "INT8 tensor payload"));
            output.write(reinterpret_cast<const char*>(entry.quantized->scales.data<float>()),
                         as_streamsize(entry.quantized->scales.nbytes(), "scale tensor payload"));
        } else {
            output.write(reinterpret_cast<const char*>(entry.source->data<float>()),
                         as_streamsize(entry.source->nbytes(), "tensor payload"));
        }
    }
    if (!output) throw std::runtime_error("model file: failed while writing '" + path + "'");
}

LoadedModel load_model_file(const std::string& path, ModelFileLimits limits) {
    const auto metadata = inspect_model_file(path, limits);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("model file: cannot open '" + path + "'");
    std::map<std::string, runtime::Tensor> tensors;
    std::vector<PrepareDequantWeight> prepare_dequant_weights;
    for (const auto& descriptor : metadata.tensors) {
        input.seekg(as_streamsize(descriptor.offset_bytes, "tensor offset"), std::ios::beg);
        if (!input) invalid("cannot seek to tensor '" + descriptor.name + "'");
        if (!descriptor.quantization) {
            auto tensor = runtime::Tensor::allocate_cpu(descriptor.shape, runtime::DType::FP32);
            input.read(reinterpret_cast<char*>(tensor.data<float>()), as_streamsize(descriptor.nbytes, "tensor payload"));
            if (!input) invalid("truncated tensor payload '" + descriptor.name + "'");
            tensors.emplace(descriptor.name, std::move(tensor));
            continue;
        }

        auto values = runtime::Tensor::allocate_cpu(descriptor.shape, runtime::DType::INT8);
        input.read(reinterpret_cast<char*>(values.data<std::int8_t>()), as_streamsize(descriptor.nbytes, "INT8 tensor payload"));
        if (!input) invalid("truncated INT8 tensor payload '" + descriptor.name + "'");
        const auto& quantization = *descriptor.quantization;
        auto scales = runtime::Tensor::allocate_cpu(quantization.scales_shape, runtime::DType::FP32);
        input.seekg(as_streamsize(quantization.scales_offset_bytes, "scale tensor offset"), std::ios::beg);
        if (!input) invalid("cannot seek to scales for tensor '" + descriptor.name + "'");
        input.read(reinterpret_cast<char*>(scales.data<float>()), as_streamsize(quantization.scales_nbytes, "scale tensor payload"));
        if (!input) invalid("truncated scale tensor payload '" + descriptor.name + "'");
        QuantizedTensor quantized{std::move(values), std::move(scales), quantization.output_axis};
        validate_per_output_channel_int8(quantized);
        auto dequantized = dequantize_per_output_channel(quantized);
        prepare_dequant_weights.push_back({descriptor.name, std::move(quantized)});
        tensors.emplace(descriptor.name, std::move(dequantized));
    }
    return {ParameterTable(metadata.config, std::move(tensors)), metadata.vocabulary_version,
            std::move(prepare_dequant_weights)};
}

} // namespace model
