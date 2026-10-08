#include "model/model_file.hpp"

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

void validate_canonical_metadata(const std::vector<ModelTensorMetadata>& tensors,
                                 const DecoderConfig& config, std::size_t metadata_end,
                                 std::size_t total_file_bytes, ModelFileLimits limits,
                                 std::size_t& payload_bytes) {
    const auto expected = parameter_specs(config);
    if (tensors.size() != expected.size()) invalid("tensor count does not match decoder configuration");

    std::set<std::string> names;
    std::vector<const ModelTensorMetadata*> intervals;
    intervals.reserve(tensors.size());
    payload_bytes = 0;
    for (const auto& tensor : tensors) {
        if (!names.insert(tensor.name).second) invalid("duplicate tensor name '" + tensor.name + "'");
        const auto spec = std::find_if(expected.begin(), expected.end(), [&](const ParameterSpec& candidate) {
            return candidate.name == tensor.name;
        });
        if (spec == expected.end()) invalid("unknown tensor name '" + tensor.name + "'");
        if (tensor.dtype != runtime::DType::FP32) invalid("tensor '" + tensor.name + "' is not FP32");
        if (tensor.shape != spec->shape) invalid("tensor '" + tensor.name + "' has incompatible shape");
        const auto expected_bytes = runtime::nbytes(spec->shape, runtime::DType::FP32);
        if (tensor.nbytes != expected_bytes) invalid("tensor '" + tensor.name + "' has incompatible byte length");
        if (tensor.offset_bytes < metadata_end || tensor.offset_bytes > total_file_bytes ||
            tensor.nbytes > total_file_bytes - tensor.offset_bytes)
            invalid("tensor '" + tensor.name + "' payload is outside file bounds");
        try {
            payload_bytes = runtime::checked_add(payload_bytes, tensor.nbytes);
        } catch (const std::overflow_error&) {
            invalid("total payload byte count overflows");
        }
        if (payload_bytes > limits.max_allocation_bytes) invalid("payload exceeds configured allocation budget");
        intervals.push_back(&tensor);
    }
    for (const auto& spec : expected)
        if (names.find(spec.name) == names.end()) invalid("missing tensor name '" + spec.name + "'");

    std::sort(intervals.begin(), intervals.end(), [](const ModelTensorMetadata* left,
                                                       const ModelTensorMetadata* right) {
        return left->offset_bytes < right->offset_bytes;
    });
    std::size_t cursor = metadata_end;
    for (const auto* tensor : intervals) {
        if (tensor->offset_bytes != cursor) invalid("payloads overlap or leave a gap");
        try {
            cursor = runtime::checked_add(cursor, tensor->nbytes);
        } catch (const std::overflow_error&) {
            invalid("payload end overflows");
        }
    }
    if (cursor != total_file_bytes) invalid("trailing bytes after payloads");
}
} // namespace

ModelFileMetadata inspect_model_file(const std::string& path, ModelFileLimits limits) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("model file: cannot open '" + path + "'");
    const auto total_file_bytes = file_size(input);
    for (const auto expected : MAGIC)
        if (read_u8(input, "magic") != expected) invalid("bad magic");
    if (read_u32(input, "format version") != MODEL_FILE_VERSION) invalid("unsupported format version");
    const auto vocabulary_version = read_u32(input, "vocabulary version");
    if (vocabulary_version != BYTE_VOCABULARY_VERSION) invalid("unsupported vocabulary version");
    const auto tensor_count = read_u32(input, "tensor count");
    const auto config = read_config(input);
    if (config.vocab != 258) invalid("byte vocabulary version requires vocab=258");
    const auto expected = parameter_specs(config);
    if (tensor_count != expected.size()) invalid("tensor count does not match decoder configuration");

    std::vector<ModelTensorMetadata> tensors;
    tensors.reserve(tensor_count);
    for (std::uint32_t index = 0; index < tensor_count; ++index) {
        const auto name_bytes = read_u32(input, "tensor name length");
        if (name_bytes == 0 || name_bytes > MAX_NAME_BYTES) invalid("tensor name length is invalid");
        std::string name(name_bytes, '\0');
        input.read(name.data(), static_cast<std::streamsize>(name_bytes));
        if (!input) invalid("truncated tensor name");
        const auto dtype_tag = read_u32(input, "tensor dtype");
        if (dtype_tag != 1) invalid("unsupported tensor dtype");
        const auto rank = read_u32(input, "tensor rank");
        if (rank > runtime::MAX_RANK) invalid("tensor rank exceeds runtime maximum");
        const auto offset_bytes = as_size(read_u64(input, "tensor offset"), "tensor offset");
        const auto nbytes = as_size(read_u64(input, "tensor byte length"), "tensor byte length");
        std::vector<std::int64_t> dimensions;
        dimensions.reserve(rank);
        for (std::uint32_t axis = 0; axis < rank; ++axis)
            dimensions.push_back(read_i64(input, "tensor dimension"));
        try {
            tensors.push_back({std::move(name), runtime::DType::FP32, runtime::Shape(std::move(dimensions)),
                               offset_bytes, nbytes});
        } catch (const std::exception& error) {
            invalid(error.what());
        }
    }

    const auto metadata_end = stream_position(input);
    std::size_t payload_bytes = 0;
    validate_canonical_metadata(tensors, config, metadata_end, total_file_bytes, limits, payload_bytes);
    return {config, vocabulary_version, std::move(tensors), payload_bytes, total_file_bytes};
}

namespace {
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

std::size_t checked_metadata_bytes(const std::vector<ParameterSpec>& specs) {
    std::size_t bytes = 0;
    for (const auto& spec : specs) {
        bytes = runtime::checked_add(bytes, 4);
        bytes = runtime::checked_add(bytes, spec.name.size());
        bytes = runtime::checked_add(bytes, 4 + 4 + 8 + 8);
        bytes = runtime::checked_add(bytes, runtime::checked_mul(spec.shape.rank(), 8));
    }
    return bytes;
}

std::streamsize as_streamsize(std::size_t value, const char* field) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error(std::string(field) + " exceeds streamsize");
    return static_cast<std::streamsize>(value);
}
} // namespace

void write_model_file(const std::string& path, const ParameterTable& parameters,
                      std::uint32_t vocabulary_version) {
    const auto& config = parameters.config();
    if (vocabulary_version != BYTE_VOCABULARY_VERSION)
        throw std::invalid_argument("model file: unsupported vocabulary version for writer");
    if (config.vocab != 258) throw std::invalid_argument("model file: byte vocabulary requires vocab=258");
    const auto specs = parameter_specs(config);
    const auto metadata_bytes = checked_metadata_bytes(specs);
    constexpr std::size_t fixed_header_bytes = 8 + 4 + 4 + 4 + 9 * 8 + 2 * 8 + 1 + 7;
    auto payload_offset = runtime::checked_add(fixed_header_bytes, metadata_bytes);

    std::vector<ModelTensorMetadata> metadata;
    metadata.reserve(specs.size());
    for (const auto& spec : specs) {
        const auto bytes = runtime::nbytes(spec.shape, runtime::DType::FP32);
        metadata.push_back({spec.name, runtime::DType::FP32, spec.shape, payload_offset, bytes});
        payload_offset = runtime::checked_add(payload_offset, bytes);
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("model file: cannot create '" + path + "'");
    for (const auto value : MAGIC) write_u8(output, value);
    write_u32(output, MODEL_FILE_VERSION);
    write_u32(output, vocabulary_version);
    write_u32(output, static_cast<std::uint32_t>(metadata.size()));
    write_config(output, config);
    for (const auto& tensor : metadata) {
        write_u32(output, static_cast<std::uint32_t>(tensor.name.size()));
        output.write(tensor.name.data(), as_streamsize(tensor.name.size(), "tensor name"));
        write_u32(output, 1);
        write_u32(output, static_cast<std::uint32_t>(tensor.shape.rank()));
        write_u64(output, tensor.offset_bytes);
        write_u64(output, tensor.nbytes);
        for (const auto dimension : tensor.shape.values()) write_i64(output, dimension);
    }
    for (const auto& spec : specs) {
        const auto& tensor = parameters.at(spec.name);
        output.write(reinterpret_cast<const char*>(tensor.data<float>()),
                     as_streamsize(tensor.nbytes(), "tensor payload"));
    }
    if (!output) throw std::runtime_error("model file: failed while writing '" + path + "'");
}

LoadedModel load_model_file(const std::string& path, ModelFileLimits limits) {
    const auto metadata = inspect_model_file(path, limits);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("model file: cannot open '" + path + "'");
    std::map<std::string, runtime::Tensor> tensors;
    for (const auto& descriptor : metadata.tensors) {
        auto tensor = runtime::Tensor::allocate_cpu(descriptor.shape, descriptor.dtype);
        input.seekg(as_streamsize(descriptor.offset_bytes, "tensor offset"), std::ios::beg);
        if (!input) invalid("cannot seek to tensor '" + descriptor.name + "'");
        input.read(reinterpret_cast<char*>(tensor.data<float>()), as_streamsize(descriptor.nbytes, "tensor payload"));
        if (!input) invalid("truncated tensor payload '" + descriptor.name + "'");
        tensors.emplace(descriptor.name, std::move(tensor));
    }
    return {ParameterTable(metadata.config, std::move(tensors)), metadata.vocabulary_version};
}

} // namespace model
