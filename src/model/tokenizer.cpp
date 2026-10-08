#include "model/tokenizer.hpp"

#include "model/model_file.hpp"

#include <stdexcept>

namespace model {
namespace {
void require_supported(const DecoderConfig& config, std::uint32_t vocabulary_version) {
    const auto status = config.validate();
    if (!status.ok()) throw std::invalid_argument("byte tokenizer: " + status.message);
    if (vocabulary_version != BYTE_VOCABULARY_VERSION)
        throw std::invalid_argument("byte tokenizer: unsupported vocabulary version");
    if (config.vocab != ByteTokenizer::VOCAB_SIZE)
        throw std::invalid_argument("byte tokenizer: vocabulary size must be 258 for version 1");
}

bool is_continuation(std::uint8_t value) { return (value & 0xc0U) == 0x80U; }

std::size_t valid_utf8_length(const std::vector<std::uint8_t>& bytes, std::size_t index) {
    const auto first = bytes[index];
    if (first <= 0x7fU) return 1;
    std::size_t length = 0;
    std::uint32_t codepoint = 0;
    if (first >= 0xc2U && first <= 0xdfU) {
        length = 2;
        codepoint = first & 0x1fU;
    } else if (first >= 0xe0U && first <= 0xefU) {
        length = 3;
        codepoint = first & 0x0fU;
    } else if (first >= 0xf0U && first <= 0xf4U) {
        length = 4;
        codepoint = first & 0x07U;
    } else {
        return 0;
    }
    if (bytes.size() - index < length) return 0;
    for (std::size_t offset = 1; offset < length; ++offset) {
        if (!is_continuation(bytes[index + offset])) return 0;
        codepoint = (codepoint << 6U) | (bytes[index + offset] & 0x3fU);
    }
    if ((length == 2 && codepoint < 0x80U) || (length == 3 && codepoint < 0x800U) ||
        (length == 4 && codepoint < 0x10000U) || (codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
        codepoint > 0x10ffffU)
        return 0;
    return length;
}
} // namespace

ByteTokenizer::ByteTokenizer(const DecoderConfig& config, std::uint32_t vocabulary_version)
    : vocabulary_version_(vocabulary_version) {
    require_supported(config, vocabulary_version_);
}

std::vector<std::int32_t> ByteTokenizer::encode(const std::vector<std::uint8_t>& bytes,
                                                 TokenizeOptions options) const {
    std::vector<std::int32_t> tokens;
    tokens.reserve(bytes.size() + (options.prepend_bos ? 1U : 0U) + (options.append_eos ? 1U : 0U));
    if (options.prepend_bos) tokens.push_back(BOS_ID);
    for (const auto byte : bytes) tokens.push_back(static_cast<std::int32_t>(byte));
    if (options.append_eos) tokens.push_back(EOS_ID);
    return tokens;
}

std::vector<std::uint8_t> ByteTokenizer::decode(const std::vector<std::int32_t>& tokens,
                                                 bool skip_special) const {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(tokens.size());
    for (const auto token : tokens) {
        if (token >= 0 && token <= 255) {
            bytes.push_back(static_cast<std::uint8_t>(token));
        } else if (token == BOS_ID || token == EOS_ID) {
            if (!skip_special) throw std::invalid_argument("byte tokenizer: special token has no byte decode");
        } else {
            throw std::out_of_range("byte tokenizer: token ID is outside v1 vocabulary");
        }
    }
    return bytes;
}

std::string ByteTokenizer::decode_for_display(const std::vector<std::int32_t>& tokens) const {
    const auto bytes = decode(tokens);
    std::string display;
    display.reserve(bytes.size());
    for (std::size_t index = 0; index < bytes.size();) {
        const auto length = valid_utf8_length(bytes, index);
        if (length == 0) {
            display += "\xef\xbf\xbd";
            ++index;
        } else {
            display.append(reinterpret_cast<const char*>(bytes.data() + index), length);
            index += length;
        }
    }
    return display;
}

} // namespace model
