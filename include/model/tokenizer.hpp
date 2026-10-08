#pragma once

#include "model/config.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace model {

struct TokenizeOptions {
    // The v1 prompt policy is BOS-on, EOS-off. Callers opt into a terminal EOS.
    bool prepend_bos = true;
    bool append_eos = false;
};

class ByteTokenizer {
public:
    static constexpr std::uint32_t VOCABULARY_VERSION = 1;
    static constexpr std::int32_t BOS_ID = 256;
    static constexpr std::int32_t EOS_ID = 257;
    static constexpr std::int64_t VOCAB_SIZE = 258;

    ByteTokenizer(const DecoderConfig& config, std::uint32_t vocabulary_version);

    std::vector<std::int32_t> encode(const std::vector<std::uint8_t>& bytes,
                                     TokenizeOptions options = {}) const;
    // The v1 decode policy drops BOS/EOS. All other IDs must be literal bytes;
    // invalid IDs have no unknown-token fallback and are rejected.
    std::vector<std::uint8_t> decode(const std::vector<std::int32_t>& tokens,
                                     bool skip_special = true) const;
    // Produces valid UTF-8 for terminals/logs by replacing malformed byte
    // sequences with U+FFFD. It never changes the underlying token IDs.
    std::string decode_for_display(const std::vector<std::int32_t>& tokens) const;

    std::uint32_t vocabulary_version() const noexcept { return vocabulary_version_; }

private:
    std::uint32_t vocabulary_version_;
};

} // namespace model
