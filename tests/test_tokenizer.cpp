#include "model/tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
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
} // namespace

int main() {
    try {
        const model::ByteTokenizer tokenizer(model::DecoderConfig::tiny(), model::ByteTokenizer::VOCABULARY_VERSION);
        const std::vector<std::uint8_t> ascii{'H', 'i', '!'};
        const auto prompted = tokenizer.encode(ascii);
        require(prompted == std::vector<std::int32_t>({256, 'H', 'i', '!'}), "default BOS prompt policy");
        require(tokenizer.decode(prompted) == ascii, "ASCII bytes round-trip with skipped BOS");

        const std::vector<std::uint8_t> utf8{0xe4, 0xbd, 0xa0, 0xe5, 0xa5, 0xbd}; // 你好
        const auto exact = tokenizer.encode(utf8, {false, true});
        require(exact.back() == model::ByteTokenizer::EOS_ID && tokenizer.decode(exact) == utf8,
                "multibyte UTF-8 round-trip with skipped EOS");
        require(tokenizer.decode_for_display(exact) == "你好", "valid UTF-8 display decode");

        const std::vector<std::uint8_t> invalid{0xc3, 0x28, 0xff};
        const auto invalid_tokens = tokenizer.encode(invalid, {false, false});
        require(tokenizer.decode(invalid_tokens) == invalid, "invalid bytes preserve token IDs exactly");
        require(tokenizer.decode_for_display(invalid_tokens) == "\xef\xbf\xbd(\xef\xbf\xbd",
                "invalid UTF-8 display replacement");
        require_rejected([&] { (void)tokenizer.decode({model::ByteTokenizer::BOS_ID}, false); },
                         "special token decode without skip accepted");
        require_rejected([&] { (void)tokenizer.decode({258}); }, "unknown token ID accepted");

        auto incompatible = model::DecoderConfig::tiny();
        incompatible.vocab = 257;
        require_rejected([&] { (void)model::ByteTokenizer(incompatible, model::ByteTokenizer::VOCABULARY_VERSION); },
                         "vocabulary mismatch accepted");
        require_rejected([&] { (void)model::ByteTokenizer(model::DecoderConfig::tiny(), 2); },
                         "unknown vocabulary version accepted");

        std::cout << "Stage 15 C3 deterministic byte tokenizer: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_tokenizer: " << error.what() << '\n';
        return 1;
    }
}
