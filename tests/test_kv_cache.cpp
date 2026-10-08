#include "model/kv_cache.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
}

int main() {
    try {
        const auto config = model::DecoderConfig::tiny();
        model::KVCache cache(config);
        const std::size_t expected_kind = static_cast<std::size_t>(config.layers * config.batch * config.max_seq *
            config.heads * config.head_dim) * sizeof(float);
        require(cache.device() == runtime::Device{} && cache.capacity() == config.max_seq && cache.valid_length() == 0,
                "KV cache initial metadata");
        require(cache.bytes_per_kind() == expected_kind && cache.persistent_bytes() == 2 * expected_kind,
                "KV cache byte accounting");
        const auto key_zero = cache.key_slot(0, 0);
        const auto key_last = cache.key_slot(0, config.max_seq - 1);
        const auto key_next_layer = cache.key_slot(1, 0);
        require(key_zero.shape() == runtime::Shape({1, config.heads, config.head_dim}) && key_zero.is_contiguous(),
                "KV slot shape/layout");
        require(key_zero.data_offset() > key_last.data_offset() && key_zero.storage() == key_last.storage(),
                "KV physical order is newest-first");
        require(key_zero.storage() != key_next_layer.storage() || key_zero.data_offset() != key_next_layer.data_offset(),
                "KV layers are isolated");
        try {
            (void)cache.key_range(0, 1);
            throw std::runtime_error("invalid range accepted");
        } catch (const std::out_of_range&) {}
        cache.commit_append(0, 4);
        const auto before_storage = key_zero.storage();
        const auto active = cache.key_range(0, 4);
        require(cache.valid_length() == 4 && active.shape() == runtime::Shape({4, config.heads, config.head_dim}) &&
                    active.storage() == before_storage && active.is_contiguous(), "active range metadata");
        try {
            cache.commit_append(3, 1);
            throw std::runtime_error("stale cache position accepted");
        } catch (const std::invalid_argument&) {}
        try {
            cache.commit_append(4, config.max_seq);
            throw std::runtime_error("overflow cache commit accepted");
        } catch (const std::out_of_range&) {}
        require(cache.valid_length() == 4, "failed commit changed valid length");
        cache.reset();
        require(cache.valid_length() == 0 && cache.key_slot(0, 0).storage() == before_storage,
                "reset invalidates ranges without reallocating cache");
        try {
            model::KVCache unsupported(config, runtime::Device(runtime::DeviceType::CUDA, 0));
            throw std::runtime_error("unsupported cache device accepted");
        } catch (const std::invalid_argument&) {}
        std::cout << "Stage 14 C1 KV cache metadata/storage: PASS bytes=" << cache.persistent_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_kv_cache: " << error.what() << '\n';
        return 1;
    }
}
