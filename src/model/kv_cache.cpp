#include "model/kv_cache.hpp"

#include <stdexcept>

namespace model {
namespace {
using namespace runtime;

std::size_t element_bytes(const DecoderConfig& config) {
    return checked_mul(checked_mul(as_size(config.heads), as_size(config.head_dim)), dtype_size(DType::FP32));
}
} // namespace

KVCache::KVCache(const DecoderConfig& config, Device device)
    : config_(config), device_(device), capacity_(config.max_seq) {
    const auto valid = config_.validate();
    if (!valid.ok()) throw std::invalid_argument(valid.message);
    if (config_.batch != 1) throw std::invalid_argument("KV cache supports fixed batch=1 only");
    if (device_.type() != DeviceType::CPU)
        throw std::invalid_argument("KV cache device allocation requires an explicit backend; Stage 14 cache is CPU-resident");
    const Shape shape{config_.layers, config_.batch, capacity_, config_.heads, config_.head_dim};
    key_storage_.emplace(Tensor::allocate_cpu(shape));
    value_storage_.emplace(Tensor::allocate_cpu(shape));
}

void KVCache::check_layer(std::int64_t layer) const {
    if (layer < 0 || layer >= config_.layers) throw std::out_of_range("KV cache layer is out of range");
}

Tensor KVCache::slot(const Tensor& storage, std::int64_t layer, std::int64_t position) const {
    check_layer(layer);
    if (position < 0 || position >= capacity_) throw std::out_of_range("KV cache append position is out of range");
    // Reverse physical order: logical p is stored at capacity-1-p.
    const auto physical_sequence = capacity_ - position - 1;
    const auto per_sequence = element_bytes(config_);
    const auto layer_bytes = checked_mul(as_size(capacity_), per_sequence);
    const auto offset = checked_add(checked_mul(as_size(layer), layer_bytes),
                                    checked_mul(as_size(physical_sequence), per_sequence));
    return storage.view({1, config_.heads, config_.head_dim}, contiguous_stride({1, config_.heads, config_.head_dim}), offset);
}

Tensor KVCache::range(const Tensor& storage, std::int64_t layer, std::int64_t length) const {
    check_layer(layer);
    if (length <= 0 || length > valid_length_) throw std::out_of_range("KV cache active range is outside valid positions");
    const auto per_sequence = element_bytes(config_);
    const auto layer_bytes = checked_mul(as_size(capacity_), per_sequence);
    const auto physical_start = capacity_ - length;
    const auto offset = checked_add(checked_mul(as_size(layer), layer_bytes),
                                    checked_mul(as_size(physical_start), per_sequence));
    const Shape shape{length, config_.heads, config_.head_dim};
    return storage.view(shape, contiguous_stride(shape), offset);
}

Tensor KVCache::key_slot(std::int64_t layer, std::int64_t position) const { return slot(*key_storage_, layer, position); }
Tensor KVCache::value_slot(std::int64_t layer, std::int64_t position) const { return slot(*value_storage_, layer, position); }
Tensor KVCache::key_range(std::int64_t layer, std::int64_t length) const { return range(*key_storage_, layer, length); }
Tensor KVCache::value_range(std::int64_t layer, std::int64_t length) const { return range(*value_storage_, layer, length); }

void KVCache::commit_append(std::int64_t expected_position, std::int64_t count) {
    if (expected_position != valid_length_) throw std::invalid_argument("KV cache position changed before commit");
    if (count <= 0 || count > capacity_ - valid_length_) throw std::out_of_range("KV cache capacity overflow before writes commit");
    valid_length_ += count;
}
} // namespace model
