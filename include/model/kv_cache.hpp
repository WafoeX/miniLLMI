#pragma once

#include "model/config.hpp"
#include "runtime/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace model {
// Persistent per-layer K/V state for the frozen batch=1 decoder. Physical
// sequence order is newest-to-oldest so a just-written slot is the first
// element of the active range; this lets an ordered graph COPY version the
// whole active range without a hidden concat/copy. Attention is permutation
// invariant over a complete key/value set, so this order preserves results.
class KVCache final {
public:
    explicit KVCache(const DecoderConfig& config, runtime::Device device = runtime::Device{});

    const DecoderConfig& config() const noexcept { return config_; }
    runtime::Device device() const noexcept { return device_; }
    std::int64_t capacity() const noexcept { return capacity_; }
    std::int64_t valid_length() const noexcept { return valid_length_; }
    std::size_t bytes_per_kind() const noexcept { return key_storage_->nbytes(); }
    std::size_t persistent_bytes() const noexcept { return key_storage_->nbytes() + value_storage_->nbytes(); }

    // State bindings for graph-visible COPY writes. Position is a logical
    // chronological token position and must fit the fixed capacity.
    runtime::Tensor key_slot(std::int64_t layer, std::int64_t position) const;
    runtime::Tensor value_slot(std::int64_t layer, std::int64_t position) const;
    // Returns the valid logical prefix in physical newest-to-oldest order.
    runtime::Tensor key_range(std::int64_t layer, std::int64_t length) const;
    runtime::Tensor value_range(std::int64_t layer, std::int64_t length) const;

    // Called only after every layer's graph-visible K/V writes succeeded.
    // Validation happens before changing the active length.
    void commit_append(std::int64_t expected_position, std::int64_t count);
    void reset() noexcept { valid_length_ = 0; }

private:
    runtime::Tensor slot(const runtime::Tensor& storage, std::int64_t layer, std::int64_t position) const;
    runtime::Tensor range(const runtime::Tensor& storage, std::int64_t layer, std::int64_t length) const;
    void check_layer(std::int64_t layer) const;

    DecoderConfig config_;
    runtime::Device device_;
    std::int64_t capacity_;
    std::int64_t valid_length_ = 0;
    std::optional<runtime::Tensor> key_storage_;
    std::optional<runtime::Tensor> value_storage_;
};
} // namespace model
