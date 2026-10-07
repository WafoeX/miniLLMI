#pragma once

#include "runtime/arena.hpp"
#include "runtime/shape_inference.hpp"

namespace runtime {
class Graph;
// One-threaded, per-execution allocation seam. Only physical roots are released,
// never individual alias views. begin/end bracket all calls; no hidden fallback.
class AllocationProvider {
public:
    virtual ~AllocationProvider() = default;
    // Prepared providers reject incompatible graph metadata before any writes.
    virtual Status validate_graph(const Graph&) const { return Status::success(); }
    virtual Status begin() = 0;
    virtual void end() noexcept = 0;
    virtual Tensor allocate(TensorId root, Shape shape, DType dtype) = 0;
    virtual void release(TensorId root) noexcept = 0;
    virtual bool backing_per_request() const noexcept = 0;
    virtual std::size_t capacity() const noexcept { return 0; }
    virtual std::size_t capacity_for(TensorId) const noexcept { return capacity(); }
};
class DynamicAllocationProvider final : public AllocationProvider {
public:
    Status begin() override { return Status::success(); }
    void end() noexcept override {}
    Tensor allocate(TensorId, Shape shape, DType dtype) override { return Tensor::allocate_cpu(std::move(shape), dtype); }
    void release(TensorId) noexcept override {}
    bool backing_per_request() const noexcept override { return true; }
};
class ArenaAllocationProvider final : public AllocationProvider {
public:
    explicit ArenaAllocationProvider(std::size_t capacity_bytes, std::size_t alignment = 64)
        : arena_(capacity_bytes, alignment) {}
    Status begin() override;
    void end() noexcept override { running_ = false; }
    Tensor allocate(TensorId root, Shape shape, DType dtype) override;
    void release(TensorId root) noexcept override;
    bool backing_per_request() const noexcept override { return false; }
    std::size_t capacity() const noexcept override { return arena_.capacity(); }
    const Arena& arena() const noexcept { return arena_; }
private:
    Arena arena_;
    std::map<TensorId, Block> blocks_;
    bool running_ = false;
};
} // namespace runtime
