#pragma once

#include "runtime/graph.hpp"
#include <limits>

namespace runtime {
enum class TraceKind { NodeBegin, Tensor, Allocate, Alias, StateWrite, Copy, NodeEnd, Release, Free, Output, Failure, BlockAllocate, BlockFree, BackendCPU, BackendCUDA };
const char* trace_name(TraceKind kind);
struct TraceEvent {
    TraceKind kind;
    std::optional<NodeId> node;
    std::optional<TensorId> tensor;
    std::optional<TensorId> base;
    std::optional<Layout> metadata;
    std::size_t bytes;
    StatusCode status;
};
// First-N bounded diagnostic events; never owns Tensor/Storage or reads values.
// A full trace or diagnostic allocation failure drops events, not execution.
class ExecutionTrace {
public:
    explicit ExecutionTrace(std::size_t limit = 1024) : limit_(limit) {}
    void reset() noexcept { events_.clear(); dropped_ = 0; }
    std::size_t limit() const noexcept { return limit_; }
    std::size_t dropped() const noexcept { return dropped_; }
    const std::vector<TraceEvent>& events() const noexcept { return events_; }
    void note_dropped() noexcept { if (dropped_ != std::numeric_limits<std::size_t>::max()) ++dropped_; }
    void record(TraceKind kind, std::optional<NodeId> node = {}, std::optional<TensorId> tensor = {},
                std::optional<TensorId> base = {}, const Layout* metadata = nullptr,
                std::size_t bytes = 0, StatusCode status = StatusCode::Ok) noexcept;
private:
    std::size_t limit_, dropped_ = 0;
    std::vector<TraceEvent> events_;
};
} // namespace runtime
