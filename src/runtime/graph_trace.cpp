#include "runtime/graph_trace.hpp"
#include <stdexcept>
#include <new>

namespace runtime {
const char* trace_name(TraceKind kind) {
    switch (kind) {
#define TRACE_NAME(name) case TraceKind::name: return #name
    TRACE_NAME(NodeBegin); TRACE_NAME(Tensor); TRACE_NAME(Allocate);
    TRACE_NAME(Alias); TRACE_NAME(StateWrite); TRACE_NAME(Copy);
    TRACE_NAME(NodeEnd); TRACE_NAME(Release); TRACE_NAME(Free);
    TRACE_NAME(Output); TRACE_NAME(Failure); TRACE_NAME(BlockAllocate); TRACE_NAME(BlockFree);
    TRACE_NAME(BackendCPU); TRACE_NAME(BackendCUDA);
#undef TRACE_NAME
    }
    throw std::invalid_argument("unknown trace kind");
}
void ExecutionTrace::record(TraceKind kind, std::optional<NodeId> node, std::optional<TensorId> tensor,
                            std::optional<TensorId> base, const Layout* metadata,
                            std::size_t bytes, StatusCode status) noexcept {
    if (events_.size() < limit_) {
        try {
            events_.push_back({kind, node, tensor, base, metadata ? std::optional<Layout>(*metadata) : std::nullopt, bytes, status});
            return;
        } catch (const std::bad_alloc&) {} // tracing must not turn a valid run into failure
        catch (const std::length_error&) {}
    }
    note_dropped();
}
} // namespace runtime
