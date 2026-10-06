#include "runtime/graph_executor.hpp"
#include "runtime/copy.hpp"
#include <algorithm>
#include <new>
#include <set>

namespace runtime {
ExecutionResult execute_graph(const Graph& graph, ExecutionTrace* trace, AllocationProvider* supplied, const Backend* supplied_backend) {
    ExecutionResult result;
    const auto& backend = supplied_backend ? *supplied_backend : default_cpu_backend();
    DynamicAllocationProvider dynamic;
    auto& provider = supplied ? *supplied : static_cast<AllocationProvider&>(dynamic);
    const auto backing = provider.backing_per_request();
    std::map<TensorId, std::size_t> root_offsets;
    std::optional<std::pair<TensorId, std::size_t>> pending_offset;
    struct Session {
        AllocationProvider& provider;
        bool begun = false;
        ~Session() { if (begun) provider.end(); }
    } session{provider};
    if (trace) trace->reset();
    const auto event = [&](TraceKind kind, std::optional<TensorId> tensor = {}, std::size_t bytes = 0,
                           StatusCode status = StatusCode::Ok) noexcept {
        if (!trace) return; // disabled path does not copy metadata or store events
        try {
            if (tensor) {
                const auto& record = graph.tensors().at(*tensor);
                auto metadata = record.layout;
                if (metadata && record.base) {
                    const auto offset = root_offsets.find(*record.base);
                    if (offset != root_offsets.end() || (pending_offset && pending_offset->first == *record.base)) {
                        metadata->offset_bytes += offset != root_offsets.end() ? offset->second : pending_offset->second;
                        if (!backing) metadata->capacity_bytes = provider.capacity();
                    }
                }
                trace->record(kind, result.failed_node, tensor, record.base, metadata ? &*metadata : nullptr, bytes, status);
            } else trace->record(kind, result.failed_node, {}, {}, nullptr, bytes, status);
        } catch (const std::bad_alloc&) { trace->note_dropped(); }
    };
    if (!graph.frozen()) {
        result.status = Status::failure(StatusCode::InvalidArgument, "executor requires a successfully frozen graph");
        event(TraceKind::Failure, {}, 0, result.status.code);
        return result;
    }
    for (const auto& item : graph.tensors()) {
        if (item.second.device.type() != DeviceType::CPU) {
            result.status = Status::failure(StatusCode::DeviceMismatch, "sequential CPU executor requires explicit CPU tensors; no hidden transfers");
            event(TraceKind::Failure, item.first, 0, result.status.code);
            return result;
        }
    }
    std::map<TensorId, Tensor> live;
    std::map<TensorId, std::size_t> remaining, root_handles, owned_bytes;
    std::set<TensorId> pinned;
    std::optional<TensorId> pending_allocation;
    const auto release = [&](TensorId id) {
        const auto found = live.find(id);
        if (found == live.end()) return;
        const auto base = *graph.tensors().at(id).base;
        event(TraceKind::Release, id);
        live.erase(found);
        if (--root_handles.at(base) == 0 && owned_bytes.count(base)) {
            ++result.counts.releases;
            if (backing) ++result.counts.frees;
            result.counts.live_bytes -= owned_bytes.at(base);
            provider.release(base);
            event(backing ? TraceKind::Free : TraceKind::BlockFree, base, owned_bytes.at(base));
            owned_bytes.erase(base);
        }
    };
    const auto cleanup = [&] {
        result.outputs.clear();
        while (!live.empty()) release(live.begin()->first);
        // Buffer allocation may succeed before a metadata insertion fails.
        // Stack unwinding has freed that unregistered buffer before this call.
        if (pending_allocation) {
            provider.release(*pending_allocation);
            event(backing ? TraceKind::Free : TraceKind::BlockFree, *pending_allocation, graph.tensors().at(*pending_allocation).layout->capacity_bytes);
        }
        result.counts.frees = result.counts.allocations;
        result.counts.releases = result.counts.allocation_requests;
        result.counts.live_bytes = 0;
        owned_bytes.clear();
    };
    const auto insert = [&](TensorId id, Tensor tensor) {
        const auto base = *graph.tensors().at(id).base;
        ++root_handles[base];
        try { live.emplace(id, std::move(tensor)); }
        catch (...) { --root_handles[base]; throw; }
    };
    try {
        result.status = provider.validate_graph(graph);
        if (!result.status.ok()) { event(TraceKind::Failure, {}, 0, result.status.code); return result; }
        result.status = provider.begin();
        if (!result.status.ok()) { event(TraceKind::Failure, {}, 0, result.status.code); return result; }
        session.begun = true;
        result.counts.arena_capacity_bytes = provider.capacity();
        for (const auto& output : graph.outputs()) pinned.insert(output.second);
        for (const auto& item : graph.tensors()) remaining[item.first] = item.second.consumers.size();
        for (const auto& item : graph.tensors()) if (item.second.external) insert(item.first, *item.second.external);
        for (const auto& item : graph.tensors())
            if (item.second.external && remaining[item.first] == 0 && !pinned.count(item.first)) release(item.first);
        for (auto id : graph.order()) {
            result.failed_node = id;
            event(TraceKind::NodeBegin);
            const auto& descriptor = graph.nodes().at(id).descriptor;
            const auto output_id = descriptor.outputs()[0];
            const auto& record = graph.tensors().at(output_id);
            Status status;
            {
                TensorInputs inputs;
                for (auto input : descriptor.inputs()) {
                    inputs.emplace_back(live.at(input));
                    event(TraceKind::Tensor, input);
                }
                const auto inferred = infer_operator(descriptor, inputs);
                if (!inferred.ok()) status = inferred.status;
                else {
                    const auto& contract = *inferred.output;
                    // End all temporary alias handles before last-use releases.
                    Tensor output = contract.kind == OutputKind::NewTensor
                        ? provider.allocate(output_id, contract.shape, contract.dtype) : *contract.alias;
                    if (contract.kind == OutputKind::NewTensor && output.nbytes() != 0) {
                        pending_allocation = output_id;
                        pending_offset = std::make_pair(output_id, output.data_offset());
                        ++result.counts.allocation_requests;
                        if (backing) ++result.counts.allocations;
                        result.counts.allocated_bytes += output.nbytes();
                        result.counts.live_bytes += output.nbytes();
                        result.counts.peak_live_bytes = std::max(result.counts.peak_live_bytes, result.counts.live_bytes);
                        event(backing ? TraceKind::Allocate : TraceKind::BlockAllocate, output_id, output.nbytes());
                        if (trace) root_offsets.emplace(output_id, output.data_offset());
                        owned_bytes.emplace(*record.base, output.nbytes());
                    }
                    insert(output_id, std::move(output));
                    pending_allocation.reset();
                    pending_offset.reset();
                    event(TraceKind::Tensor, output_id);
                    status = backend.execute(descriptor, inputs, live.at(output_id));
                    if (status.ok() && contract.kind == OutputKind::Alias) event(TraceKind::Alias, output_id);
                    if (status.ok() && contract.kind == OutputKind::Write) event(TraceKind::StateWrite, output_id);
                    if (status.ok() && (descriptor.code() == OpCode::COPY || descriptor.code() == OpCode::MATERIALIZE)) {
                        const auto& source = inputs[0].get();
                        const auto& destination = live.at(output_id);
                        if (source.numel() != 0 && !same_tensor_layout(source, destination)) {
                            ++result.counts.copies;
                            result.counts.copy_bytes += source.nbytes();
                            event(TraceKind::Copy, output_id, source.nbytes());
                        }
                    }
                }
            }
            if (!status.ok()) {
                result.status = Status::failure(status.code, "node " + std::to_string(id) + " (" + op_name(descriptor.code()) + "): " + status.message);
                event(TraceKind::Failure, output_id, 0, status.code);
                cleanup();
                return result;
            }
            ++result.counts.nodes_completed;
            event(TraceKind::NodeEnd);
            std::set<TensorId> unique_inputs(descriptor.inputs().begin(), descriptor.inputs().end());
            for (auto input : unique_inputs)
                if (--remaining.at(input) == 0 && !pinned.count(input)) release(input);
            if (remaining.at(output_id) == 0 && !pinned.count(output_id)) release(output_id);
        }
        result.failed_node.reset();
        for (const auto& output : graph.outputs()) {
            result.outputs.emplace(output.first, live.at(output.second));
            event(TraceKind::Output, output.second);
        }
        result.status = Status::success();
        return result;
    } catch (const ArenaExhausted& error) {
        result.status = Status::failure(StatusCode::ResourceExhausted, error.what());
        event(TraceKind::Failure, {}, 0, result.status.code);
        cleanup();
        return result;
    } catch (const std::bad_alloc&) {
        result.status = Status::failure(StatusCode::ResourceExhausted, "CPU graph allocation failed");
        event(TraceKind::Failure, {}, 0, result.status.code);
        cleanup();
        return result;
    }
}
} // namespace runtime
