#include "runtime/graph_executor.hpp"
#include "runtime/copy.hpp"
#include "runtime/reference.hpp"
#include <algorithm>
#include <new>
#include <set>

namespace runtime {
ExecutionResult execute_graph(const Graph& graph) {
    ExecutionResult result;
    if (!graph.frozen()) {
        result.status = Status::failure(StatusCode::InvalidArgument, "executor requires a successfully frozen graph");
        return result;
    }
    for (const auto& item : graph.tensors()) {
        if (item.second.device.type() != DeviceType::CPU) {
            result.status = Status::failure(StatusCode::DeviceMismatch, "sequential CPU executor requires explicit CPU tensors; no hidden transfers");
            return result;
        }
    }
    std::map<TensorId, Tensor> live;
    std::map<TensorId, std::size_t> remaining, root_handles, owned_bytes;
    std::set<TensorId> pinned;
    for (const auto& output : graph.outputs()) pinned.insert(output.second);
    for (const auto& item : graph.tensors()) remaining[item.first] = item.second.consumers.size();
    const auto release = [&](TensorId id) {
        const auto found = live.find(id);
        if (found == live.end()) return;
        const auto base = *graph.tensors().at(id).base;
        live.erase(found);
        if (--root_handles.at(base) == 0 && owned_bytes.count(base)) {
            ++result.counts.frees;
            result.counts.live_bytes -= owned_bytes.at(base);
            owned_bytes.erase(base);
        }
    };
    const auto cleanup = [&] {
        result.outputs.clear();
        while (!live.empty()) release(live.begin()->first);
        // A metadata-container allocation can fail after a buffer was created
        // but before it acquired a live TensorId. Stack unwinding frees it too.
        result.counts.frees = result.counts.allocations;
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
        for (const auto& item : graph.tensors()) if (item.second.external) insert(item.first, *item.second.external);
        // Unused external roots need not keep a runtime handle. Graph still owns
        // their persistent bindings, which are excluded from dynamic counts.
        for (const auto& item : graph.tensors())
            if (item.second.external && remaining[item.first] == 0 && !pinned.count(item.first)) release(item.first);
        for (auto id : graph.order()) {
            result.failed_node = id;
            const auto& descriptor = graph.nodes().at(id).descriptor;
            const auto output_id = descriptor.outputs()[0];
            const auto& record = graph.tensors().at(output_id);
            Status status;
            {
                TensorInputs inputs;
                for (auto input : descriptor.inputs()) inputs.emplace_back(live.at(input));
                const auto inferred = infer_operator(descriptor, inputs);
                if (!inferred.ok()) { result.status = inferred.status; cleanup(); return result; }
                const auto& contract = *inferred.output;
                // Scope ends before releases: inference alias handles must not
                // accidentally extend an intermediate beyond its actual last use.
                Tensor output = contract.kind == OutputKind::NewTensor
                    ? Tensor::allocate_cpu(contract.shape, contract.dtype) : *contract.alias;
                if (contract.kind == OutputKind::NewTensor && output.nbytes() != 0) {
                    ++result.counts.allocations;
                    result.counts.allocated_bytes += output.nbytes();
                    result.counts.live_bytes += output.nbytes();
                    result.counts.peak_live_bytes = std::max(result.counts.peak_live_bytes, result.counts.live_bytes);
                    owned_bytes.emplace(*record.base, output.nbytes());
                }
                insert(output_id, std::move(output));
                if (contract.kind == OutputKind::Alias) status = Status::success();
                else status = reference::execute(descriptor, inputs, live.at(output_id));
                if (status.ok() && (descriptor.code() == OpCode::COPY || descriptor.code() == OpCode::MATERIALIZE)) {
                    const auto& source = inputs[0].get();
                    const auto& destination = live.at(output_id);
                    if (source.numel() != 0 && !same_tensor_layout(source, destination)) {
                        ++result.counts.copies;
                        result.counts.copy_bytes += source.nbytes();
                    }
                }
            }
            if (!status.ok()) {
                result.status = Status::failure(status.code, "node " + std::to_string(id) + " (" + op_name(descriptor.code()) + "): " + status.message);
                cleanup();
                return result;
            }
            ++result.counts.nodes_completed;
            std::set<TensorId> unique_inputs(descriptor.inputs().begin(), descriptor.inputs().end());
            for (auto input : unique_inputs)
                if (--remaining.at(input) == 0 && !pinned.count(input)) release(input);
            if (remaining.at(output_id) == 0 && !pinned.count(output_id)) release(output_id);
        }
        for (const auto& output : graph.outputs()) result.outputs.emplace(output.first, live.at(output.second));
        result.failed_node.reset();
        result.status = Status::success();
        return result;
    } catch (const std::bad_alloc&) {
        result.status = Status::failure(StatusCode::ResourceExhausted, "CPU graph allocation failed");
        cleanup();
        return result;
    }
}
} // namespace runtime
