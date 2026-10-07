#include "runtime/scheduler.hpp"
#include "runtime/copy.hpp"
#include <memory>
#include <stdexcept>

namespace runtime {
Scheduler::Scheduler(const Backend& cpu, const Backend* cuda, PlacementFallback fallback)
    : cpu_(cpu), cuda_(cuda), fallback_(fallback) {
    if (cpu.device() != Device{} || (cuda && cuda->device().type() != DeviceType::CUDA))
        throw std::invalid_argument("scheduler requires CPU:0 and optionally one CUDA backend");
    if (fallback != PlacementFallback::CPU && fallback != PlacementFallback::Error)
        throw std::invalid_argument("unknown placement fallback");
}
Status Scheduler::supports(const Backend& backend, OpCode code, DType dtype, const LayoutInputs& inputs) const {
    const auto capability = backend.capability(code, backend.device(), dtype);
    if (!capability.ok()) return capability;
    for (const auto& item : inputs) {
        const auto& layout = item.get();
        try { layout.validate(); }
        catch (const std::exception& e) { return Status::failure(StatusCode::LayoutMismatch, e.what()); }
        if (backend.device().type() == DeviceType::CUDA &&
            (layout.dtype != DType::FP32 || !layout.is_contiguous()))
            return Status::failure(StatusCode::LayoutMismatch, "CUDA requires contiguous FP32 bindings; MATERIALIZE must be explicit");
    }
    return Status::success();
}
PlacementDecision Scheduler::place(OpCode code, DType dtype, const LayoutInputs& inputs, Device requested) const {
    const Backend* preferred = requested == cpu_.device() ? &cpu_
        : cuda_ && requested == cuda_->device() ? cuda_ : nullptr;
    const auto status = preferred ? supports(*preferred, code, dtype, inputs)
        : Status::failure(StatusCode::Unsupported, "requested device has no registered backend");
    if (status.ok()) return {status, requested, false, "requested placement supported"};
    if (fallback_ == PlacementFallback::CPU && requested != cpu_.device()) {
        const auto fallback = supports(cpu_, code, dtype, inputs);
        if (fallback.ok()) return {fallback, cpu_.device(), true, status.message + "; explicit CPU placement"};
        return {fallback, cpu_.device(), true, status.message + "; CPU also unsupported: " + fallback.message};
    }
    return {status, requested, false, status.message};
}
bool Scheduler::has_device(Device d) const noexcept {
    return d == cpu_.device() || (cuda_ && d == cuda_->device());
}
const Backend& Scheduler::backend(Device d) const {
    if (d == cpu_.device()) return cpu_;
    if (cuda_ && d == cuda_->device()) return *cuda_;
    throw std::invalid_argument("scheduler device has no registered backend");
}
ScheduledGraph Scheduler::rewrite(const Graph& logical) const {
    ScheduledGraph result;
    if (!logical.frozen()) {
        result.status = Status::failure(StatusCode::InvalidArgument, "scheduler requires frozen logical graph");
        return result;
    }
    try {
        Graph graph;
        std::map<TensorId, Layout> layouts;
        for (const auto& name : logical.inputs()) {
            const auto& r = logical.tensors().at(name.second);
            if (!has_device(r.device)) throw std::invalid_argument("external binding has no registered backend");
            graph.add_input(r.id, name.first, *r.external, r.state_binding);
            layouts.emplace(r.id, *r.layout);
        }
        // Copies are cached by exact logical tensor version AND destination.
        // Never cache by physical root: different COPY-write outputs are versions.
        std::map<std::pair<TensorId, Device>, TensorId> copies;
        std::uint64_t next_tensor = logical.tensors().empty() ? 0 : static_cast<std::uint64_t>(logical.tensors().rbegin()->first) + 1;
        std::uint64_t next_node = 0;
        const auto tensor_id = [&]() {
            if (next_tensor >= INVALID_TENSOR_ID) throw std::overflow_error("scheduler tensor ID exhaustion");
            return static_cast<TensorId>(next_tensor++);
        };
        const auto node_id = [&]() {
            if (next_node >= INVALID_NODE_ID) throw std::overflow_error("scheduler node ID exhaustion");
            return static_cast<NodeId>(next_node++);
        };
        const auto transfer = [&](TensorId source, Device destination) {
            const auto& layout = layouts.at(source);
            if (layout.device == destination) return source;
            const auto key = std::make_pair(source, destination);
            const auto cached = copies.find(key);
            if (cached != copies.end()) return cached->second;
            const auto gpu = layout.device.type() == DeviceType::CUDA ? layout.device : destination;
            if (!has_device(destination) || !has_device(layout.device)) throw std::invalid_argument("COPY requires registered source/destination");
            const auto supported = supports(backend(gpu), OpCode::COPY, layout.dtype, {layout});
            if (!supported.ok()) throw std::invalid_argument(supported.message);
            const auto output = tensor_id(); const auto node = node_id();
            OpDesc copy(OpCode::COPY, {source}, {output}, CopyAttrs{CopyOverlap::RejectExceptExactSelf, destination});
            const auto inferred = infer_layout(copy, {layout});
            if (!inferred.ok()) throw std::invalid_argument(inferred.status.message);
            auto out = inferred.output->layout; out.storage_key = output;
            graph.add_tensor(output, out.shape, out.dtype, destination);
            graph.add_node(node, std::move(copy)); layouts.emplace(output, out); copies.emplace(key, output);
            result.inserted_copies.push_back({node, source, output, layout.device, destination, nbytes(layout.shape, layout.dtype)});
            return output;
        };
        for (const auto original : logical.order()) {
            const auto& desc = logical.nodes().at(original).descriptor;
            const auto output = desc.outputs()[0]; const auto& record = logical.tensors().at(output);
            LayoutInputs inputs;
            for (const auto input : desc.inputs()) inputs.emplace_back(layouts.at(input));
            Device destination;
            PlacementDecision decision;
            if (record.kind == OutputKind::Alias) {
                // Views are metadata, not an absent CUDA kernel. A view cannot
                // move its underlying Storage or persistent-state destination.
                destination = inputs[0].get().device;
                if (desc.backend_hint() && *desc.backend_hint() != destination)
                    throw std::invalid_argument("alias placement hint cannot move Storage; insert COPY explicitly");
                decision = {Status::success(), destination, false, "metadata alias stays with its Storage"};
            } else if (desc.code() == OpCode::COPY) {
                destination = record.kind == OutputKind::Write ? inputs[1].get().device : *std::get<CopyAttrs>(desc.attrs()).destination;
                if (desc.backend_hint() && *desc.backend_hint() != destination)
                    throw std::invalid_argument("COPY destination cannot be changed by a placement hint");
                if (!has_device(destination)) throw std::invalid_argument("explicit COPY destination unavailable");
                const auto executor = inputs[0].get().device.type() == DeviceType::CUDA ? inputs[0].get().device : destination;
                const auto supported = supports(backend(executor), OpCode::COPY, record.dtype, inputs);
                if (!supported.ok()) throw std::invalid_argument(supported.message);
                decision = {Status::success(), destination, false, "explicit COPY destination preserved"};
            } else {
                decision = place(desc.code(), record.dtype, inputs, desc.backend_hint().value_or(record.device));
                if (!decision.ok()) { result.status = decision.status; return result; }
                destination = decision.device;
            }
            std::vector<TensorId> ids = desc.inputs();
            if (record.kind != OutputKind::Alias && desc.code() != OpCode::COPY)
                for (auto& input : ids) input = transfer(input, destination);
            LayoutInputs moved;
            for (auto input : ids) moved.emplace_back(layouts.at(input));
            OpDesc rewritten(desc.code(), ids, {output}, desc.attrs(), destination);
            const auto inferred = infer_layout(rewritten, moved);
            if (!inferred.ok()) throw std::invalid_argument(inferred.status.message);
            auto out = inferred.output->layout;
            if (inferred.output->kind == OutputKind::NewTensor) out.storage_key = output;
            if (out.device != destination) throw std::invalid_argument("inferred placement differs from policy");
            graph.add_tensor(output, out.shape, out.dtype, out.device);
            const auto node = node_id(); graph.add_node(node, std::move(rewritten));
            layouts.emplace(output, std::move(out));
            result.placements.emplace(original, std::move(decision)); result.rewritten_nodes.emplace(original, node);
        }
        for (const auto& name : logical.outputs())
            graph.add_output(name.first, transfer(name.second, logical.tensors().at(name.second).device));
        result.status = graph.freeze(); // topology, aliases and state versions revalidated
        if (result.status.ok()) result.graph.emplace(std::move(graph));
    } catch (const std::overflow_error& e) { result.status = Status::failure(StatusCode::Overflow, e.what()); }
      catch (const std::exception& e) { result.status = Status::failure(StatusCode::InvalidArgument, e.what()); }
    return result;
}
BackendPreparation Scheduler::prepare_node(const OpDesc& desc, const TensorInputs& inputs, const Tensor& output) const {
    const auto inferred = infer_operator(desc, inputs);
    if (!inferred.ok()) return {inferred.status, 0};
    const auto binding = validate_output_binding(*inferred.output, output);
    if (!binding.ok()) return {binding, 0};
    if (!has_device(output.device())) return {Status::failure(StatusCode::DeviceMismatch, "unregistered output device"), 0};
    for (const auto& input : inputs) if (!has_device(input.get().device()))
        return {Status::failure(StatusCode::DeviceMismatch, "unregistered input device"), 0};
    if (desc.backend_hint() && *desc.backend_hint() != output.device())
        return {Status::failure(StatusCode::DeviceMismatch, "scheduled node hint differs from physical placement"), 0};
    if (inferred.output->kind == OutputKind::Alias) return {Status::success(), 0};
    if (desc.code() == OpCode::COPY) {
        const auto& source = inputs[0].get();
        if (source.device().type() == DeviceType::CUDA || output.device().type() == DeviceType::CUDA) {
            if (!cuda_ || source.dtype() != DType::FP32 || !source.is_contiguous() || !output.is_contiguous())
                return {Status::failure(StatusCode::Unsupported, "CUDA COPY requires contiguous FP32 bindings"), 0};
        }
        const auto executor = source.device().type() == DeviceType::CUDA ? source.device() : output.device();
        const auto supported = backend(executor).capability(OpCode::COPY, executor, output.dtype());
        if (!supported.ok()) return {supported, 0};
        if (!same_tensor_layout(source, output) && memory_spans_overlap(source, output))
            return {Status::failure(StatusCode::Aliasing, "COPY spans overlap"), 0};
        return {Status::success(), 0};
    }
    return backend(output.device()).prepare(desc, inputs, output);
}
Status Scheduler::execute_node(const OpDesc& desc, const TensorInputs& inputs, Tensor& output, std::optional<Device>& executed) const {
    executed.reset();
    const auto prepared = prepare_node(desc, inputs, output);
    if (!prepared.ok()) return prepared.status;
    if (prepared.workspace_bytes) return Status::failure(StatusCode::Unsupported, "scheduler current backends require zero workspace");
    const auto inferred = infer_operator(desc, inputs);
    if (inferred.output->kind == OutputKind::Alias) return Status::success();
    if (desc.code() == OpCode::COPY) {
        const auto& source = inputs[0].get();
        if (source.numel() == 0 || same_tensor_layout(source, output)) return Status::success();
        const auto d = source.device().type() == DeviceType::CUDA ? source.device() : output.device();
        executed = d;
        return backend(d).copy(source, output); // synchronous completion before any consumer/write
    }
    executed = output.device();
    return backend(output.device()).execute(desc, inputs, output);
}
ScheduledAllocationProvider::ScheduledAllocationProvider(const Graph& graph, const Scheduler& scheduler, PlanPolicy policy)
    : plan_(plan_memory(graph, policy)) {
    const auto valid = validate_memory_plan(graph, plan_);
    if (!valid.ok()) throw std::invalid_argument(valid.message);
    for (const auto& item : plan_.device_capacity_bytes) {
        if (item.second % sizeof(float) || item.second / sizeof(float) > static_cast<std::size_t>(INT64_MAX))
            throw std::overflow_error("scheduled backing size overflow");
        auto buffer = scheduler.backend(item.first).allocate({static_cast<std::int64_t>(item.second / sizeof(float))}, DType::FP32, item.first);
        if (!buffer.ok()) throw std::runtime_error(buffer.status.message);
        storage_.emplace(item.first, buffer.tensor->storage());
    }
    std::map<TensorId, Tensor> tensors;
    for (const auto& item : graph.tensors()) {
        const auto& r = item.second; const auto root = *r.base;
        if (r.device.type() == DeviceType::CUDA && r.dtype != DType::FP32)
            throw std::invalid_argument("CUDA scheduled storage supports FP32 only");
        const auto& base = graph.tensors().at(root);
        auto storage = base.external ? base.external->storage() : storage_.at(r.device);
        const auto offset = r.layout->offset_bytes + (base.external ? 0 : plan_.slots.at(root).offset);
        tensors.emplace(r.id, Tensor(std::move(storage), r.dtype, r.shape, r.layout->stride, offset));
        if (r.kind == OutputKind::NewTensor) bindings_.emplace(r.id, Binding{r.shape, r.dtype, r.device, false});
    }
    // All backend requirements checked before execution and external mutations.
    // Current S6/S8 backends publish zero scratch. Reject future nonzero scratch
    // rather than accidentally omitting it from the device memory plan.
    for (auto node : graph.order()) {
        const auto& desc = graph.nodes().at(node).descriptor; TensorInputs inputs;
        for (auto input : desc.inputs()) inputs.emplace_back(tensors.at(input));
        const auto prepared = scheduler.prepare_node(desc, inputs, tensors.at(desc.outputs()[0]));
        if (!prepared.ok()) throw std::invalid_argument(prepared.status.message);
        if (prepared.workspace_bytes) throw std::invalid_argument("nonzero scheduler workspace not implemented");
    }
}
Status ScheduledAllocationProvider::validate_graph(const Graph& graph) const { return validate_memory_plan(graph, plan_); }
Status ScheduledAllocationProvider::begin() {
    if (running_) return Status::failure(StatusCode::InvalidArgument, "scheduled context already running");
    for (const auto& item : storage_) if (item.second.use_count() != 1)
        return Status::failure(StatusCode::InvalidArgument, "scheduled context busy: release returned Tensor/Storage aliases");
    for (auto& item : bindings_) item.second.active = false;
    running_ = true; return Status::success();
}
Tensor ScheduledAllocationProvider::allocate(TensorId root, Shape shape, DType dtype) {
    const auto found = bindings_.find(root);
    if (!running_ || found == bindings_.end() || found->second.active || found->second.shape != shape || found->second.dtype != dtype)
        throw std::invalid_argument("invalid scheduled binding; explicitly prepare again");
    const auto& slot = plan_.slots.at(root); auto storage = storage_.at(slot.device);
    if (slot.device.type() == DeviceType::CPU && slot.bytes) {
        auto* pointer = static_cast<unsigned char*>(storage->data()) + slot.offset;
        if (dtype == DType::FP32) std::uninitialized_value_construct_n(reinterpret_cast<float*>(pointer), numel(shape));
        else std::uninitialized_value_construct_n(reinterpret_cast<std::int32_t*>(pointer), numel(shape));
    }
    found->second.active = true;
    return Tensor(std::move(storage), dtype, shape, contiguous_stride(shape), slot.offset);
}
void ScheduledAllocationProvider::release(TensorId root) noexcept {
    const auto found = bindings_.find(root); if (found != bindings_.end()) found->second.active = false;
}
std::size_t ScheduledAllocationProvider::capacity_for(TensorId root) const noexcept {
    const auto slot = plan_.slots.find(root);
    if (slot == plan_.slots.end()) return 0;
    const auto capacity = plan_.device_capacity_bytes.find(slot->second.device);
    return capacity == plan_.device_capacity_bytes.end() ? 0 : capacity->second;
}
} // namespace runtime
