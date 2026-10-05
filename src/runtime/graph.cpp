#include "runtime/graph.hpp"
#include <algorithm>
#include <stdexcept>
#include <set>
#include <utility>

namespace runtime {
void Graph::editable() const {
    if (frozen_) throw std::logic_error("graph topology is frozen");
}
void Graph::add_tensor(TensorId id, Shape shape, DType dtype, Device device) {
    editable();
    if (id == INVALID_TENSOR_ID || tensors_.count(id)) throw std::invalid_argument("duplicate/invalid tensor ID");
    (void)nbytes(shape, dtype);
    tensors_.emplace(id, TensorRecord{id, std::move(shape), dtype, device, std::nullopt, false,
                                     std::nullopt, {}, std::nullopt, std::nullopt, OutputKind::NewTensor});
}
void Graph::add_input(TensorId id, std::string name, Tensor binding, bool persistent_state) {
    editable();
    if (name.empty() || inputs_.count(name)) throw std::invalid_argument("duplicate/empty input name");
    auto tensors = tensors_;
    auto inputs = inputs_;
    if (id == INVALID_TENSOR_ID || tensors.count(id)) throw std::invalid_argument("duplicate/invalid tensor ID");
    const auto shape = binding.shape();
    const auto dtype = binding.dtype();
    const auto device = binding.device();
    tensors.emplace(id, TensorRecord{id, shape, dtype, device, std::move(binding), persistent_state,
                                    std::nullopt, {}, std::nullopt, std::nullopt, OutputKind::Alias});
    inputs.emplace(std::move(name), id);
    tensors_.swap(tensors);
    inputs_.swap(inputs);
}
void Graph::add_node(NodeId id, OpDesc descriptor) {
    editable();
    if (id == INVALID_NODE_ID || nodes_.count(id)) throw std::invalid_argument("duplicate/invalid node ID");
    const auto schema = validate_schema(descriptor.code(), descriptor.inputs(), descriptor.outputs(), descriptor.attrs());
    if (!schema.ok()) throw std::invalid_argument(schema.message);
    for (auto input : descriptor.inputs())
        if (!tensors_.count(input)) throw std::invalid_argument("node references unknown input tensor");
    for (auto output : descriptor.outputs()) {
        const auto found = tensors_.find(output);
        if (found == tensors_.end()) throw std::invalid_argument("node references unknown output tensor");
        if (found->second.external || found->second.producer) throw std::invalid_argument("tensor requires exactly one producer; inputs cannot be overwritten");
    }
    auto tensors = tensors_;
    auto nodes = nodes_;
    for (auto input : descriptor.inputs()) {
        auto& consumers = tensors.at(input).consumers;
        if (std::find(consumers.begin(), consumers.end(), id) == consumers.end()) consumers.push_back(id);
    }
    for (auto output : descriptor.outputs()) tensors.at(output).producer = id;
    nodes.emplace(id, NodeRecord{id, std::move(descriptor)});
    tensors_.swap(tensors);
    nodes_.swap(nodes);
}
void Graph::add_output(std::string name, TensorId id) {
    editable();
    if (name.empty() || outputs_.count(name)) throw std::invalid_argument("duplicate/empty output name");
    if (!tensors_.count(id)) throw std::invalid_argument("unknown graph output tensor");
    outputs_.emplace(std::move(name), id);
}
namespace {
Status graph_failure(StatusCode code, const std::string& message) {
    return Status::failure(code, "graph: " + message);
}
}
Status Graph::freeze() {
    if (frozen_) return Status::success();
    for (const auto& item : tensors_) {
        if (!item.second.external && !item.second.producer)
            return graph_failure(StatusCode::InvalidArgument, "tensor " + std::to_string(item.first) + " has no producer or named input");
    }
    std::map<NodeId, std::set<NodeId>> dependencies, followers;
    for (const auto& item : nodes_) {
        for (auto input : item.second.descriptor.inputs()) {
            const auto producer = tensors_.at(input).producer;
            if (producer) dependencies[item.first].insert(*producer);
        }
        for (auto producer : dependencies[item.first]) followers[producer].insert(item.first);
    }
    std::set<NodeId> ready;
    std::map<NodeId, std::size_t> pending;
    for (const auto& item : nodes_) {
        pending[item.first] = dependencies[item.first].size();
        if (pending[item.first] == 0) ready.insert(item.first);
    }
    std::vector<NodeId> order;
    while (!ready.empty()) {
        const auto id = *ready.begin();
        ready.erase(ready.begin());
        order.push_back(id);
        for (auto next : followers[id]) if (--pending[next] == 0) ready.insert(next);
    }
    if (order.size() != nodes_.size()) {
        std::string remaining;
        for (const auto& item : pending) if (item.second) remaining += " " + std::to_string(item.first);
        return graph_failure(StatusCode::InvalidArgument, "cycle among nodes:" + remaining);
    }
    auto records = tensors_;
    std::map<const Storage*, TensorId> external_roots;
    std::map<NodeId, std::set<NodeId>> ancestors;
    std::map<TensorId, std::set<NodeId>> versions;
    std::vector<NodeId> writes;
    try {
        for (auto& item : records) {
            auto& record = item.second;
            if (!record.external) continue;
            const auto& tensor = *record.external;
            // Different wrappers over intersecting physical allocations cannot
            // be modelled as independent roots. Reject rather than hide hazards.
            for (const auto& root : external_roots) {
                const auto& other = *records.at(root.second).external;
                if (tensor.storage() == other.storage() || tensor.device() != other.device()) continue;
                const auto begin = reinterpret_cast<std::uintptr_t>(tensor.storage()->data());
                const auto other_begin = reinterpret_cast<std::uintptr_t>(other.storage()->data());
                const auto end = checked_add(begin, tensor.storage()->capacity_bytes());
                const auto other_end = checked_add(other_begin, other.storage()->capacity_bytes());
                if (begin < other_end && other_begin < end)
                    return graph_failure(StatusCode::Aliasing, "overlapping external Storage wrappers require one shared Storage");
            }
            const auto inserted = external_roots.emplace(tensor.storage().get(), record.id);
            const auto base = inserted.first->second;
            if (records.at(base).state_binding != record.state_binding)
                return graph_failure(StatusCode::Aliasing, "shared external Storage requires consistent persistent-state declarations");
            record.base = base;
            record.layout = Layout{tensor.shape(), tensor.stride(), tensor.dtype(), tensor.device(),
                                   tensor.storage()->capacity_bytes(), tensor.data_offset(), base};
        }
        for (auto id : order) {
            const auto& node = nodes_.at(id);
            for (auto parent : dependencies[id]) {
                ancestors[id].insert(parent);
                ancestors[id].insert(ancestors[parent].begin(), ancestors[parent].end());
            }
            LayoutInputs inputs;
            for (auto input : node.descriptor.inputs()) inputs.emplace_back(*records.at(input).layout);
            const auto inferred = infer_layout(node.descriptor, inputs);
            if (!inferred.ok())
                return graph_failure(inferred.status.code, "node " + std::to_string(id) + " (" + op_name(node.descriptor.code()) + "): " + inferred.status.message);
            auto& output = records.at(node.descriptor.outputs()[0]);
            const auto& contract = *inferred.output;
            if (output.shape != contract.layout.shape)
                return graph_failure(StatusCode::ShapeMismatch, "node " + std::to_string(id) + " output " + std::to_string(output.id) + " shape differs from inference");
            if (output.dtype != contract.layout.dtype)
                return graph_failure(StatusCode::DTypeMismatch, "node " + std::to_string(id) + " output dtype differs from inference");
            if (output.device != contract.layout.device)
                return graph_failure(StatusCode::DeviceMismatch, "node " + std::to_string(id) + " output device differs from inference");
            output.kind = contract.kind;
            output.layout = contract.layout;
            if (contract.kind == OutputKind::NewTensor) {
                output.base = output.id;
                output.layout->storage_key = output.id;
            } else {
                const auto source = node.descriptor.inputs()[contract.kind == OutputKind::Write ? 1 : 0];
                output.base = records.at(source).base;
                versions[output.id] = versions[source];
                if (contract.kind == OutputKind::Write) {
                    const auto& base = records.at(*output.base);
                    if (!base.external || !base.state_binding)
                        return graph_failure(StatusCode::Aliasing, "node " + std::to_string(id) + " COPY destination is not an explicit persistent state binding");
                    versions[output.id].insert(id);
                    writes.push_back(id);
                }
            }
        }
        // Node-ID order is NOT a mutation dependency. Check physical spans and
        // logical versions, including aliases and multiple names for one Storage.
        for (auto writer : writes) {
            const auto& write = nodes_.at(writer).descriptor;
            const auto& target = *records.at(write.inputs()[1]).layout;
            for (const auto& item : nodes_) {
                const auto reader = item.first;
                if (reader == writer) continue;
                const auto& desc = item.second.descriptor;
                const auto kind = records.at(desc.outputs()[0]).kind;
                if (kind == OutputKind::Alias) continue; // no value read; consumers are checked below
                for (std::size_t i = 0; i < desc.inputs().size(); ++i) {
                    const auto input = desc.inputs()[i];
                    if (!spans_overlap(target, *records.at(input).layout)) continue;
                    if (ancestors[writer].count(reader)) continue; // reader completes before mutation
                    if (!ancestors[reader].count(writer))
                        return graph_failure(StatusCode::Aliasing, "unordered overlapping state access: writer " + std::to_string(writer) + ", node " + std::to_string(reader));
                    if (!versions[input].count(writer))
                        return graph_failure(StatusCode::Aliasing, "stale state version at node " + std::to_string(reader) + ", tensor " + std::to_string(input) + "; depend on write " + std::to_string(writer));
                }
            }
            for (const auto& output : outputs_) {
                if (spans_overlap(target, *records.at(output.second).layout) && !versions[output.second].count(writer))
                    return graph_failure(StatusCode::Aliasing, "graph output '" + output.first + "' pins a stale state version");
            }
        }
    } catch (const std::overflow_error& error) {
        return graph_failure(StatusCode::Overflow, error.what());
    } catch (const std::out_of_range& error) {
        return graph_failure(StatusCode::OutOfRange, error.what());
    } catch (const std::invalid_argument& error) {
        return graph_failure(StatusCode::InvalidArgument, error.what());
    }
    tensors_.swap(records);
    order_.swap(order);
    frozen_ = true;
    return Status::success();
}
} // namespace runtime
