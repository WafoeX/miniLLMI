#include "runtime/graph.hpp"
#include <algorithm>
#include <stdexcept>
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
} // namespace runtime
