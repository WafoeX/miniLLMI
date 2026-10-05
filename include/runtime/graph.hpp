#pragma once

#include "runtime/layout.hpp"
#include "runtime/shape_inference.hpp"
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace runtime {
using NodeId = std::uint32_t;
inline constexpr NodeId INVALID_NODE_ID = UINT32_MAX;
struct TensorRecord {
    TensorId id;
    Shape shape;
    DType dtype;
    Device device;
    std::optional<Tensor> external; // only explicit named roots own persistent buffers
    bool state_binding = false;
    std::optional<NodeId> producer;
    std::vector<NodeId> consumers; // unique, ordered by node insertion
    std::optional<Layout> layout; // resolved transactionally by freeze
    std::optional<TensorId> base; // physical allocation root, NOT a producer edge
    OutputKind kind = OutputKind::NewTensor;
};
struct NodeRecord { NodeId id; OpDesc descriptor; };

class Graph {
public:
    void add_tensor(TensorId id, Shape shape, DType dtype = DType::FP32, Device device = Device{});
    void add_input(TensorId id, std::string name, Tensor binding, bool persistent_state = false);
    void add_node(NodeId id, OpDesc descriptor);
    void add_output(std::string name, TensorId id);
    Status freeze(); // failure leaves an editable, unvalidated graph
    bool frozen() const noexcept { return frozen_; }
    const std::map<TensorId, TensorRecord>& tensors() const noexcept { return tensors_; }
    const std::map<NodeId, NodeRecord>& nodes() const noexcept { return nodes_; }
    const std::map<std::string, TensorId>& inputs() const noexcept { return inputs_; }
    const std::map<std::string, TensorId>& outputs() const noexcept { return outputs_; }
    const std::vector<NodeId>& order() const noexcept { return order_; }
private:
    void editable() const;
    std::map<TensorId, TensorRecord> tensors_;
    std::map<NodeId, NodeRecord> nodes_;
    std::map<std::string, TensorId> inputs_, outputs_;
    std::vector<NodeId> order_;
    bool frozen_ = false;
};
} // namespace runtime
