#include "runtime/memory_planner.hpp"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace runtime {
LifetimeAnalysis analyze_lifetimes(const Graph& graph) {
    if (!graph.frozen()) throw std::invalid_argument("lifetime analysis requires frozen graph");
    if (graph.order().size() > static_cast<std::size_t>(INT64_MAX)) throw std::overflow_error("graph interval overflow");
    LifetimeAnalysis analysis;
    analysis.node_count = graph.order().size();
    std::map<NodeId, std::int64_t> positions;
    for (std::size_t i = 0; i < graph.order().size(); ++i) positions.emplace(graph.order()[i], static_cast<std::int64_t>(i));
    std::set<TensorId> outputs;
    for (const auto& output : graph.outputs()) outputs.insert(output.second);
    for (const auto& item : graph.tensors()) {
        const auto& record = item.second;
        const auto birth = record.producer ? positions.at(*record.producer) : -1;
        TensorLifetime life{item.first, *record.base, birth, birth, {}, record.consumers.size(),
                            bool(record.external), bool(outputs.count(item.first)), *record.base != item.first};
        for (const auto consumer : record.consumers) {
            const auto use = positions.at(consumer);
            if (!life.first_use || static_cast<std::size_t>(use) < *life.first_use) life.first_use = static_cast<std::size_t>(use);
            life.last_use = std::max(life.last_use, use);
        }
        if (life.output) life.last_use = static_cast<std::int64_t>(analysis.node_count);
        analysis.tensors.emplace(item.first, life);
        if (*record.base == item.first)
            analysis.roots.emplace(item.first, RootLifetime{item.first, birth, life.last_use,
                record.layout->capacity_bytes, record.device, bool(record.external)});
    }
    for (const auto& item : analysis.tensors) {
        auto& root = analysis.roots.at(item.second.root);
        root.last_use = std::max(root.last_use, item.second.last_use);
    }
    return analysis;
}
} // namespace runtime
