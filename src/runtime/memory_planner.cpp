#include "runtime/memory_planner.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <set>
#include <sstream>
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
namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (b > SIZE_MAX - a) throw std::overflow_error("plan capacity overflow");
    return a + b;
}
std::size_t align_up(std::size_t value, std::size_t alignment) {
    const auto remainder = value % alignment;
    return remainder ? add(value, alignment - remainder) : value;
}
void alignment_check(std::size_t alignment) {
    if (alignment < alignof(std::max_align_t) || (alignment & (alignment - 1)))
        throw std::invalid_argument("plan alignment must be a power of two >= max_align_t");
}
std::string signature(const Graph& graph) {
    if (!graph.frozen()) throw std::invalid_argument("plan requires frozen graph");
    std::ostringstream text;
    text << OP_SEMANTICS_VERSION << ';';
    for (const auto& item : graph.tensors()) {
        const auto& r = item.second;
        const auto& l = *r.layout;
        text << item.first << ':' << static_cast<int>(r.dtype) << ':' << static_cast<int>(r.device.type()) << ':'
             << r.device.index() << ':' << *r.base << ':' << static_cast<int>(r.kind) << ':'
             << bool(r.external) << ':' << r.state_binding << ':' << l.capacity_bytes << ':' << l.offset_bytes << ';';
        text << l.shape.rank() << ':';
        for (auto dimension : l.shape.values()) text << dimension << ',';
        text << '/'; for (auto stride : l.stride.values()) text << stride << ',';
        text << ';';
    }
    for (const auto id : graph.order()) {
        const auto json = graph.nodes().at(id).descriptor.serialize();
        text << id << ':' << json.size() << ':' << json << ';';
    }
    for (const auto* names : {&graph.inputs(), &graph.outputs()}) {
        text << names->size() << ';';
        for (const auto& name : *names) text << name.first.size() << ':' << name.first << ':' << name.second << ';';
    }
    return text.str();
}
bool overlap(const PlannedSlot& a, const PlannedSlot& b) {
    return a.bytes && b.bytes && a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
}
bool simultaneous(const PlannedSlot& a, const PlannedSlot& b) {
    return a.birth <= b.last_use && b.birth <= a.last_use;
}
} // namespace
MemoryPlan plan_memory(const Graph& graph, PlanPolicy policy, std::size_t alignment) {
    alignment_check(alignment);
    if (policy != PlanPolicy::Reuse && policy != PlanPolicy::NoReuse) throw std::invalid_argument("unknown plan policy");
    const auto lifetimes = analyze_lifetimes(graph);
    MemoryPlan plan;
    plan.alignment = alignment; plan.policy = policy; plan.graph_signature = signature(graph);
    std::vector<RootLifetime> roots;
    for (const auto& item : lifetimes.roots) {
        const auto& root = item.second;
        if (root.device.type() != DeviceType::CPU) throw std::invalid_argument("CPU-only planner: CUDA planning not implemented");
        if (!root.external) roots.push_back(root);
    }
    std::sort(roots.begin(), roots.end(), [](const auto& a, const auto& b) {
        return a.birth < b.birth || (a.birth == b.birth && a.root < b.root);
    });
    for (const auto& root : roots) {
        std::vector<PlannedSlot> active;
        std::size_t live = root.bytes;
        for (const auto& item : plan.slots) {
            if (item.second.last_use >= root.birth) live = add(live, item.second.bytes);
            if (item.second.bytes && (policy == PlanPolicy::NoReuse || item.second.last_use >= root.birth)) active.push_back(item.second);
        }
        plan.peak_live_bytes = std::max(plan.peak_live_bytes, live);
        std::sort(active.begin(), active.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
        std::size_t offset = 0;
        if (root.bytes) for (const auto& slot : active) {
            if (add(offset, root.bytes) <= slot.offset) break;
            offset = align_up(add(slot.offset, slot.bytes), alignment);
        }
        PlannedSlot slot{root.root, offset, root.bytes, root.birth, root.last_use, root.device};
        for (const auto& previous : plan.slots) if (overlap(slot, previous.second)) { ++plan.reuse_count; break; }
        plan.capacity_bytes = std::max(plan.capacity_bytes, root.bytes ? align_up(add(offset, root.bytes), alignment) : std::size_t{0});
        plan.slots.emplace(root.root, slot);
    }
    return plan;
}
Status validate_memory_plan(const Graph& graph, const MemoryPlan& plan) {
    try {
        alignment_check(plan.alignment);
        if (plan.policy != PlanPolicy::Reuse && plan.policy != PlanPolicy::NoReuse) throw std::invalid_argument("unknown plan policy");
        if (signature(graph) != plan.graph_signature) throw std::invalid_argument("stale plan: graph topology/layout/bindings changed; explicitly prepare again");
        if (plan.capacity_bytes % plan.alignment) throw std::invalid_argument("unaligned plan capacity");
        const auto life = analyze_lifetimes(graph);
        std::size_t expected = 0;
        for (const auto& item : life.roots) {
            if (item.second.device.type() != DeviceType::CPU) throw std::invalid_argument("CPU-only plan requires CPU roots");
            if (item.second.external) continue;
            ++expected;
            const auto found = plan.slots.find(item.first);
            if (found == plan.slots.end()) throw std::invalid_argument("missing plan root");
            const auto& a = found->second;
            const auto& r = item.second;
            if (a.root != item.first || a.bytes != r.bytes || a.birth != r.birth || a.last_use != r.last_use ||
                a.device != r.device || a.device.type() != DeviceType::CPU || a.offset % plan.alignment ||
                a.offset > plan.capacity_bytes || a.bytes > plan.capacity_bytes - a.offset)
                throw std::invalid_argument("invalid planned span/interval/device");
        }
        if (expected != plan.slots.size()) throw std::invalid_argument("extra planned roots");
        for (auto a = plan.slots.begin(); a != plan.slots.end(); ++a) for (auto b = std::next(a); b != plan.slots.end(); ++b)
            if (overlap(a->second, b->second) && (plan.policy == PlanPolicy::NoReuse || simultaneous(a->second, b->second)))
                throw std::invalid_argument("overlapping live planned spans");
        // Accounting is derived independently at every birth; includes outputs.
        std::size_t peak = 0, reused = 0, capacity = 0;
        for (const auto& a : plan.slots) {
            std::size_t live = 0;
            bool reuse = false;
            for (const auto& b : plan.slots) {
                if (b.second.birth <= a.second.birth && a.second.birth <= b.second.last_use) live = add(live, b.second.bytes);
                if (b.second.birth < a.second.birth && overlap(a.second, b.second)) reuse = true;
            }
            peak = std::max(peak, live); reused += reuse;
            if (a.second.bytes) capacity = std::max(capacity, align_up(add(a.second.offset, a.second.bytes), plan.alignment));
        }
        if (peak != plan.peak_live_bytes || reused != plan.reuse_count || capacity != plan.capacity_bytes)
            throw std::invalid_argument("invalid plan accounting");
        return Status::success();
    } catch (const std::invalid_argument& e) { return Status::failure(StatusCode::InvalidArgument, e.what()); }
      catch (const std::overflow_error& e) { return Status::failure(StatusCode::Overflow, e.what()); }
}
} // namespace runtime
