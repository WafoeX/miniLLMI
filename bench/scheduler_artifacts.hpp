#pragma once
#include "runtime/graph_executor.hpp"
#include "runtime/memory_planner.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace scheduler_artifacts {
using namespace runtime;
// Diagnostic-only, outside timers. Full CPU output/reference and actual trace;
// uses the single Graph/Tensor runtime and never performs hidden transfers.
inline void snapshot(const std::filesystem::path& directory, const std::string& label, const Graph& graph,
                     const MemoryPlan& plan, const ExecutionResult& result, const Tensor& oracle, const ExecutionTrace& trace) {
    if (directory.empty()) return;
    const auto path = directory / (label + ".json");
    if (std::filesystem::exists(path)) throw std::invalid_argument("snapshot must be a new file");
    std::ofstream out(path); out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << "{\"schema_version\":1,\"atol\":0.001,\"rtol\":0.001,\"counts\":{\"copies\":" << result.counts.copies
        << ",\"copy_bytes\":" << result.counts.copy_bytes << ",\"backend_dispatches\":" << result.counts.backend_dispatches
        << ",\"backend_switches\":" << result.counts.backend_switches << ",\"allocations\":" << result.counts.allocations
        << "},\"workspace_bytes\":0,\"device_capacities\":[";
    bool first = true;
    for (const auto& item : plan.device_capacity_bytes) {
        if (!first) out << ',';
        first = false;
        out << "{\"device\":\"" << (item.first.type() == DeviceType::CPU ? "cpu:" : "cuda:") << item.first.index()
            << "\",\"bytes\":" << item.second << '}';
    }
    out << "],\"nodes\":["; first = true;
    for (auto node : graph.order()) {
        if (!first) out << ',';
        first = false;
        out << "{\"id\":" << node << ",\"descriptor\":" << graph.nodes().at(node).descriptor.serialize() << '}';
    }
    out << "],\"trace_dropped\":" << trace.dropped() << ",\"trace\":["; first = true;
    for (const auto& event : trace.events()) {
        if (!first) out << ',';
        first = false;
        out << "{\"kind\":\"" << trace_name(event.kind) << "\",\"bytes\":" << event.bytes << ",\"node\":";
        if (event.node) out << *event.node; else out << "null";
        out << ",\"tensor\":"; if (event.tensor) out << *event.tensor; else out << "null";
        if (event.metadata) out << ",\"device\":\"" << (event.metadata->device.type() == DeviceType::CPU ? "cpu:" : "cuda:")
            << event.metadata->device.index() << "\",\"offset_bytes\":" << event.metadata->offset_bytes << ",\"capacity_bytes\":" << event.metadata->capacity_bytes;
        out << '}';
    }
    const auto values = [&](const Tensor& tensor) {
        out << '['; for (std::size_t i = 0; i < tensor.numel(); ++i) { if (i) out << ','; out << tensor.data<float>()[i]; } out << ']';
    };
    out << "],\"output\":"; values(result.outputs.at("result"));
    out << ",\"cpu_reference\":"; values(oracle); out << "}\n";
}
} // namespace scheduler_artifacts
