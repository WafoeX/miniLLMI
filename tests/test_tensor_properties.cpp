#include "runtime/copy.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace runtime;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Reference {
    std::vector<std::int64_t> dimensions;
    std::vector<std::int64_t> strides;
    std::size_t offset; // elements
};
std::size_t count(const Reference& ref) {
    std::size_t result = 1;
    for (auto dim : ref.dimensions) result *= static_cast<std::size_t>(dim);
    return result;
}
std::vector<std::int64_t> coordinates(std::size_t flat, const Reference& ref) {
    std::vector<std::int64_t> result(ref.dimensions.size());
    for (std::size_t axis = result.size(); axis-- > 0;) {
        result[axis] = static_cast<std::int64_t>(flat % static_cast<std::size_t>(ref.dimensions[axis]));
        flat /= static_cast<std::size_t>(ref.dimensions[axis]);
    }
    return result;
}
std::size_t address(const Reference& ref, const std::vector<std::int64_t>& index) {
    auto result = ref.offset;
    for (std::size_t axis = 0; axis < index.size(); ++axis)
        result += static_cast<std::size_t>(index[axis] * ref.strides[axis]);
    return result;
}
std::size_t span(const Reference& ref) {
    if (count(ref) == 0) return 0;
    std::size_t result = 1;
    for (std::size_t axis = 0; axis < ref.dimensions.size(); ++axis)
        result += static_cast<std::size_t>((ref.dimensions[axis] - 1) * ref.strides[axis]);
    return result;
}
Reference layout(std::vector<std::int64_t> dimensions, std::mt19937& rng) {
    Reference ref{std::move(dimensions), {}, 1 + rng() % 3};
    ref.strides.resize(ref.dimensions.size());
    std::vector<std::size_t> axes(ref.dimensions.size());
    std::iota(axes.begin(), axes.end(), 0);
    std::shuffle(axes.begin(), axes.end(), rng);
    std::size_t bounding = 1;
    for (auto axis : axes) {
        const auto step = bounding + rng() % 3;
        ref.strides[axis] = static_cast<std::int64_t>(step);
        if (ref.dimensions[axis] > 1)
            bounding += static_cast<std::size_t>(ref.dimensions[axis] - 1) * step;
    }
    return ref;
}
template<class T> T value(std::size_t index) {
    return static_cast<T>(static_cast<std::int64_t>(index) * 7 - 19);
}
template<class T> void trial(std::mt19937& rng, std::size_t trial_index) {
    const std::size_t rank = trial_index % (MAX_RANK + 1);
    std::vector<std::int64_t> dimensions(rank);
    for (auto& dim : dimensions) dim = 1 + rng() % 3;
    if (rank && trial_index % 7 == 0) dimensions[rng() % rank] = 0;
    auto ref = layout(dimensions, rng);
    Tensor base = Tensor::allocate_cpu(Shape{static_cast<std::int64_t>(ref.offset + span(ref) + 2)}, dtype_of<T>());
    for (std::size_t i = 0; i < base.numel(); ++i) base.data<T>()[i] = value<T>(i);
    const auto before = testing::cpu_allocation_counts();
    Tensor view = base.view(Shape(ref.dimensions), Stride(ref.strides), ref.offset * sizeof(T));
    std::vector<std::size_t> axes(rank);
    std::iota(axes.begin(), axes.end(), 0);
    std::shuffle(axes.begin(), axes.end(), rng);
    view = view.permute(axes);
    auto old = ref;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        ref.dimensions[axis] = old.dimensions[axes[axis]];
        ref.strides[axis] = old.strides[axes[axis]];
    }
    if (rank) {
        const auto axis = rng() % rank;
        const auto size = ref.dimensions[axis];
        const auto start = static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(size + 1));
        const std::int64_t step = 1 + rng() % 3;
        const auto max_length = start == size ? 0 : 1 + (size - 1 - start) / step;
        const auto length = static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(max_length + 1));
        view = view.slice(axis, start, length, step);
        ref.dimensions[axis] = length;
        if (count(ref)) ref.offset += static_cast<std::size_t>(start * ref.strides[axis]);
        ref.strides[axis] *= step;
    }
    require(testing::cpu_allocation_counts().allocations == before.allocations, "property view allocated backing buffer");
    require(view.numel() == count(ref) && view.data_offset() == ref.offset * sizeof(T), "property view metadata mismatch");
    Tensor materialized = view.contiguous();
    require(materialized.is_contiguous(), "property materialization not contiguous");
    if (!view.is_contiguous() && view.numel())
        require(materialized.storage() != view.storage(), "property materialization still aliases");
    for (std::size_t flat = 0; flat < count(ref); ++flat) {
        const auto index = coordinates(flat, ref);
        const auto expected = value<T>(address(ref, index));
        require(view.at<T>(index) == expected, "property checked strided indexing mismatch");
        require(materialized.data<T>()[flat] == expected, "property materialization/reference mismatch");
    }
    auto dest_ref = layout(ref.dimensions, rng);
    Tensor dest_owner = Tensor::allocate_cpu(Shape{static_cast<std::int64_t>(dest_ref.offset + span(dest_ref) + 2)}, dtype_of<T>());
    for (std::size_t i = 0; i < dest_owner.numel(); ++i) dest_owner.data<T>()[i] = static_cast<T>(-999);
    Tensor destination = dest_owner.view(Shape(dest_ref.dimensions), Stride(dest_ref.strides), dest_ref.offset * sizeof(T));
    const auto copy_before = testing::cpu_allocation_counts();
    require(copy_cpu(view, destination) == count(ref) * sizeof(T), "property copied bytes mismatch");
    require(testing::cpu_allocation_counts().allocations == copy_before.allocations, "property copy allocated buffers");
    std::vector<bool> touched(dest_owner.numel(), false);
    for (std::size_t flat = 0; flat < count(ref); ++flat) {
        const auto index = coordinates(flat, ref);
        require(destination.at<T>(index) == value<T>(address(ref, index)), "property strided destination mismatch");
        touched[address(dest_ref, index)] = true;
    }
    for (std::size_t i = 0; i < touched.size(); ++i)
        if (!touched[i]) require(dest_owner.data<T>()[i] == static_cast<T>(-999), "property copy changed padding");
    if (count(ref)) {
        const auto index = coordinates(count(ref) - 1, ref);
        view.at<T>(index) = static_cast<T>(-123);
        require(base.data<T>()[address(ref, index)] == static_cast<T>(-123), "property alias mutation mismatch");
    }
}
} // namespace
int main() {
    constexpr std::uint32_t seed = 0x51a7;
    constexpr std::size_t trials = 600;
    std::size_t current = 0;
    try {
        std::mt19937 rng(seed);
        const auto before = runtime::testing::cpu_allocation_counts();
        for (; current < trials; ++current) {
            if (current % 2) trial<float>(rng, current);
            else trial<std::int32_t>(rng, current);
        }
        const auto after = runtime::testing::cpu_allocation_counts();
        require(after.live == before.live && after.allocations - before.allocations == after.frees - before.frees,
                "property storage lifetime leak");
        std::cout << "Tensor properties: PASS seed=" << seed << " trials=" << trials
                  << " ranks=0..8 FP32/INT32; padding/offset/permutation/slice/materialization/copy/alias\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Tensor property failure seed=" << seed << " trial=" << current << ": " << error.what() << '\n';
        return 1;
    }
}
