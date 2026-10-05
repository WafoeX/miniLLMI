#include "runtime/device.hpp"
#include "runtime/shape.hpp"
#include "runtime/storage.hpp"
#include "runtime/tensor.hpp"
#include "runtime/copy.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Exception, class F> void throws(F&& fn, const char* message) {
    try { fn(); }
    catch (const Exception& error) {
        require(!std::string(error.what()).empty(), "exception must have diagnostic");
        return;
    }
    throw std::runtime_error(message);
}
void test_metadata() {
    using namespace runtime;
    require(numel(Shape{}) == 1 && nbytes(Shape{}, DType::FP32) == 4, "scalar semantics");
    require(numel(Shape{2, 0, 3}) == 0 && nbytes(Shape{0}, DType::INT32) == 0, "empty semantics");
    require(numel(Shape{2, 3, 4}) == 24 && nbytes(Shape{2, 3, 4}, DType::INT32) == 96, "byte counts");
    require(contiguous_stride(Shape{2, 3, 4}) == Stride{12, 4, 1}, "row-major strides");
    require(contiguous_stride(Shape{2, 0, 3}) == Stride{3, 3, 1}, "positive empty strides");
    require(storage_span_bytes(Shape{2, 3}, Stride{4, 1}, DType::FP32) == 28, "strided span");
    require(storage_span_bytes(Shape{}, Stride{}, DType::FP32) == 4, "scalar span");
    require(Shape(std::vector<std::int64_t>(MAX_RANK, 1)).rank() == MAX_RANK, "rank limit");
    throws<std::invalid_argument>([] { Shape bad{-1}; }, "negative dimension accepted");
    throws<std::invalid_argument>([] { Stride bad{-1}; }, "negative stride accepted");
    throws<std::invalid_argument>([] { Shape bad(std::vector<std::int64_t>(MAX_RANK + 1, 1)); }, "rank overflow accepted");
    throws<std::out_of_range>([] { (void)Shape{1}[1]; }, "axis bounds accepted");
    const auto huge = std::numeric_limits<std::int64_t>::max();
    require(numel(Shape{huge, huge, 0}) == 0, "empty numel must short-circuit");
    throws<std::overflow_error>([&] { (void)numel(Shape{huge, huge}); }, "numel overflow accepted");
    throws<std::overflow_error>([&] { (void)nbytes(Shape{huge}, DType::FP32); }, "byte overflow accepted");
    throws<std::overflow_error>([&] { (void)contiguous_stride(Shape{1, huge, 2}); }, "stride overflow accepted");
    throws<std::overflow_error>([&] { (void)storage_span_bytes(Shape{3}, Stride{huge}, DType::FP32); }, "span overflow accepted");
    throws<std::overflow_error>([] { (void)checked_add(std::numeric_limits<std::size_t>::max(), 1); }, "addition overflow accepted");
    throws<std::invalid_argument>([] { (void)storage_span_bytes(Shape{2}, Stride{}, DType::FP32); }, "rank mismatch accepted");
    throws<std::invalid_argument>([] { (void)dtype_size(static_cast<DType>(99)); }, "unknown dtype accepted");
    require(dtype_of<float>() == DType::FP32 && dtype_of<const std::int32_t>() == DType::INT32, "typed metadata");
    require(Device{} == Device(DeviceType::CPU, 0), "CPU device");
    require(Device(DeviceType::CUDA, 2).index() == 2, "backend-neutral CUDA declaration");
    throws<std::invalid_argument>([] { Device bad(DeviceType::CPU, 1); }, "invalid CPU index accepted");
    throws<std::invalid_argument>([] { Device bad(DeviceType::CUDA, -1); }, "negative device accepted");
    throws<std::invalid_argument>([] { Device bad(static_cast<DeviceType>(99)); }, "unknown device accepted");
}
void test_storage() {
    using namespace runtime;
    const auto before = testing::cpu_allocation_counts();
    std::weak_ptr<Storage> weak;
    {
        auto source = Storage::allocate_cpu(64);
        require(source->device() == Device{} && source->capacity_bytes() == 64 && source->data(), "CPU storage metadata");
        weak = source;
        auto alias = source;
        require(alias.use_count() == 2 && alias->data() == source->data(), "shared storage alias");
        source.reset();
        require(!weak.expired() && alias.use_count() == 1, "alias must keep storage alive");
        static_cast<unsigned char*>(alias->data())[63] = 42;
    }
    auto after = testing::cpu_allocation_counts();
    require(weak.expired() && after.allocations == before.allocations + 1 &&
            after.frees == before.frees + 1 && after.live == before.live, "exactly one CPU release");
    auto empty = Storage::allocate_cpu(0);
    require(empty->data() == nullptr && empty->capacity_bytes() == 0, "zero-byte storage policy");
    require(testing::cpu_allocation_counts().allocations == after.allocations, "empty buffer must not allocate");
    testing::fail_next_cpu_allocation();
    throws<std::bad_alloc>([] { (void)Storage::allocate_cpu(16); }, "failure injection ignored");
    require(testing::cpu_allocation_counts().live == before.live, "failed allocation must not leak");
    int calls = 0;
    auto token = std::make_shared<int>(7);
    std::weak_ptr<int> weak_token = token;
    {
        auto* pointer = new std::int32_t[4]{};
        auto wrapped = Storage::wrap(Device{}, 16, pointer,
            [token, &calls](void* p) noexcept { ++calls; delete[] static_cast<std::int32_t*>(p); });
        token.reset();
        auto alias = wrapped;
        wrapped.reset();
        require(!weak_token.expired() && calls == 0, "custom deleter capture/lifetime");
    }
    require(calls == 1 && weak_token.expired(), "custom deleter exactly once");
    {
        auto zero = Storage::wrap(Device{}, 0, nullptr, [&calls](void*) noexcept { ++calls; });
    }
    require(calls == 2, "zero wrapped deleter exactly once");
    int invalid_calls = 0;
    throws<std::invalid_argument>([&] {
        (void)Storage::wrap(Device{}, 4, nullptr, [&invalid_calls](void*) { ++invalid_calls; });
    }, "nonnull policy accepted");
    require(invalid_calls == 0, "invalid wrap must not transfer ownership");
    throws<std::invalid_argument>([] { (void)Storage::wrap(Device{}, 0, nullptr, {}); }, "missing deleter accepted");
}
void test_construction() {
    using namespace runtime;
    for (const Shape& shape : {Shape{5}, Shape{2, 3}, Shape{2, 3, 4}, Shape{2, 3, 4, 5}}) {
        auto tensor = Tensor::allocate_cpu(shape);
        require(tensor.is_contiguous() && tensor.data_offset() == 0 && tensor.nbytes() == tensor.numel() * 4, "contiguous construction");
        for (std::size_t i = 0; i < tensor.numel(); ++i) tensor.data<float>()[i] = static_cast<float>(i);
        for (std::size_t i = 0; i < tensor.numel(); ++i) {
            std::vector<std::int64_t> index(shape.rank());
            auto remainder = i;
            for (std::size_t axis = shape.rank(); axis-- > 0;) {
                index[axis] = static_cast<std::int64_t>(remainder % as_size(shape[axis]));
                remainder /= as_size(shape[axis]);
            }
            require(tensor.at<float>(index) == static_cast<float>(i), "1D-4D checked address mapping");
        }
    }
    auto scalar = Tensor::allocate_cpu(Shape{});
    require(scalar.at<float>({}) == 0 && scalar.is_contiguous(), "initialized scalar");
    scalar.at<float>({}) = 7;
    require(*scalar.data<float>() == 7, "scalar access");
    auto ids = Tensor::allocate_cpu(Shape{3}, DType::INT32);
    ids.at<std::int32_t>({1}) = 257;
    const auto& const_ids = ids;
    require(const_ids.at<std::int32_t>({1}) == 257 && const_ids.data<std::int32_t>()[0] == 0, "INT32 token-ID/const access");
    throws<std::invalid_argument>([&] { (void)ids.data<float>(); }, "dtype mismatch accepted");
    throws<std::invalid_argument>([&] { (void)ids.at<float>({0}); }, "indexed dtype mismatch accepted");
    throws<std::invalid_argument>([&] { (void)ids.at<std::int32_t>({}); }, "index rank mismatch accepted");
    throws<std::out_of_range>([&] { (void)ids.at<std::int32_t>({-1}); }, "negative index accepted");
    throws<std::out_of_range>([&] { (void)ids.at<std::int32_t>({3}); }, "out-of-bounds accepted");
    auto empty = Tensor::allocate_cpu(Shape{2, 0, 3});
    require(empty.data<float>() == nullptr && empty.is_contiguous() && empty.nbytes() == 0, "empty tensor policy");
    throws<std::out_of_range>([&] { (void)empty.at<float>({0, 0, 0}); }, "empty indexing accepted");
    auto storage = Storage::allocate_cpu(32);
    Tensor offset(storage, DType::FP32, Shape{2}, Stride{1}, 24);
    require(offset.data_offset() == 24 && offset.is_contiguous(), "offset subrange");
    Tensor empty_end(storage, DType::FP32, Shape{0}, Stride{1}, 32);
    require(empty_end.data<float>() == nullptr, "empty one-past offset");
    throws<std::invalid_argument>([] { Tensor bad(nullptr, DType::FP32, Shape{}, Stride{}); }, "null storage accepted");
    throws<std::invalid_argument>([&] { Tensor bad(storage, DType::FP32, Shape{2}, Stride{}); }, "stride rank mismatch accepted");
    throws<std::invalid_argument>([&] { Tensor bad(storage, DType::FP32, Shape{2}, Stride{0}); }, "broadcast stride accepted");
    throws<std::invalid_argument>([&] { Tensor bad(storage, DType::FP32, Shape{2, 2}, Stride{1, 1}); }, "overlap accepted");
    throws<std::invalid_argument>([&] { Tensor bad(storage, DType::FP32, Shape{1}, Stride{1}, 2); }, "unaligned offset accepted");
    throws<std::out_of_range>([&] { Tensor bad(storage, DType::FP32, Shape{3}, Stride{1}, 24); }, "capacity overrun accepted");
    throws<std::out_of_range>([&] { Tensor bad(storage, DType::FP32, Shape{0}, Stride{1}, 36); }, "empty past capacity accepted");
    throws<std::overflow_error>([&] {
        Tensor bad(storage, DType::FP32, Shape{2}, Stride{1}, std::numeric_limits<std::size_t>::max() - 3);
    }, "offset/span overflow accepted");
    auto* raw = new unsigned char[9];
    auto misaligned = Storage::wrap(Device{}, 8, raw + 1, [raw](void*) noexcept { delete[] raw; });
    throws<std::invalid_argument>([&] { Tensor bad(misaligned, DType::FP32, Shape{1}, Stride{1}); }, "misaligned pointer accepted");
    // Device declaration compiles without CUDA. This is not an actual GPU allocation.
    auto simulated = Storage::wrap(Device(DeviceType::CUDA, 0), 4, scalar.data<float>(), [](void*) noexcept {});
    Tensor cuda_metadata(simulated, DType::FP32, Shape{}, Stride{});
    throws<std::runtime_error>([&] { (void)cuda_metadata.data<float>(); }, "CUDA host dereference accepted");
}
void test_views() {
    using namespace runtime;
    auto source = Tensor::allocate_cpu(Shape{2, 3, 4});
    for (std::size_t i = 0; i < source.numel(); ++i) source.data<float>()[i] = static_cast<float>(i);
    const auto before = testing::cpu_allocation_counts();
    auto reshaped = source.reshape(Shape{6, 4});
    require(reshaped.storage() == source.storage() && reshaped.data_offset() == 0 && reshaped.is_contiguous(), "reshape must share storage");
    reshaped.at<float>({1, 2}) = 123;
    require(source.at<float>({0, 1, 2}) == 123, "reshape mutation alias");
    auto head = source.view(Shape{3, 4}, Stride{4, 1}, 12 * 4);
    require(head.at<float>({1, 2}) == source.at<float>({1, 1, 2}), "byte-relative view address");
    auto nested = head.view(Shape{4}, Stride{1}, 4 * 4);
    require(nested.data_offset() == 16 * 4 && nested.data<float>() == source.data<float>() + 16, "nested byte offset");
    auto range = source.narrow(1, 1, 2);
    require(range.shape() == Shape{2, 2, 4} && range.stride() == source.stride() && range.data_offset() == 16, "narrow metadata");
    auto stepped = source.slice(2, 1, 2, 2);
    require(stepped.stride() == Stride{12, 4, 2} && !stepped.is_contiguous(), "positive-step slice metadata");
    require(stepped.at<float>({1, 1, 1}) == source.at<float>({1, 1, 3}), "positive-step slice value");
    stepped.at<float>({1, 1, 1}) = -42;
    require(source.at<float>({1, 1, 3}) == -42, "slice mutation alias");
    auto empty = stepped.slice(2, 2, 0, 2);
    require(empty.numel() == 0 && empty.data_offset() == stepped.data_offset() && empty.data<float>() == nullptr, "empty slice offset rule");
    require(testing::cpu_allocation_counts().allocations == before.allocations, "reshape/view/slice buffer allocation delta must be zero");
    auto empty_reshape = Tensor::allocate_cpu(Shape{0, 3}).reshape(Shape{2, 0});
    require(empty_reshape.numel() == 0, "empty reshape");
    auto scalar = Tensor::allocate_cpu(Shape{}).reshape(Shape{1, 1});
    require(scalar.numel() == 1, "scalar reshape");
    auto after = testing::cpu_allocation_counts();
    // scalar factory above intentionally allocates; all metadata-only operations do not.
    require(after.allocations == before.allocations + 1, "views unexpectedly allocated backing buffer");
    const auto aliases = source.storage().use_count();
    { auto temporary = source.reshape(Shape{24}); require(source.storage().use_count() == aliases + 1, "view refcount"); }
    require(source.storage().use_count() == aliases, "view destruction changed live source");
    std::weak_ptr<Storage> weak;
    {
        auto survivor = [&] {
            auto owner = Tensor::allocate_cpu(Shape{4});
            owner.at<float>({2}) = 9;
            weak = owner.storage();
            return owner.narrow(0, 2, 1);
        }();
        require(!weak.expired() && survivor.at<float>({0}) == 9, "view outlives source");
    }
    require(weak.expired(), "last view must release storage");
    throws<std::invalid_argument>([&] { (void)source.reshape(Shape{25}); }, "numel-changing reshape accepted");
    throws<std::invalid_argument>([&] { (void)stepped.reshape(Shape{12}); }, "noncontiguous reshape accepted");
    throws<std::out_of_range>([&] { (void)source.view(Shape{1}, Stride{1}, source.nbytes()); }, "view past capacity accepted");
    throws<std::invalid_argument>([&] { (void)source.view(Shape{2, 2}, Stride{1, 1}); }, "overlapping view accepted");
    throws<std::out_of_range>([&] { (void)source.slice(3, 0, 1); }, "slice axis accepted");
    throws<std::invalid_argument>([&] { (void)source.slice(1, -1, 1); }, "negative slice accepted");
    throws<std::invalid_argument>([&] { (void)source.slice(1, 0, -1); }, "negative length accepted");
    throws<std::invalid_argument>([&] { (void)source.slice(1, 0, 1, 0); }, "zero step accepted");
    throws<std::invalid_argument>([&] { (void)source.slice(1, 0, 1, -1); }, "negative step accepted");
    throws<std::out_of_range>([&] { (void)source.slice(1, 3, 1); }, "nonempty end slice accepted");
    throws<std::out_of_range>([&] { (void)source.slice(1, 4, 0); }, "empty out-of-bounds start accepted");
    throws<std::out_of_range>([&] { (void)source.slice(1, 1, 2, 2); }, "stepped overrun accepted");
    throws<std::overflow_error>([&] { (void)source.slice(1, 0, 1, std::numeric_limits<std::int64_t>::max()); }, "slice stride overflow accepted");
    throws<std::overflow_error>([&] { (void)head.view(Shape{0}, Stride{1}, std::numeric_limits<std::size_t>::max()); }, "nested offset overflow accepted");
}
void test_transpose_copy() {
    using namespace runtime;
    auto source = Tensor::allocate_cpu(Shape{2, 3});
    for (std::size_t i = 0; i < source.numel(); ++i) source.data<float>()[i] = static_cast<float>(i);
    const auto before = testing::cpu_allocation_counts();
    auto transposed = source.transpose(0, 1);
    require(transposed.shape() == Shape{3, 2} && transposed.stride() == Stride{1, 3} && !transposed.is_contiguous(), "2D transpose strides/contiguity");
    require(transposed.at<float>({2, 1}) == source.at<float>({1, 2}), "2D transpose indexing");
    transposed.at<float>({1, 0}) = 99;
    require(source.at<float>({0, 1}) == 99, "transpose mutation alias");
    require(testing::cpu_allocation_counts().allocations == before.allocations, "transpose buffer allocation delta must be zero");
    auto materialized = transposed.contiguous();
    require(materialized.is_contiguous() && materialized.storage() != source.storage() &&
            testing::cpu_allocation_counts().allocations == before.allocations + 1, "materialization must allocate once");
    for (std::int64_t i = 0; i < 3; ++i) for (std::int64_t j = 0; j < 2; ++j)
        require(materialized.at<float>({i, j}) == source.at<float>({j, i}), "materialized transpose values");
    materialized.at<float>({0, 0}) = -7;
    require(source.at<float>({0, 0}) == 0, "materialized data must be independent");
    require(source.contiguous().storage() == source.storage(), "already contiguous is an alias");
    auto destination = Tensor::allocate_cpu(transposed.shape());
    const auto copy_before = testing::cpu_allocation_counts();
    require(copy_cpu(transposed, destination) == 24, "copy reports logical bytes");
    require(testing::cpu_allocation_counts().allocations == copy_before.allocations, "copy allocates no buffers");
    require(copy_cpu(destination, destination) == 0, "exact self-copy no-op");
    auto cube = Tensor::allocate_cpu(Shape{2, 3, 4}, DType::INT32);
    for (std::size_t i = 0; i < cube.numel(); ++i) cube.data<std::int32_t>()[i] = static_cast<std::int32_t>(i);
    auto permutation = cube.permute({2, 0, 1});
    require(permutation.shape() == Shape{4, 2, 3} && permutation.stride() == Stride{1, 12, 4} && !permutation.is_contiguous(), "3D permutation strides");
    auto dense = permutation.contiguous();
    for (std::int64_t i = 0; i < 4; ++i) for (std::int64_t j = 0; j < 2; ++j) for (std::int64_t k = 0; k < 3; ++k)
        require(dense.at<std::int32_t>({i, j, k}) == cube.at<std::int32_t>({j, k, i}), "3D materialization values");
    auto padded_owner = Tensor::allocate_cpu(Shape{4, 2, 6}, DType::INT32);
    auto padded = padded_owner.slice(2, 0, 3, 2);
    require(copy_cpu(permutation, padded) == cube.nbytes(), "strided destination copy");
    require(padded.at<std::int32_t>({3, 1, 2}) == cube.at<std::int32_t>({1, 2, 3}) &&
            padded_owner.at<std::int32_t>({3, 1, 5}) == 0, "strided copy preserves holes");
    auto overlap_owner = Tensor::allocate_cpu(Shape{8});
    auto left = overlap_owner.narrow(0, 0, 4);
    auto right = overlap_owner.narrow(0, 2, 4);
    throws<std::invalid_argument>([&] { (void)copy_cpu(left, right); }, "overlapping spans accepted");
    auto disjoint = overlap_owner.narrow(0, 4, 4);
    left.at<float>({0}) = 8;
    require(copy_cpu(left, disjoint) == 16 && disjoint.at<float>({0}) == 8, "disjoint same-storage copy");
    auto second_wrapper = Storage::wrap(Device{}, overlap_owner.nbytes(), overlap_owner.data<float>(), [](void*) noexcept {});
    Tensor duplicate(second_wrapper, DType::FP32, Shape{4}, Stride{1});
    throws<std::invalid_argument>([&] { (void)copy_cpu(left, duplicate); }, "overlap across Storage wrappers accepted");
    throws<std::invalid_argument>([&] { (void)copy_cpu(source, transposed); }, "copy shape mismatch accepted");
    auto int_destination = Tensor::allocate_cpu(source.shape(), DType::INT32);
    throws<std::invalid_argument>([&] { (void)copy_cpu(source, int_destination); }, "copy dtype mismatch accepted");
    throws<std::invalid_argument>([&] { (void)source.permute({0, 0}); }, "duplicate permutation accepted");
    throws<std::invalid_argument>([&] { (void)source.permute({0}); }, "permutation rank mismatch accepted");
    throws<std::invalid_argument>([&] { (void)source.permute({0, 2}); }, "invalid permutation axis accepted");
    throws<std::out_of_range>([&] { (void)source.transpose(0, 2); }, "transpose axis accepted");
    auto singleton = Tensor::allocate_cpu(Shape{2, 1, 3}).transpose(0, 1);
    require(singleton.is_contiguous(), "singleton axes do not constrain contiguous layout");
    auto empty = Tensor::allocate_cpu(Shape{2, 0, 3}).transpose(0, 2);
    const auto empty_before = testing::cpu_allocation_counts();
    auto empty_dense = empty.contiguous();
    require(empty_dense.is_contiguous() && copy_cpu(empty, empty_dense) == 0 &&
            testing::cpu_allocation_counts().allocations == empty_before.allocations, "empty materialization has no backing allocation");
    auto scalar = Tensor::allocate_cpu(Shape{});
    require(scalar.permute({}).is_contiguous(), "scalar identity permutation");
    auto simulated = Storage::wrap(Device(DeviceType::CUDA, 0), source.nbytes(), source.data<float>(), [](void*) noexcept {});
    Tensor cuda_tensor(simulated, DType::FP32, source.shape(), source.stride());
    throws<std::runtime_error>([&] { (void)cuda_tensor.contiguous(); }, "CUDA contiguous silently uses host");
    throws<std::runtime_error>([&] { (void)cuda_tensor.transpose(0, 1).contiguous(); }, "CUDA strided materialization accepted");
    throws<std::runtime_error>([&] { (void)copy_cpu(cuda_tensor, source); }, "CUDA source copy accepted");
    throws<std::runtime_error>([&] { (void)copy_cpu(source, cuda_tensor); }, "CUDA destination copy accepted");
    std::cout << "Allocation checks: views/transpose=0 buffers; nonempty materialize=1; copy=24 logical bytes, 0 buffers\n";
}
} // namespace
int main() {
    try {
        test_metadata();
        test_storage();
        test_construction();
        test_views();
        test_transpose_copy();
        std::cout << "Tensor metadata/storage/construction/views/copy: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_tensor: " << error.what() << '\n';
        return 1;
    }
}
