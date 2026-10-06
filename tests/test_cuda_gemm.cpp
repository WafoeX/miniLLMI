// This executable exists only in ENABLE_CUDA builds; do not require a toolkit
// for host-only static analysis.
#if defined(STAGE0_CUDA_BUILD)
#include "stage0/gemm_cuda.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        using namespace stage0;
        CUDA_CHECK(cudaSetDevice(0));
        Stream stream; BlasHandle blas(stream.get());
        const std::vector<Shape> shapes{{1,1,1},{2,3,4},{31,33,17},{65,7,19},{3,5,1},{64,64,64}};
        require(gemm_kernel_from_name("v0") == GemmKernel::V0Naive, "v0 registry selection");
        require(gemm_kernel_from_name("tiled") == GemmKernel::V1Tiled, "v1 registry selection");
        require(gemm_kernel_from_name("cublas") == GemmKernel::CuBlas, "cuBLAS registry selection");
        require(!gemm_kernel_from_name("missing"), "unknown registry selection");
        const auto& tiled = gemm_kernel_config(GemmKernel::V1Tiled);
        require(tiled.bm == 16 && tiled.bn == 16 && tiled.bk == 16 && tiled.tm == 1 && tiled.tn == 1 &&
                    tiled.threads == 256 && tiled.vector_width == 1 && tiled.shared_bytes == 2048,
                "v1 fixed configuration");
        for (const auto s : shapes) {
            std::vector<float> a(elements(s.m,s.k)), b(elements(s.k,s.n)), ref(elements(s.m,s.n));
            for (bool zeros : {false,true}) {
                if (zeros) { std::fill(a.begin(),a.end(),0); std::fill(b.begin(),b.end(),0); }
                else random_inputs(a,b,42);
                cpu_reference(a.data(),b.data(),ref.data(),s);
                constexpr std::size_t guard = 16;
                for (const std::size_t offset : {std::size_t{0}, std::size_t{1}}) {
                    DeviceBuffer da(a.size()+offset), db(b.size()+offset), dc(ref.size()+2*guard+offset);
                    CUDA_CHECK(cudaMemcpyAsync(da.get()+offset,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice,stream.get()));
                    CUDA_CHECK(cudaMemcpyAsync(db.get()+offset,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice,stream.get()));
                    for (const auto kernel : {GemmKernel::V0Naive, GemmKernel::V1Tiled, GemmKernel::CuBlas}) {
                    CUDA_CHECK(cudaMemsetAsync(dc.get(),0xff,(ref.size()+2*guard+offset)*sizeof(float),stream.get()));
                    auto launch = [&] { launch_gemm(kernel, blas.get(), da.get()+offset, db.get()+offset, dc.get()+guard+offset, s, stream.get()); };
                    launch(); launch();
                    std::vector<float> out(ref.size()+2*guard);
                    CUDA_CHECK(cudaMemcpyAsync(out.data(),dc.get()+offset,out.size()*sizeof(float),cudaMemcpyDeviceToHost,stream.get()));
                    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
                    const std::vector<float> actual(out.begin()+guard,out.end()-guard);
                    if (!compare(actual,ref,1e-4,1e-4).passed()) throw std::runtime_error("CUDA GEMM mismatch");
                    for (std::size_t i = 0; i < guard; ++i) {
                        for (auto offset : {i, out.size()-guard+i}) {
                            std::uint32_t bits; std::memcpy(&bits,&out[offset],sizeof(bits));
                            if (bits != 0xffffffffu) throw std::runtime_error("output guard overwritten");
                        }
                    }
                    if (s.m == 64 && !zeros) {
                        EventTimer timer;
                        for (int i = 0; i < 10; ++i) launch();
                        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
                        std::vector<double> times;
                        for (int i = 0; i < 30; ++i) times.push_back(timer.measure(stream.get(),launch));
                        if (!(summarize(times).median > 0)) throw std::runtime_error("invalid events");
                    }
                    }
                }
            }
        }
        std::cout << "CUDA registry/v0/v1/cuBLAS, rectangles/tails/zero/offsets/guards/events: PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "test_cuda_gemm: " << e.what() << '\n'; return 1; }
}
#endif // STAGE0_CUDA_BUILD
