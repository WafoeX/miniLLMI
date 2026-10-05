#include "stage0/gemm_cuda.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        using namespace stage0;
        CUDA_CHECK(cudaSetDevice(0));
        Stream stream; BlasHandle blas(stream.get());
        const std::vector<Shape> shapes{{1,1,1},{2,3,4},{31,33,17},{65,7,19},{3,5,1},{64,64,64}};
        for (const auto s : shapes) {
            std::vector<float> a(elements(s.m,s.k)), b(elements(s.k,s.n)), ref(elements(s.m,s.n));
            for (bool zeros : {false,true}) {
                if (zeros) { std::fill(a.begin(),a.end(),0); std::fill(b.begin(),b.end(),0); }
                else random_inputs(a,b,42);
                cpu_reference(a.data(),b.data(),ref.data(),s);
                constexpr std::size_t guard = 16;
                DeviceBuffer da(a.size()), db(b.size()), dc(ref.size()+2*guard);
                CUDA_CHECK(cudaMemcpyAsync(da.get(),a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice,stream.get()));
                CUDA_CHECK(cudaMemcpyAsync(db.get(),b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice,stream.get()));
                for (bool library : {false,true}) {
                    CUDA_CHECK(cudaMemsetAsync(dc.get(),0xff,(ref.size()+2*guard)*sizeof(float),stream.get()));
                    auto launch = [&] {
                        if (library) launch_cublas(blas.get(),da.get(),db.get(),dc.get()+guard,s);
                        else launch_naive(da.get(),db.get(),dc.get()+guard,s,stream.get());
                    };
                    launch();
                    std::vector<float> out(ref.size()+2*guard);
                    CUDA_CHECK(cudaMemcpyAsync(out.data(),dc.get(),out.size()*sizeof(float),cudaMemcpyDeviceToHost,stream.get()));
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
        std::cout << "CUDA naive/cuBLAS, rectangles/tails/zero/guards/events: PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "test_cuda_gemm: " << e.what() << '\n'; return 1; }
}
