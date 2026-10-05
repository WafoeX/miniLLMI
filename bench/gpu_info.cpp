#include "stage0/gemm_cuda.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count == 0) throw std::runtime_error("no visible CUDA devices");
        int selected = -1;
        if (argc == 3 && std::string(argv[1]) == "--device") {
            std::size_t used = 0; selected = std::stoi(argv[2], &used);
            if (used != std::string(argv[2]).size() || selected < 0 || selected >= count)
                throw std::invalid_argument("invalid device index");
        } else if (argc != 1) throw std::invalid_argument("usage: gpu_info [--device INDEX]");
        int runtime = 0, driver = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
        std::cout << "CUDA Runtime Version: " << runtime << "\nCUDA Driver API Version: " << driver << '\n';
        for (int device = 0; device < count; ++device) {
            if (selected >= 0 && device != selected) continue;
            cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p, device));
            std::cout << "\nDevice: " << device << "\nGPU Name: " << p.name
                      << "\nGPU UUID (hex): " << stage0::uuid_string(p.uuid)
                      << "\nCompute Capability: " << p.major << '.' << p.minor
                      << "\nSM Count: " << p.multiProcessorCount
                      << "\nGlobal Memory (bytes): " << p.totalGlobalMem
                      << "\nMemory Bus Width (bits): " << p.memoryBusWidth;
#if CUDART_VERSION < 13000
            std::cout << "\nMemory Clock (kHz, CUDA property): " << p.memoryClockRate;
#else
            std::cout << "\nMemory Clock: removed from CUDA 13 device properties; see nvidia-smi clock query below";
#endif
            std::cout << "\nShared Memory Per Block (bytes): " << p.sharedMemPerBlock
                      << "\nRegisters Per Block: " << p.regsPerBlock
                      << "\nWarp Size: " << p.warpSize
                      << "\nMax Threads Per Block: " << p.maxThreadsPerBlock
                      << "\nMax Grid Size: " << p.maxGridSize[0] << ',' << p.maxGridSize[1] << ',' << p.maxGridSize[2]
                      << "\nECC Enabled: " << p.ECCEnabled << '\n';
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "gpu_info: " << e.what() << '\n'; return 1; }
}
