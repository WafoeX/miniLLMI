#pragma once
#include "stage0/common.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace stage0 {
inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " +
                                 expression + ": " + cudaGetErrorString(status));
}
inline void check_cublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " +
                                 expression + ": cuBLAS status " + std::to_string(int(status)));
}
#define CUDA_CHECK(expr) ::stage0::check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) ::stage0::check_cublas((expr), #expr, __FILE__, __LINE__)

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        if (!count || count > SIZE_MAX / sizeof(float)) throw std::invalid_argument("invalid allocation");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(float)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    float* get() const { return ptr_; }
private:
    float* ptr_ = nullptr;
};
class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_{};
};
class BlasHandle {
public:
    explicit BlasHandle(cudaStream_t stream) {
        CUBLAS_CHECK(cublasCreate(&handle_));
        try {
            CUBLAS_CHECK(cublasSetStream(handle_, stream));
            CUBLAS_CHECK(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
            CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH));
            CUBLAS_CHECK(cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED));
        } catch (...) { cublasDestroy(handle_); throw; }
    }
    ~BlasHandle() { cublasDestroy(handle_); }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;
    cublasHandle_t get() const { return handle_; }
private:
    cublasHandle_t handle_{};
};
class EventTimer {
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        try { CUDA_CHECK(cudaEventCreate(&stop_)); }
        catch (...) { cudaEventDestroy(start_); throw; }
    }
    ~EventTimer() { cudaEventDestroy(stop_); cudaEventDestroy(start_); }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;
    template<class Launch> double measure(cudaStream_t stream, Launch&& launch) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
        std::forward<Launch>(launch)();
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        if (!(ms > 0)) throw std::runtime_error("CUDA event returned nonpositive time");
        return ms;
    }
private:
    cudaEvent_t start_{}, stop_{};
};
inline std::string uuid_string(const cudaUUID_t& uuid) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    for (unsigned char b : uuid.bytes) { result += digits[b >> 4]; result += digits[b & 15]; }
    return result;
}
}
