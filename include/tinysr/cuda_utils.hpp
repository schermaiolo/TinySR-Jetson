#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace tinysr {

inline void cuda_check(cudaError_t err, const char* action) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(err));
    }
}

class CudaStream final {
public:
    CudaStream() {
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
    }
    ~CudaStream() { if (stream_) cudaStreamDestroy(stream_); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t nbytes) : size_(nbytes) {
        cuda_check(cudaMalloc(&data_, nbytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() { return data_; }
    const void* data() const { return data_; }
    std::size_t size() const { return size_; }
private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

class CudaEvent final {
public:
    CudaEvent() { cuda_check(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~CudaEvent() { if (event_) cudaEventDestroy(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    void record(cudaStream_t stream) {
        cuda_check(cudaEventRecord(event_, stream), "cudaEventRecord");
    }
    void synchronize() {
        cuda_check(cudaEventSynchronize(event_), "cudaEventSynchronize");
    }
    float elapsed_to(const CudaEvent& next) const {
        float milliseconds = 0.0f;
        cuda_check(cudaEventElapsedTime(&milliseconds, event_, next.event_),
                   "cudaEventElapsedTime");
        return milliseconds;
    }
private:
    cudaEvent_t event_ = nullptr;
};

}  // namespace tinysr
