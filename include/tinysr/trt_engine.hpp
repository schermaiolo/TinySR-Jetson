#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace tinysr {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TrtEngine final {
public:
    TrtEngine(const std::filesystem::path& file, TrtLogger& logger);
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    void expect(const char* name, nvinfer1::TensorIOMode mode,
                nvinfer1::DataType dtype, const std::vector<int>& shape) const;
    void bind(const char* name, void* ptr);
    void enqueue(cudaStream_t stream);

private:
    // Destruct in reverse order: context -> engine -> runtime.
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

}  // namespace tinysr
