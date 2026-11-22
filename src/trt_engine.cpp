#include "tinysr/trt_engine.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace tinysr {

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT] " << msg << '\n';
    }
}

TrtEngine::TrtEngine(const std::filesystem::path& file, TrtLogger& logger) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open TensorRT engine: " + file.string());
    std::vector<char> bytes{std::istreambuf_iterator<char>(stream),
                            std::istreambuf_iterator<char>()};
    if (bytes.empty()) throw std::runtime_error("Empty TensorRT engine: " + file.string());

    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed: " + file.string());
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("createExecutionContext failed: " + file.string());
    std::cout << "Loaded: " << file.string() << '\n';
}

void TrtEngine::expect(const char* name, nvinfer1::TensorIOMode mode,
                       nvinfer1::DataType dtype, const std::vector<int>& shape) const {
    // getTensorIOMode returns kNONE for unknown names.
    if (engine_->getTensorIOMode(name) != mode) {
        throw std::runtime_error(std::string("Missing or incorrectly directed tensor: ") + name);
    }
    if (engine_->getTensorDataType(name) != dtype) {
        throw std::runtime_error(std::string("Unexpected TensorRT dtype: ") + name);
    }
    const auto dims = context_->getTensorShape(name);
    if (dims.nbDims != static_cast<int>(shape.size())) {
        throw std::runtime_error(std::string("Unexpected TensorRT rank: ") + name);
    }
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] != shape[static_cast<std::size_t>(i)]) {
            throw std::runtime_error(std::string("Unexpected TensorRT shape: ") + name);
        }
    }
}

void TrtEngine::bind(const char* name, void* ptr) {
    if (!ptr || !context_->setTensorAddress(name, ptr)) {
        throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
    }
}

void TrtEngine::enqueue(cudaStream_t stream) {
    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
}

}  // namespace tinysr
