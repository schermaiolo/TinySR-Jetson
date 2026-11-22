#pragma once

#include "tinysr/cuda_utils.hpp"
#include "tinysr/image_processor.hpp"
#include "tinysr/trt_engine.hpp"

#include <array>
#include <filesystem>
#include <vector>

namespace tinysr {

struct StageTimings {
    float h2d_ms = 0;
    float encoder_ms = 0;
    float cast_ms = 0;
    float transformer_ms = 0;
    float residual_ms = 0;
    float decoder_ms = 0;
    float d2h_ms = 0;
    float total_ms = 0;
};

class TinySRPipeline final {
public:
    explicit TinySRPipeline(const std::filesystem::path& engine_dir);

    // The input is already an NCHW float32 tensor of size 1x3x512x512.
    // Returns a 1x3x512x512 RGB image in [-1,1].
    StageTimings run(const std::vector<float>& input, std::vector<float>& result);

    void print_memory() const;

private:
    CudaStream stream_;
    TrtLogger logger_;
    TrtEngine encoder_;
    TrtEngine transformer_;
    TrtEngine decoder_;
    DeviceBuffer rgb_;
    DeviceBuffer encoded_;
    DeviceBuffer latent_;      // half precision
    DeviceBuffer prediction_;  // half precision
    DeviceBuffer residual_;    // float32 VAE input
    DeviceBuffer output_;
    std::array<CudaEvent, 8> events_;
};

}  // namespace tinysr
