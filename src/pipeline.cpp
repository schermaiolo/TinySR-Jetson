#include "tinysr/pipeline.hpp"
#include "tinysr/cuda_kernels.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace tinysr {

TinySRPipeline::TinySRPipeline(const std::filesystem::path& dir)
    : encoder_(dir / "vae_encoder_512_fp16.engine", logger_),
      transformer_(dir / "tinysr_transformer_64_fp16.engine", logger_),
      decoder_(dir / "vae_decoder_512_fp16.engine", logger_),
      rgb_(kRgbCount * sizeof(float)),
      encoded_(kLatentCount * sizeof(float)),
      latent_(kLatentCount * sizeof(std::uint16_t)),
      prediction_(kLatentCount * sizeof(std::uint16_t)),
      residual_(kLatentCount * sizeof(float)),
      output_(kRgbCount * sizeof(float)) {

    using nvinfer1::DataType;
    using nvinfer1::TensorIOMode;
    const std::vector<int> rgb_shape = {1, 3, kImagePixels, kImagePixels};
    const std::vector<int> latent_shape = {1, kLatentChannels, kLatentPixels, kLatentPixels};

    encoder_.expect("rgb", TensorIOMode::kINPUT, DataType::kFLOAT, rgb_shape);
    encoder_.expect("latent", TensorIOMode::kOUTPUT, DataType::kFLOAT, latent_shape);
    transformer_.expect("latent_input", TensorIOMode::kINPUT, DataType::kHALF, latent_shape);
    transformer_.expect("prediction", TensorIOMode::kOUTPUT, DataType::kHALF, latent_shape);
    decoder_.expect("latent", TensorIOMode::kINPUT, DataType::kFLOAT, latent_shape);
    decoder_.expect("rgb", TensorIOMode::kOUTPUT, DataType::kFLOAT, rgb_shape);

    encoder_.bind("rgb", rgb_.data());
    encoder_.bind("latent", encoded_.data());
    transformer_.bind("latent_input", latent_.data());
    transformer_.bind("prediction", prediction_.data());
    decoder_.bind("latent", residual_.data());
    decoder_.bind("rgb", output_.data());
}

StageTimings TinySRPipeline::run(const std::vector<float>& input,
                                 std::vector<float>& result) {
    if (input.size() != kRgbCount) throw std::invalid_argument("Invalid input tensor size");
    result.resize(kRgbCount);
    auto stream = stream_.get();

    events_[0].record(stream);
    cuda_check(cudaMemcpyAsync(rgb_.data(), input.data(), rgb_.size(),
                               cudaMemcpyHostToDevice, stream), "H2D RGB");
    events_[1].record(stream);

    encoder_.enqueue(stream);
    events_[2].record(stream);

    float_to_half(static_cast<const float*>(encoded_.data()), latent_.data(),
                  kLatentCount, stream);
    events_[3].record(stream);

    transformer_.enqueue(stream);
    events_[4].record(stream);

    subtract_half_to_float(latent_.data(), prediction_.data(),
                           static_cast<float*>(residual_.data()), kLatentCount, stream);
    events_[5].record(stream);

    decoder_.enqueue(stream);
    events_[6].record(stream);

    cuda_check(cudaMemcpyAsync(result.data(), output_.data(), output_.size(),
                               cudaMemcpyDeviceToHost, stream), "D2H RGB");
    events_[7].record(stream);
    events_[7].synchronize();

    StageTimings timing;
    timing.h2d_ms = events_[0].elapsed_to(events_[1]);
    timing.encoder_ms = events_[1].elapsed_to(events_[2]);
    timing.cast_ms = events_[2].elapsed_to(events_[3]);
    timing.transformer_ms = events_[3].elapsed_to(events_[4]);
    timing.residual_ms = events_[4].elapsed_to(events_[5]);
    timing.decoder_ms = events_[5].elapsed_to(events_[6]);
    timing.d2h_ms = events_[6].elapsed_to(events_[7]);
    timing.total_ms = events_[0].elapsed_to(events_[7]);
    return timing;
}

void TinySRPipeline::print_memory() const {
    std::size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    constexpr double mib = 1024.0 * 1024.0;
    std::cout << "CUDA memory after engine load: " << free_bytes / mib
              << " MiB free / " << total_bytes / mib << " MiB total\n";
}

}  // namespace tinysr
