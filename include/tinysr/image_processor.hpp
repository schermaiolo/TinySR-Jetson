#pragma once

#include <string>
#include <vector>

namespace tinysr {
constexpr int kInputPixels = 128;
constexpr int kImagePixels = 512;
constexpr int kLatentPixels = 64;
constexpr int kLatentChannels = 16;
constexpr int kRgbCount = 3 * kImagePixels * kImagePixels;
constexpr int kLatentCount = kLatentChannels * kLatentPixels * kLatentPixels;

// Converts 128x128 BGR image to float32 NCHW RGB [-1,1] resized to 512x512.
// INTER_CUBIC approximates but is not bit-identical to PyTorch bicubic interpolation.
std::vector<float> load_and_preprocess(const std::string& filename);

// Converts float32 NCHW RGB [-1,1] to a 512x512 8-bit PNG/JPEG.
void save_image(const std::vector<float>& chw, const std::string& filename);
}  // namespace tinysr
