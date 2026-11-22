#include "tinysr/image_processor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace tinysr {

std::vector<float> load_and_preprocess(const std::string& filename) {
    const auto image = cv::imread(filename, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("Cannot read image: " + filename);
    if (image.rows != kInputPixels || image.cols != kInputPixels) {
        throw std::runtime_error("Only 128x128 inputs are supported in v0.1 (fixed 4x mode). Got " +
                                 std::to_string(image.cols) + "x" + std::to_string(image.rows));
    }

    cv::Mat normalized, resized;
    image.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    cv::resize(normalized, resized, cv::Size(kImagePixels, kImagePixels),
               0.0, 0.0, cv::INTER_CUBIC);

    std::vector<float> chw(kRgbCount);
    constexpr int stride = kImagePixels * kImagePixels;
    for (int y = 0; y < kImagePixels; ++y) {
        for (int x = 0; x < kImagePixels; ++x) {
            const auto bgr = resized.at<cv::Vec3f>(y, x);
            const int pixel = y * kImagePixels + x;
            for (int c = 0; c < 3; ++c) {
                chw[static_cast<std::size_t>(c * stride + pixel)] = bgr[2 - c] * 2.f - 1.f;
            }
        }
    }
    return chw;
}

void save_image(const std::vector<float>& chw, const std::string& filename) {
    if (chw.size() != kRgbCount) throw std::runtime_error("Output RGB tensor has wrong size");
    cv::Mat image(kImagePixels, kImagePixels, CV_8UC3);
    constexpr int stride = kImagePixels * kImagePixels;
    for (int y = 0; y < kImagePixels; ++y) {
        for (int x = 0; x < kImagePixels; ++x) {
            auto& bgr = image.at<cv::Vec3b>(y, x);
            const int pixel = y * kImagePixels + x;
            for (int c = 0; c < 3; ++c) {
                const float raw = chw[static_cast<std::size_t>(c * stride + pixel)];
                if (!std::isfinite(raw)) throw std::runtime_error("Non-finite RGB output");
                const float mapped = (std::clamp(raw, -1.f, 1.f) + 1.f) * 127.5f;
                bgr[2 - c] = static_cast<unsigned char>(std::lround(mapped));
            }
        }
    }
    const std::filesystem::path destination(filename);
    if (!destination.parent_path().empty()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    if (!cv::imwrite(filename, image)) throw std::runtime_error("Cannot write image: " + filename);
}

}  // namespace tinysr
