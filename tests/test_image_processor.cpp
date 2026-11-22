#include "tinysr/image_processor.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "tinysr_image_test";
    std::filesystem::create_directories(dir);
    const auto input = dir / "input.png";
    const auto output = dir / "output.png";
    // OpenCV BGR: (blue=10, green=20, red=200)
    cv::Mat constant(128, 128, CV_8UC3, cv::Scalar(10, 20, 200));
    if (!cv::imwrite(input.string(), constant)) return 1;
    const auto tensor = tinysr::load_and_preprocess(input.string());
    if (tensor.size() != tinysr::kRgbCount) return 2;
    const float expected_red = 200.f / 255.f * 2.f - 1.f;
    const float expected_blue = 10.f / 255.f * 2.f - 1.f;
    if (std::abs(tensor[0] - expected_red) > 1e-4f) return 3;
    if (std::abs(tensor[2 * 512 * 512] - expected_blue) > 1e-4f) return 4;
    tinysr::save_image(tensor, output.string());
    const auto restored = cv::imread(output.string());
    if (restored.empty() || restored.rows != 512 || restored.cols != 512) return 5;
    const auto value = restored.at<cv::Vec3b>(256, 256);
    if (std::abs(int(value[0]) - 10) > 1 || std::abs(int(value[2]) - 200) > 1) return 6;
    std::filesystem::remove_all(dir);
    std::cout << "Image preprocessing test passed\n";
    return 0;
}
