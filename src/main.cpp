#include "tinysr/image_processor.hpp"
#include "tinysr/pipeline.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string input;
    std::string output;
    std::filesystem::path engines = "models_trt";
    int warmup = 2;
    int runs = 5;
};

void help() {
    std::cout << "TinySR-Jetson v0.1 (unofficial) — fixed 4x super-resolution\n"
              << "Usage: tinysr-jetson --input file.png --output output.png "
              << "[--engines models_trt] [--warmup 2] [--runs 5]\n"
              << "Input must be 128x128. Output is 512x512. No AdaIN or tiling yet.\n";
}

int positive_int(const std::string& arg, bool allow_zero) {
    std::size_t position = 0;
    const int value = std::stoi(arg, &position);
    if (position != arg.size() || value < (allow_zero ? 0 : 1) || value > 10000) {
        throw std::invalid_argument("Invalid repetition count: " + arg);
    }
    return value;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            help();
            std::exit(EXIT_SUCCESS);
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value after " + arg);
        const std::string value = argv[++i];
        if (arg == "--input" || arg == "-i") options.input = value;
        else if (arg == "--output" || arg == "-o") options.output = value;
        else if (arg == "--engines") options.engines = value;
        else if (arg == "--warmup") options.warmup = positive_int(value, true);
        else if (arg == "--runs") options.runs = positive_int(value, false);
        else throw std::invalid_argument("Unknown option: " + arg);
    }
    if (options.input.empty() || options.output.empty()) {
        throw std::invalid_argument("Both --input and --output are required; use --help");
    }
    return options;
}

std::vector<float> stage_values(const std::vector<tinysr::StageTimings>& samples,
                                float tinysr::StageTimings::*member) {
    std::vector<float> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) values.push_back(sample.*member);
    return values;
}

void print_stat(const char* title, std::vector<float> values) {
    std::sort(values.begin(), values.end());
    float sum = 0.f;
    for (float value : values) sum += value;
    const auto p95index = static_cast<std::size_t>((95 * (values.size() - 1) + 99) / 100);
    std::cout << std::left << std::setw(17) << title
              << std::right << std::fixed << std::setprecision(3)
              << " mean=" << std::setw(9) << (sum / values.size()) << " ms"
              << " p95=" << std::setw(9) << values[p95index] << " ms\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options args = parse(argc, argv);
        const auto input = tinysr::load_and_preprocess(args.input);
        tinysr::TinySRPipeline pipeline(args.engines);
        pipeline.print_memory();

        std::vector<float> output;
        for (int i = 0; i < args.warmup; ++i) pipeline.run(input, output);
        std::vector<tinysr::StageTimings> samples;
        samples.reserve(args.runs);
        for (int i = 0; i < args.runs; ++i) {
            samples.push_back(pipeline.run(input, output));
        }
        tinysr::save_image(output, args.output);

        std::cout << "\nCUDA-event timing (" << args.runs
                  << " runs, excluding engine load, resize and PNG save):\n";
        print_stat("H2D", stage_values(samples, &tinysr::StageTimings::h2d_ms));
        print_stat("VAE encoder", stage_values(samples, &tinysr::StageTimings::encoder_ms));
        print_stat("FP32 -> FP16", stage_values(samples, &tinysr::StageTimings::cast_ms));
        print_stat("Transformer", stage_values(samples, &tinysr::StageTimings::transformer_ms));
        print_stat("Residual", stage_values(samples, &tinysr::StageTimings::residual_ms));
        print_stat("VAE decoder", stage_values(samples, &tinysr::StageTimings::decoder_ms));
        print_stat("D2H", stage_values(samples, &tinysr::StageTimings::d2h_ms));
        print_stat("GPU pipeline", stage_values(samples, &tinysr::StageTimings::total_ms));
        std::cout << "Saved: " << args.output << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
