# TinySR-Jetson: C++ TensorRT Implementation of TinySR on NVIDIA Jetson Orin Nano

![Jetson Orin Nano](https://img.shields.io/badge/Jetson-Orin%20Nano%208GB-76B900?style=flat-square&logo=nvidia&logoColor=white)
![JetPack](https://img.shields.io/badge/JetPack-6.2.3-76B900?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-10.3-76B900?style=flat-square)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=cplusplus&logoColor=white)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)](LICENSE)

A native **C++17 / CUDA / TensorRT** implementation of [TinySR](https://github.com/Microtreei/TinySR) for real-world **4× image super-resolution** on NVIDIA Jetson. This project converts the original PyTorch inference models to ONNX and TensorRT and runs the resulting pipeline in **C++**.

## Example

| Input (128 × 128) | TinySR-Jetson (512 × 512) |
|:---:|:---:|
| <img src="assets/examples/input.png" width="256" alt="128x128 input image"> | <img src="assets/examples/output.png" width="256" alt="512x512 TinySR-Jetson output"> |

One example from my Jetson tests. TinySR reconstructs plausible details; the output is not a ground-truth image.

---

## Requirements

| | Tested with |
|---|---|
| Board | NVIDIA Jetson Orin Nano 8GB |
| JetPack / L4T | 6.2.3 / R36.5.2 |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| C++ | C++17 / CUDA C++17 |
| Build | CMake ≥ 3.20, OpenCV 4.8.0, TensorRT development headers |
| Model conversion only | Python 3.10, CUDA-enabled PyTorch 2.8.0, ONNX 1.17.0, diffusers 0.29.1, transformers 4.49.0, peft 0.15.0 |

**[Installation, quick start and model conversion →](docs/README.md)**

---

## Performance

Jetson Orin Nano 8GB, **5 warm-up + 30 measured runs**, CUDA events:

| Stage | Mean | P95 |
|---|---:|---:|
| VAE encoder | 15.24 ms | 15.64 ms |
| Transformer | 171.38 ms | 171.93 ms |
| VAE decoder | 17.63 ms | 17.67 ms |
| **GPU pipeline** | **205.67 ms** | **206.43 ms** |

The pipeline timing includes GPU execution and measured transfers, but **excludes** engine loading, image resizing and PNG saving. Clock/power modes were not controlled, and I am not claiming a speedup over PyTorch.

For one test image, the C++ output reached **53.29 dB PSNR against the original PyTorch output** with `--align_method=nofix`. This measures **port fidelity**, not image restoration quality. More details: [benchmarks](docs/benchmarks.md) · [validation](docs/validation.md).

**Current scope:** 128 × 128 RGB input → 512 × 512 output, batch size 1. No AdaIN, arbitrary resolutions or tiling yet.

---

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The original TinySR architecture files included under `third_party/tinysr_python/` retain their upstream attribution and license. Pretrained weights and compiled TensorRT engines are **not** distributed here.

---

## Credits

The model, training and original PyTorch implementation are by **Linwei Dong et al.** — [Microtreei/TinySR](https://github.com/Microtreei/TinySR), *TinySR: Pruning Diffusion for Real-World Image Super-Resolution* (CVPR 2026 Findings).

I used their published implementation and pretrained weights as the starting point for this Jetson deployment. Thanks to the original authors for making TinySR available.