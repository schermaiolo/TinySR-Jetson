# Current limitations and publication checklist

## Already verified on Jetson Orin Nano 8GB

- [x] CMake build of modular native C++/CUDA/TensorRT executable.
- [x] CTest: `image_processor` and `help` passed (2/2).
- [x] End-to-end inference with three TensorRT engines, single 128×128 image → 512×512 PNG.
- [x] Per-stage CUDA-event timings: initial 2+5 run and independently rebuilt 5+30 run benchmarks.
- [x] One output compared against upstream PyTorch with `--align_method=nofix`: **53.7001 dB PSNR** with rebuilt engines.

## Release readiness: verified and outstanding tasks

- [x] Export all three ONNX graphs using `tools/export_models.py` with existing upstream checkout.
- [x] Rebuild all three TensorRT engines on JetPack 6.2.3 into isolated paths.
- [x] Verify the rebuilt transformer against the export-time golden (`RESULT: PASS`).
- [x] Export and rebuild using vendored Python architecture files and local checkpoint data (2026-10-29).
- [x] Export transformer with original `../TinySR` directory temporarily renamed; restore checkout afterward (2026-10-29).
- [ ] Verify repeatability with a **fresh independent checkout and independent checkpoint/asset download**.
- [ ] Add deterministic checksums and a model/export manifest (upstream commit + checkpoint filenames/checksums).
- [ ] Expand parity testing from one illustration to a collection of natural photographs and textures.
- [ ] Add AdaIN to match default upstream post-processing.
- [ ] Implement and validate tiling and arbitrary input resolutions (currently **only 128×128 input**).
- [x] Add 30-run timing of rebuilt pipeline (5 warmups; 206.414 ms mean); updated standalone runtime measured 205.668 ms mean.
- [ ] Repeat under a recorded power mode, clocks and thermal conditions, paired with PyTorch.
- [ ] Test cold start and sustained operation; record memory peak, temperatures and power.
- [ ] Confirm pretrained checkpoint redistribution conditions independently of source-code license.
- [x] Add transformer-specific numerical validation with PASS/FAIL thresholds.
- [ ] Add VAE numerical regression automation and integration tests over multiple inputs.

---

## Notes

- TinySR may hallucinate plausible details; perceived sharpness is not proof of faithful reconstruction.
- The successful **runtime port** does not mean TensorRT is already faster than PyTorch. Existing measurements were collected under different conditions.
- Engine plan files are not generally portable; rebuild on target, and retain known-good engine backups during upgrades.
- Currently AdaIN alignment, dynamic resolution and streaming/video are intentionally unsupported.

The updated standalone run produced **53.2869 dB PSNR** vs upstream PyTorch `nofix` on the single test image. Earlier values (**53.7001 dB**) describe a different locally rebuilt engine set. Do not combine results from different builds as though they were one experiment.
