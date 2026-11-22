# Benchmarks — measured on Jetson Orin Nano 8GB

## Hardware and software

- Jetson Orin Nano 8GB, JetPack 6.2.3, L4T R36.5.2
- CUDA 12.6, TensorRT 10.3.0; native C++17/CUDA pipeline
- Fixed input: 128×128 → 512×512 output
- Model: TinySR transformer, LoRA merged, FP16; VAE engines with FP32 I/O and TensorRT FP16 enabled

**Caution:** GPU clock, power mode, ambient temperature and thermals were not recorded systematically. The source image was one downsampled monochrome illustration, *not* a benchmark dataset. The initial set of VAE engines issued an engine portability warning. The three engines used for the newer 30-run benchmark were rebuilt on JetPack 6.2.3 and loaded without that warning.

---

## Modular C++ runtime (actual run)

Command:

```bash
./build/tinysr-jetson \
  --input ../TinySR/dataset/p2j_single/test.png \
  --output outputs/test.png --warmup 2 --runs 5
```

CUDA-event timings from **2 warmup + 5 measured runs**:

| Stage | Mean (ms) | p95 (ms) |
|---|---:|---:|
| H2D | 0.572 | 0.645 |
| VAE encoder | 15.695 | 17.959 |
| FP32 → FP16 | 0.009 | 0.011 |
| Transformer | 173.734 | 183.379 |
| Residual | 0.013 | 0.013 |
| VAE decoder | 17.834 | 18.093 |
| D2H | 0.785 | 0.790 |
| **GPU pipeline** | **208.642** | **220.541** |

**Interpretation:** approximately **4.79 GPU pipeline executions/s** (1/0.208642 s); the transformer accounted for about **83.3%** of the measured pipeline time. The pipeline timing **excludes** engine load, CPU image preprocessing, PNG save and possibly other host-side work; do not label it end-to-end wall clock FPS. With only 5 measured runs, p95 is not robust.

After the engines loaded, the application printed: `CUDA memory after engine load: 5243.56 MiB free / 7607.38 MiB total`. This is a snapshot of **free device-visible memory**, not an isolated peak allocation attributable to the application.

CTest result: **2/2 tests passed** (`image_processor` and `help`), in 0.10 s.

---

## Independent full-rebuild run (2026-10-29)

The consolidated exporter produced three new ONNX models, and `scripts/build_engines.sh`
created three engines under a separate validation directory on the target Jetson.
Run with **5 warmups + 30 measured inferences**:

```bash
./build/tinysr-jetson \
  --engines "$HOME/paper2jetson/tinysr-validation/engines" \
  --input ../TinySR/dataset/p2j_single/test.png \
  --output "$HOME/paper2jetson/tinysr-validation/outputs/rebuilt.png" \
  --warmup 5 --runs 30
```

| Stage | Mean (ms) | p95 (ms) |
|---|---:|---:|
| H2D | 0.561 | 0.601 |
| VAE encoder | 15.460 | 15.691 |
| FP32 → FP16 | 0.009 | 0.009 |
| Transformer | 171.813 | 172.446 |
| Residual | 0.013 | 0.014 |
| VAE decoder | 17.630 | 17.926 |
| D2H | 0.929 | 1.039 |
| **GPU pipeline** | **206.414** | **207.329** |

About **4.84 measured GPU pipeline executions/s**; excludes setup and image I/O.
After engine loading, CUDA reported **5327.96 MiB free** of 7607.38 MiB device-visible memory. The transformer build emitted a **CUDA OOM while evaluating one tactic** but subsequently produced a valid serialized engine and returned exit status 0. This is a warning about build-time memory pressure, *not* proof of a failed engine build.

---

## Standalone `trtexec` results (different runs)

| Component | TensorRT GPU compute mean (ms) | Source |
|---|---:|---|
| VAE encoder | 15.0506 | `trtexec` |
| Transformer | 172.558 | `trtexec` (10 s duration) |
| VAE decoder | 17.8829 | `trtexec` |

These runs are not a paired, controlled comparison against the native C++ app. Do not sum them and claim an end-to-end result.

A single initialized + LoRA-merged PyTorch transformer forward measured **131.0 ms** in an earlier run. This is **not** a fair speed-up comparison: warmup, sampling, clocks and power were not held constant. In the current measurements, TensorRT has **not** yet been demonstrated to be faster than PyTorch.

---

## End-to-end parity (one sample)

Official PyTorch inference with `--align_method=nofix` compared to native TensorRT output:

| Metric | Value |
|---|---:|
| Resolution | 512×512×3 |
| MAE | 0.135600 / 255 |
| RMSE | 0.556073 / 255 |
| PSNR | 53.2282 dB |
| Maximum absolute channel difference | 31 / 255 |

These are **agreement metrics between two generated images**, not PSNR against a high-resolution reference. A single sample does not establish generalization; test natural photographs and textures separately. See [validation.md](validation.md).

---

## New end-to-end parity (rebuilt engines, one image)

| Comparison | MAE / 255 | RMSE / 255 | PSNR | Max channel difference / 255 |
|---|---:|---:|---:|---:|
| Rebuilt TensorRT vs upstream PyTorch `nofix` | 0.131028 | 0.526667 | **53.7001 dB** | 28 |
| Rebuilt TensorRT vs previous C++ engine set | 0.091518 | 0.463239 | **54.8147 dB** | 19 |

The reference uses a single 512×512 RGB image generated from the same 128×128 input. These numbers measure implementation agreement, not image restoration quality.

---

## More rigorous benchmarking still needed

1. Record `nvpmodel -q`, GPU clocks and temperature; include `tegrastats` log.
2. Use ≥30 runs after warmup, report mean, median, p95, p99 with sample size.
3. Measure **host wall time** separately, including engine load, preprocessing and image encoding.
4. Record startup, memory high-water mark and sustained power/thermal conditions.
5. Compare multiple input photographs and structured textures at identical effective resolution.
6. Benchmark PyTorch baseline under the same power/clock/thermal state.
7. Record engine-build tactic warnings and full build memory usage for reproducibility.

---

## Vendored standalone export and runtime (2026-10-29, latest run)

This run used the architecture sources under `third_party/tinysr_python/`,
local checkpoints and embeddings under `TinySR-Jetson/`, newly exported ONNX
files, and three newly compiled TensorRT engines. No `--upstream` argument
was used. The original TinySR checkout was still present but its Python
model classes were not used. Engine builds returned exit status 0.

Native executable: 5 warmups + 30 measured inferences; output 512×512.

| Stage | Mean (ms) | p95 (ms) |
|---|---:|---:|
| H2D | 0.570 | 0.603 |
| VAE encoder | 15.236 | 15.637 |
| FP32 → FP16 | 0.009 | 0.009 |
| Transformer | 171.380 | 171.932 |
| Residual | 0.013 | 0.014 |
| VAE decoder | 17.626 | 17.673 |
| D2H | 0.834 | 1.026 |
| **GPU pipeline** | **205.668** | **206.430** |

This equates to approximately **4.86 GPU-pipeline executions/s**, excluding
engine loading, CPU preprocessing and PNG encoding. Free device-visible
memory after engine loading: **5341.96 MiB / 7607.38 MiB total**.

The transformer compiler skipped one memory-hungry tactic after a CUDA OOM
warning, but successfully serialized the engine; the subsequent numerical
validation and native C++ inference passed.

---

### Standalone parity against original PyTorch (`nofix`)

Single 512×512 image: **PSNR 53.2869 dB**, MAE **0.134584/255**,
RMSE **0.552322/255**, maximum channel error **31/255**. These are
implementation agreement metrics, not measures of super-resolution quality.
