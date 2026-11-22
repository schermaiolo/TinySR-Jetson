# Numerical validation log — TinySR-Jetson

These values were obtained on a **Jetson Orin Nano 8GB**, JetPack 6.2.3 / CUDA 12.6 / TensorRT 10.3. The initial VAE numerical checks came from the development worktree; the **rebuilt transformer** was validated using `tools/validate_transformer.py` in TinySR-Jetson. The checks were run locally, not in CI.

## Per-component checks

| Component | Reference | Relative RMSE | Max absolute error | Within `rtol=0.01, atol=0.01` | Finite values |
|---|---|---:|---:|---:|---|
| VAE encoder | PyTorch FP32 | 0.188% | 0.007970 | 100.00% | Yes |
| VAE decoder | PyTorch FP32 | 0.551% | 0.080980 | 99.49% | Yes |
| Transformer | PyTorch FP16, initialized + LoRA merged | 0.220% | 0.042664 | 99.88% | Yes |

All are **initial checks on a single representative tensor/image**. The merged-LoRA transformer also agreed with the unmerged reference with relative RMSE ~0.115%. Its first and second initialized forwards were bit-identical on the test input (maximum difference 0.0).

## Final saved-image comparison

Official TinySR PyTorch with `--align_method=nofix` was compared to the modular native TensorRT application output generated from the same 128×128 image.

```text
Shape: (512, 512, 3)
MAE: 0.135600 / 255
RMSE: 0.556073 / 255
PSNR: 53.2282 dB
Max channel error: 31 / 255
```

To reproduce the comparison after generating both images:

```bash
python3 tools/compare_images.py \
  ../TinySR/outputs/p2j_python_nofix/test.png \
  outputs/test.png
```

High PSNR between implementations validates **parity**, not perceptual quality compared to ground truth. The strongest next test is an image set containing natural photographs, faces/textures, low-quality camera inputs and repeated runs at matched shape/dtype.

## Freshly rebuilt transformer vs export-time golden (2026-10-29)

```text
Shape: (1, 16, 64, 64)
MAE: 0.000910380
RMSE: 0.001731097
Relative RMSE: 0.002202044 (0.2202%)
Max absolute error: 0.042663574
P99 absolute error: 0.006835938
Within atol=rtol=0.01: 99.8810%
All finite: True
RESULT: PASS
```

The script checks relative RMSE ≤ 1% and at least 99% elementwise closeness, and requires finite outputs. The checks are regression criteria for this fixed shape and model configuration, not generalized accuracy guarantees. Example invocation:

```bash
python tools/validate_transformer.py \
  --engine "$HOME/paper2jetson/tinysr-validation/engines/tinysr_transformer_64_fp16.engine" \
  --golden "$HOME/paper2jetson/tinysr-validation/golden/transformer_merged_golden.safetensors"
```

## Newly rebuilt full-pipeline parity (2026-10-29)

The rebuilt engines generated a 512×512 PNG. Compared to the original PyTorch `nofix` image, the **PSNR was 53.7001 dB** (MAE 0.131028 / 255, RMSE 0.526667 / 255, maximum channel error 28 / 255). Compared to the previously validated C++ image, the rebuilt version reached **54.8147 dB PSNR**. All comparisons use only one input image.

## Known differences and limitations

- Native pre-processing uses OpenCV bicubic; PyTorch's interpolate implementation can produce different pixels.
- The current pipeline does **not** include AdaIN and must be compared to upstream `nofix`, not its default color-aligned output.
- Some TensorRT operations run in FP16, while VAE engine boundaries use FP32.
- One output has isolated errors up to 31/255 per channel; consider adding a difference heatmap.
- Dataset and high-resolution ground truth have not yet been evaluated; no SR-quality claim is made.
- `tools/export_models.py` has completed all three ONNX exports and new engine builds on the existing Jetson upstream workspace, but independent clean-checkout reproducibility is not yet tested.

## Vendored standalone export and engine validation (2026-10-29)

The exporter used the four model definition files under
`third_party/tinysr_python/` and local checkpoint/embedding data. All three
new ONNX models were generated and compiled on-device with TensorRT 10.3.
The newly built transformer was tested against the golden tensor generated
by **the same export run**:

```text
MAE: 0.000910380
RMSE: 0.001731097
Relative RMSE: 0.002202044 (0.2202%)
Max absolute error: 0.042663574
P99 absolute error: 0.006835938
Within atol=rtol=0.01: 99.8810%
All finite: True
RESULT: PASS
```

Native C++ output compared with the original PyTorch `nofix` PNG for the
same input: **53.2869 dB PSNR**, MAE **0.134584/255**, RMSE
**0.552322/255**, maximum channel error **31/255**.

A separate transformer-export test was also completed while the original
checkout directory `../TinySR` had been temporarily renamed. The exporter
produced a valid ONNX file (652.24 MiB) without access to the checkout at that
path and restored the original directory afterward. Its reported merge
comparison MAE (0.000310677) was obtained with a different reference input
configuration, so it must not be compared directly with the earlier MAE.

This proves independence from that checkout path for the tested export, but
is **not** a clean install on an independent Jetson.
