# Model conversion & provenance

## Upstream and version

Original repository: https://github.com/Microtreei/TinySR
Model architecture source is vendored in this repository with full attribution;
checkpoint files and embeddings must be obtained from the original authors. The original repository is
licensed under Apache-2.0. Check checkpoint redistribution permissions
separately before distributing weights.

The Jetson development worktree in the first validation was:
`~/paper2jetson/TinySR`. The original development worktree supplied the first validated ONNX models. The **consolidated exporter in this repository has since reproduced all three ONNX models** on the same Jetson, and `scripts/build_engines.sh` has built three new engines under an isolated validation directory.

Expected exported ONNX files:

```text
models_onnx/vae_encoder_512_fp32.onnx
models_onnx/vae_decoder_512_fp32.onnx
models_onnx/tinysr_transformer_64_fp16.onnx
```

Expected compiled engines:

```text
models_trt/vae_encoder_512_fp16.engine
models_trt/vae_decoder_512_fp16.engine
models_trt/tinysr_transformer_64_fp16.engine
```

These exact input/output shape and dtype contracts are checked when loading
all engines, before inference begins.

## Transformer preparation

The transformer is special: upstream initializes and caches conditioning on
its first forward. For the exported model, LoRA adapters are first loaded,
the first forward initializes the 12 blocks, then `merge_and_unload()` folds
adapter weights into base layers. Export **after initialization and merge**.
The fixed timestep and pooled embeddings are therefore embodied in the
exported graph and are **not runtime inputs**.

Saved reference: `transformer_merged_golden.safetensors` containing
`latent_input` and `prediction_merged`, freshly emitted by `tools/export_models.py`.
The ONNX exported on Jetson contained 1,549 nodes, 223 initializers and a
fixed FP16 input/output of `1x16x64x64`; checker passed.

## Rebuilding the engines

On the Jetson, after exporting ONNX via upstream-adapted Python scripts:

```bash
./scripts/build_engines.sh models_onnx models_trt
```

The transformer build uses `--memPoolSize=workspace:512` and
`--builderOptimizationLevel=0` to limit tactic exploration. `workspace:512`
does **not** cap total builder memory. On R36.4.7 the transformer build
failed with `NvMapMemAllocInternalTagged` and CUDA OOM. After moving to
JetPack 6.2.3 / R36.5.2, the same build completed successfully on-device.

Numerical validation must be repeated after rebuilding or changing the engine.
Do not assume ONNX parser success guarantees numerical accuracy.

## Work remaining for reproducibility

- Test full setup on an **independent fresh clone** of TinySR-Jetson, without cloning upstream.
- Preserve exact upstream source commit and model checksums in a manifest.
- Validate a range of input photographs against original Python inference.
- Add tests for CUDA kernels and engine I/O on the Jetson CI runner.
- Implement exact Python interpolation if needed for improved parity.

## Standalone consolidated exporter

The runtime remains native C++/TensorRT. For **model conversion only**, the
repository vendors four original TinySR architecture modules under
`third_party/tinysr_python/` with upstream attribution, license and SHA-256
provenance. The exporter does not require a separate TinySR checkout.

A user needs the original pretrained checkpoints and two embedding tensors in
`checkpoint/` and `dataset/`, respectively. With the pinned Jetson-compatible
Python packages installed, run:

```bash
python tools/export_models.py \
  --checkpoint-root checkpoint \
  --assets-root dataset \
  --component all \
  --out models_onnx \
  --golden-dir logs/export_references
./scripts/build_engines.sh models_onnx models_trt
```

The exporter generates three static-shape ONNX graphs (opset 17) and a
transformer golden tensor. The transformer model's original positional
embedding loader uses a relative `dataset/cache/pos_embed.pt` path: the
standalone exporter provides a temporary isolated working directory only
during model construction.

### Isolated verification on a Jetson

```bash
export VALIDATION="$HOME/paper2jetson/tinysr-validation/standalone"
mkdir -p "$VALIDATION"/{onnx,engines,golden}
python tools/export_models.py \
  --checkpoint-root checkpoint \
  --assets-root dataset \
  --component all \
  --out "$VALIDATION/onnx" \
  --golden-dir "$VALIDATION/golden"
./scripts/build_engines.sh "$VALIDATION/onnx" "$VALIDATION/engines"
python tools/validate_transformer.py \
  --engine "$VALIDATION/engines/tinysr_transformer_64_fp16.engine" \
  --golden "$VALIDATION/golden/transformer_merged_golden.safetensors"
```

**Standalone verification (2026-10-29):** The vendored exporter produced
all three ONNX graphs from local checkpoint and embedding files on Jetson
Orin Nano 8GB / JetPack 6.2.3. The three engine builds passed. The
regenerated transformer passed numerical validation at **0.2202% relative
RMSE**; the full C++ image had **53.2869 dB PSNR** compared with the
original PyTorch `nofix` output (single image). This tests the code path
without `--upstream` and confirms local import paths, but it is not yet a
clean installation test on a separate Jetson.
