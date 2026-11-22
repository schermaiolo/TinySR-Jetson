# Installation, quick start and model conversion

This guide is for **TinySR-Jetson**, the native C++/CUDA/TensorRT port of [TinySR](https://github.com/Microtreei/TinySR). No clone of the original PyTorch repository is required: the Python model definitions needed for exporting are already included in `third_party/tinysr_python/`.

All commands below run from the **TinySR-Jetson repository root** on the Jetson, unless noted otherwise.

## 1. Build the C++ runtime

The tested platform is Jetson Orin Nano 8GB with JetPack 6.2.3, CUDA 12.6 and TensorRT 10.3.

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev libnvinfer-dev

# Once this repository has been published:
git clone https://github.com/schermaiolo/TinySR-Jetson.git
cd TinySR-Jetson

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The executable is `build/tinysr-jetson`. You still need to generate the TensorRT engines before running inference.

---

## 2. Download the original model files

Download the official backbone, VAE and LoRA weights from the [TinySR project](https://github.com/Microtreei/TinySR#-quick-start). Place them, together with the two required auxiliary `.pt` tensors, in this layout:

```text
TinySR-Jetson/
├── checkpoint/
│   ├── tinybackbone/prune-12-merge-tinysr/transformer/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   ├── tinysr/transformer.safetensors
│   └── vae/separable/
│       ├── config.json
│       └── diffusion_pytorch_model.safetensors
└── dataset/
    ├── cache/pos_embed.pt
    └── default/pool_embeds.pt
```

The `.pt` tensors are required in addition to the pretrained checkpoints. If your checkpoint download does not contain them, obtain the original TinySR assets separately; **do not substitute generated or random embeddings**. These data files are intentionally not included in Git.

---

## 3. Export PyTorch to ONNX

Create or activate a Python 3.10 environment with a **JetPack-compatible CUDA build of PyTorch**. The tested environment used PyTorch 2.8.0 and torchvision 0.23.0 from the Jetson AI Lab package index. Check CUDA first:

```bash
source /path/to/your/jetson-python-venv/bin/activate
python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available()'
```

The tested exporter packages were ONNX 1.17.0, diffusers 0.29.1, transformers 4.49.0, peft 0.15.0, safetensors and Pillow. Install compatible versions without replacing the working CUDA PyTorch build. **Do not install the original TinySR `requirements.txt` blindly**: its PyTorch pin is not appropriate for this Jetson environment.

```bash
python tools/export_models.py \
  --checkpoint-root checkpoint \
  --assets-root dataset \
  --component all \
  --out models_onnx \
  --golden-dir logs/export_references
```

The exporter initializes the transformer, merges its LoRA weights and writes three static-shape ONNX models (opset 17). It also saves a golden transformer input/output pair for validation. `--reference-image /path/to/image_128.png` is optional.

---

## 4. Build TensorRT engines on the Jetson

```bash
./scripts/build_engines.sh models_onnx models_trt
```

| Model | Generated engine |
|---|---|
| VAE encoder | `models_trt/vae_encoder_512_fp16.engine` |
| TinySR transformer | `models_trt/tinysr_transformer_64_fp16.engine` |
| VAE decoder | `models_trt/vae_decoder_512_fp16.engine` |

Build the engines on the target device. `--skipInference` is used during engine building, so successful compilation alone does not verify numerical correctness.

---

## 5. Run inference

Use an **RGB image of 128 × 128 pixels**. If you included the example images shown in the main README:

```bash
./build/tinysr-jetson \
  --engines models_trt \
  --input assets/examples/input.png \
  --output outputs/result.png \
  --warmup 5 \
  --runs 30
```

The output is 512 × 512. The C++ executable does not use Python at inference time. To list available options:

```bash
./build/tinysr-jetson --help
```

---

## 6. Verify the result

```bash
python tools/validate_transformer.py \
  --engine models_trt/tinysr_transformer_64_fp16.engine \
  --golden logs/export_references/transformer_merged_golden.safetensors

# If you also have an output from original TinySR with align_method=nofix:
python tools/compare_images.py /path/to/pytorch_nofix.png outputs/result.png

ctest --test-dir build --output-on-failure
```

The original Python implementation uses AdaIN by default; **compare with `--align_method=nofix`** because this C++ port does not yet implement AdaIN.

For the tested numbers and current limitations, see [benchmarks.md](benchmarks.md), [validation.md](validation.md) and [limitations.md](limitations.md).