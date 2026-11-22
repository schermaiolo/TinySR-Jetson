#!/usr/bin/env bash
# Build on the target Jetson. Never distribute the engine artifacts as portable binaries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="${1:-$ROOT/models_onnx}"
ENGINE_DIR="${2:-$ROOT/models_trt}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$TRTEXEC" ]]; then
  echo "trtexec not found or not executable: $TRTEXEC" >&2
  exit 1
fi
mkdir -p "$ENGINE_DIR"
for model in vae_encoder_512_fp32 vae_decoder_512_fp32 tinysr_transformer_64_fp16; do
  if [[ ! -f "$ONNX_DIR/$model.onnx" ]]; then
    echo "Missing ONNX: $ONNX_DIR/$model.onnx" >&2
    exit 1
  fi
done
"$TRTEXEC" \
  --onnx="$ONNX_DIR/vae_encoder_512_fp32.onnx" \
  --saveEngine="$ENGINE_DIR/vae_encoder_512_fp16.engine" \
  --fp16 --skipInference
"$TRTEXEC" \
  --onnx="$ONNX_DIR/vae_decoder_512_fp32.onnx" \
  --saveEngine="$ENGINE_DIR/vae_decoder_512_fp16.engine" \
  --fp16 --skipInference
"$TRTEXEC" \
  --onnx="$ONNX_DIR/tinysr_transformer_64_fp16.onnx" \
  --saveEngine="$ENGINE_DIR/tinysr_transformer_64_fp16.engine" \
  --fp16 --memPoolSize=workspace:512 --builderOptimizationLevel=0 \
  --skipInference
printf '\nTensorRT builds completed. Validate against the Python golden tensors.\n'
