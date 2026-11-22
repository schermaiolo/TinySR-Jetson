#!/usr/bin/env python3
"""Export pretrained TinySR checkpoints to static ONNX without an upstream clone.

Requires local, vendored Python model definitions and downloaded checkpoint
assets. Retest freshly generated engines against the golden tensors.
"""
import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import onnx
import torch
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoint"),
                        help="Directory containing official checkpoint/tinybackbone, tinysr and vae")
    parser.add_argument("--assets-root", type=Path, default=Path("dataset"),
                        help="Directory containing cache/pos_embed.pt and default/pool_embeds.pt")
    parser.add_argument("--out", type=Path, default=Path("models_onnx"))
    parser.add_argument("--component", choices=("encoder", "decoder", "transformer", "all"),
                        default="all")
    parser.add_argument("--reference-image", type=Path,
                        help="128x128 PNG for transformer initialization / golden data")
    parser.add_argument("--golden-dir", type=Path,
                        default=Path("logs/export_references"))
    return parser.parse_args()


@contextmanager
def model_working_directory(assets_root: Path):
    """Scope compatibility with upstream's relative dataset/cache lookup.

    Vendored TinySR PatchEmbed opens `dataset/cache/pos_embed.pt` from CWD.
    The exported model uses a local, self-contained asset directory.
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="tinysr-export-") as temporary:
        temp_root = Path(temporary)
        (temp_root / "dataset").symlink_to(assets_root, target_is_directory=True)
        previous = Path.cwd()
        try:
            os.chdir(temp_root)
            yield
        finally:
            os.chdir(previous)


def load_lora_state_dict(state_dict, model):
    """TinySR-specific parameter key mapping (ported from upstream utils/util.py)."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".default" not in name:
                continue
            key = "transformer." + name.replace(".default", "")
            if key not in state_dict:
                raise KeyError(f"Missing LoRA parameter {key}")
            parameter.copy_(state_dict.pop(key))
    if state_dict:
        raise ValueError(f"Unconsumed LoRA parameters: {list(state_dict)[:8]}")


def export(model, example, target, input_name, output_name):
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {target} ...", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            model, example, str(target), opset_version=17,
            input_names=[input_name], output_names=[output_name],
            do_constant_folding=False, dynamo=False,
        )
    onnx.checker.check_model(str(target))
    print(f"OK: {target} ({target.stat().st_size / 2**20:.2f} MiB)")


class Encoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.core = vae.encoder
        self.scale = float(vae.config.scaling_factor)

    def forward(self, rgb):
        return self.core(rgb) * self.scale


class Decoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.core = vae.decoder
        self.scale = float(vae.config.scaling_factor)

    def forward(self, latent):
        return self.core(latent / self.scale).clamp(-1, 1)


class Transformer(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, latent):
        return self.core(hidden_states=latent, return_dict=False)[0]


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.golden_dir = args.golden_dir.expanduser().resolve()
    if args.reference_image is not None:
        args.reference_image = args.reference_image.expanduser().resolve()
    vendor_root = root / "third_party/tinysr_python"
    expected = (
        vendor_root / "models/vae/autoencoder_tiny.py",
        vendor_root / "models/vae/tinyvae.py",
        vendor_root / "models/tinysr/tinysd3.py",
        vendor_root / "models/tinysr/tinysd3block.py",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise SystemExit("Vendored TinySR source missing. See docs/standalone.md:\n" + "\n".join(missing))
    if not checkpoint_root.is_dir():
        raise SystemExit(f"Checkpoint directory missing: {checkpoint_root}")
    sys.path.insert(0, str(vendor_root))

    from models.vae.autoencoder_tiny import AutoencoderTiny

    vae_path = checkpoint_root / "vae/separable"
    if args.component in ("encoder", "decoder", "all"):
        vae = AutoencoderTiny.from_pretrained(str(vae_path), torch_dtype=torch.float32).eval()
        if args.component in ("encoder", "all"):
            rgb = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
            export(Encoder(vae).eval(), rgb,
                   args.out / "vae_encoder_512_fp32.onnx", "rgb", "latent")
        if args.component in ("decoder", "all"):
            latent = torch.zeros((1, 16, 64, 64), dtype=torch.float32)
            export(Decoder(vae).eval(), latent,
                   args.out / "vae_decoder_512_fp32.onnx", "latent", "rgb")
        del vae

    if args.component not in ("transformer", "all"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Transformer export requires a CUDA-capable PyTorch build")

    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig
    from safetensors.torch import save_file
    from models.tinysr.tinysd3 import TinySD3Transformer2DModel
    import torch.nn.functional as F

    if args.reference_image is not None:
        with Image.open(args.reference_image) as img:
            if img.size != (128, 128):
                raise ValueError("Expected a 128x128 reference image")
            rgb = to_tensor(img.convert("RGB")).unsqueeze(0).to("cuda", dtype=torch.float16)
    else:
        # The reference latent is only used to initialize/capture the model
        # and establish a reproducible golden input, not as trained weights.
        rgb = torch.zeros((1, 3, 128, 128), device="cuda", dtype=torch.float16)
    rgb = F.interpolate(rgb, (512, 512), mode="bicubic", align_corners=False) * 2 - 1
    vae = AutoencoderTiny.from_pretrained(str(vae_path), torch_dtype=torch.float16).to("cuda").eval()
    with torch.inference_mode():
        latent = (vae.encode(rgb).latents * vae.config.scaling_factor).contiguous()
    del vae, rgb
    torch.cuda.empty_cache()

    pos_embed = assets_root / "cache/pos_embed.pt"
    if not pos_embed.is_file():
        raise FileNotFoundError(f"TinySR positional embedding asset missing: {pos_embed}")
    # The unchanged upstream PatchEmbed model reads a relative cache path.
    # Limit this compatibility workaround to transformer construction.
    with model_working_directory(assets_root):
        transformer = TinySD3Transformer2DModel.from_pretrained(
            str(checkpoint_root / "tinybackbone/prune-12-merge-tinysr"),
            subfolder="transformer", torch_dtype=torch.float16,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
        )
    lora = LoraConfig(
        r=64, lora_alpha=64, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear",
                        "linear_1", "linear_2", "net.2"],
    )
    transformer.add_adapter(lora)
    transformer.enable_adapters()
    state = StableDiffusion3Pipeline.lora_state_dict(
        str(checkpoint_root / "tinysr"), weight_name="transformer.safetensors"
    )
    load_lora_state_dict(state, transformer)
    transformer = transformer.to("cuda", dtype=torch.float16).eval()

    timestep = torch.tensor([1000.0], device="cuda", dtype=torch.float16)
    pooled = torch.load(assets_root / "default/pool_embeds.pt",
                        map_location="cpu", weights_only=True).to("cuda", dtype=torch.float16)
    with torch.inference_mode():
        before = transformer(hidden_states=latent, timestep=timestep,
                             pooled_projections=pooled, return_dict=False)[0]
        second = transformer(hidden_states=latent, timestep=timestep,
                              pooled_projections=pooled, return_dict=False)[0]
    assert transformer.initialized
    assert all(block.initialized for block in transformer.transformer_blocks)
    assert torch.equal(before, second), "Unexpected difference after initialization"
    transformer.merge_and_unload(safe_merge=False)
    assert not any(hasattr(module, "lora_A") for module in transformer.modules())

    wrapped = Transformer(transformer).eval()
    with torch.inference_mode():
        prediction = wrapped(latent)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Transformer output has non-finite values")
    print("Merge difference MAE:", (second.float() - prediction.float()).abs().mean().item())

    args.golden_dir.mkdir(parents=True, exist_ok=True)
    save_file({
        "latent_input": latent.float().cpu().contiguous(),
        "prediction_merged": prediction.float().cpu().contiguous(),
    }, str(args.golden_dir / "transformer_merged_golden.safetensors"))
    export(wrapped, latent, args.out / "tinysr_transformer_64_fp16.onnx",
           "latent_input", "prediction")


if __name__ == "__main__":
    main()
