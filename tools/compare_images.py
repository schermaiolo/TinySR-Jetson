#!/usr/bin/env python3
"""Compare two saved RGB outputs; image-space metrics are not a substitute for ground truth."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    with Image.open(args.reference) as image:
        a = np.asarray(image.convert("RGB"), dtype=np.float64)
    with Image.open(args.candidate) as image:
        b = np.asarray(image.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        parser.error(f"Different image dimensions: {a.shape} != {b.shape}")
    error = a - b
    mae = np.mean(np.abs(error))
    rmse = math.sqrt(np.mean(error**2))
    psnr = float("inf") if rmse == 0 else 20 * math.log10(255.0 / rmse)
    print(f"Shape: {a.shape}")
    print(f"MAE: {mae:.6f} / 255")
    print(f"RMSE: {rmse:.6f} / 255")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"Max channel error: {np.abs(error).max():.0f} / 255")
    print("Note: use original TinySR with --align_method=nofix for fairer comparison.")


if __name__ == "__main__":
    main()
