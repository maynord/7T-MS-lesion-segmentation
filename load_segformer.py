#!/usr/bin/env python3
"""Minimal SegFormer3D release loader + sliding-window smoke test.

Expected checkpoint compatibility:
- Python 3.10
- MONAI 1.2.0
- SegFormer3DBase(in_channels=1, num_classes=2)
- local segformer3d.py implementation matching the checkpoint
"""

import argparse
import importlib.util
from pathlib import Path
import sys

import torch
from monai.inferers import sliding_window_inference


def import_module_from_file(py_file: str):
    py_path = Path(py_file).resolve()
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    stripped = {}
    for k, v in state_dict.items():
        stripped[k[len("_model.") : ] if k.startswith("_model.") else k] = v
    return stripped


def build_model(segformer_file: str):
    module = import_module_from_file(segformer_file)
    return module.SegFormer3DBase(
        in_channels=1,
        sr_ratios=[4, 2, 1, 1],
        embed_dims=[32, 64, 160, 256],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        decoder_head_embedding_dim=256,
        num_classes=2,
        decoder_dropout=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.ckpt or .pt)")
    parser.add_argument("--segformer-file", required=True, help="Path to segformer3d.py")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for smoke test (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--random-shape",
        nargs=3,
        type=int,
        default=(400, 400, 400),
        metavar=("D", "H", "W"),
        help="Random input shape for sliding-window smoke test",
    )
    parser.add_argument(
        "--roi-size",
        nargs=3,
        type=int,
        default=(96, 96, 96),
        metavar=("D", "H", "W"),
        help="ROI size for sliding_window_inference",
    )
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--no-forward", action="store_true", help="Skip random-input smoke test")
    args = parser.parse_args()

    state_dict = load_checkpoint_state_dict(args.ckpt)
    model = build_model(args.segformer_file)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing:", missing)
    print("Unexpected:", unexpected)
    if missing or unexpected:
        raise RuntimeError("State dict mismatch: strict load would fail.")

    model.load_state_dict(state_dict, strict=True)
    print("Loaded cleanly with strict=True ✅")
    print("Model class:", model.__class__.__name__)
    print("Parameter count:", sum(p.numel() for p in model.parameters()))

    if args.no_forward:
        return 0

    device = torch.device(args.device)
    model = model.to(device).eval()
    x = torch.randn((1, 1, *tuple(args.random_shape)), device=device)
    with torch.inference_mode():
        y = sliding_window_inference(
            inputs=x,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
        )
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Smoke test passed ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
