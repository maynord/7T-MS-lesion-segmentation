#!/usr/bin/env python3
"""Minimal UNETR release loader + sliding-window smoke test.

Expected checkpoint compatibility:
- Python 3.10
- MONAI 1.2.0
- UNETR(img_size=(96,96,96), qkv_bias=False, pos_embed="perceptron")
"""

import argparse
import sys
from typing import Sequence

import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR


def parse_3ints(values: Sequence[str]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected exactly 3 integers.")
    return tuple(int(v) for v in values)


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
        stripped[k[len("model.") : ] if k.startswith("model.") else k] = v
    return stripped


def build_model() -> UNETR:
    return UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        spatial_dims=3,
        qkv_bias=False,
        save_attn=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.ckpt or .pt)")
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
    model = build_model()

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
