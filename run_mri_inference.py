#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    SpatialPadd,
    Spacingd,
)


def load_python_module(module_name: str, file_path: str):
    file_path = str(Path(file_path).resolve())
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str) -> Dict:
    ckpt = torch.load(str(Path(path).resolve()), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


def build_unetr():
    from monai.networks.nets import UNETR

    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        conv_block=True,
        spatial_dims=3,
        qkv_bias=False,
        save_attn=False,
    )
    return model, "model."


def build_segformer(segformer_file: str):
    module = load_python_module("segformer3d_local", segformer_file)
    if not hasattr(module, "SegFormer3DBase"):
        raise AttributeError(f"SegFormer3DBase not found in {segformer_file}")
    SegFormer3DBase = module.SegFormer3DBase
    model = SegFormer3DBase(in_channels=1, num_classes=2)
    return model, "_model."


def get_preprocess_no_intensity(
    pixdim: Tuple[float, float, float],
    pad_size,
):
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
    ]
    if pad_size is not None:
        transforms.append(SpatialPadd(keys=["image"], spatial_size=tuple(pad_size)))
    return Compose(transforms)


def apply_intensity_scaling(image: torch.Tensor, a_min: float, a_max: float) -> torch.Tensor:
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
        raise ValueError(f"Invalid intensity scaling bounds: a_min={a_min}, a_max={a_max}")
    return torch.clamp((image - a_min) / (a_max - a_min), 0.0, 1.0).float()


def save_nifti(array: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(array, affine)
    nib.save(nii, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MRI inference on a .nii/.nii.gz volume.")
    parser.add_argument("--model", choices=["unetr", "segformer"], required=True)
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument("--image", required=True, help="Path to input .nii or .nii.gz")
    parser.add_argument("--output", required=True, help="Path to output segmentation .nii.gz")
    parser.add_argument("--segformer-file", default=None, help="Path to segformer3d.py (required for segformer)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--pixdim", type=float, nargs=3, default=(0.5, 0.5, 0.5), help="Target voxel spacing.")
    parser.add_argument(
        "--pad-size",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Optional spatial pad size. Omit to disable dataset-specific full-volume padding.",
    )
    parser.add_argument("--a-min", type=float, default=-175.0, help="Lower bound for intensity scaling.")
    parser.add_argument("--a-max", type=float, default=25000.0, help="Upper bound for intensity scaling.")
    parser.add_argument(
        "--auto-intensity-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Override a-min/a-max using image percentiles computed after loading/orientation/spacing/padding and before intensity scaling.",
    )
    parser.add_argument("--save-probs-npy", default=None, help="Optional path to save raw logits/probabilities as .npy")
    args = parser.parse_args()

    if args.model == "segformer" and not args.segformer_file:
        raise ValueError("--segformer-file is required for --model segformer")

    if args.model == "unetr":
        model, prefix = build_unetr()
    else:
        model, prefix = build_segformer(args.segformer_file)

    state_dict = load_checkpoint(args.ckpt)
    state_dict = strip_prefix_if_present(state_dict, prefix)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Missing:", missing)
        print("Unexpected:", unexpected)
        raise RuntimeError("Checkpoint did not load cleanly with strict=False.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Loaded model cleanly with strict=True ✅")
    print("Model class:", model.__class__.__name__)
    print("pixdim:", tuple(args.pixdim))
    print("pad_size:", None if args.pad_size is None else tuple(args.pad_size))
    if args.auto_intensity_percentiles is None:
        print("intensity scaling:", (args.a_min, args.a_max), "->", (0.0, 1.0))
    else:
        print("intensity scaling: auto from percentiles", tuple(args.auto_intensity_percentiles), "->", (0.0, 1.0))

    data = {"image": str(Path(args.image).resolve())}
    preprocess = get_preprocess_no_intensity(
        tuple(args.pixdim),
        None if args.pad_size is None else tuple(args.pad_size),
    )
    batch = preprocess(data)

    x = batch["image"].unsqueeze(0).float()
    affine = np.asarray(batch["image_meta_dict"]["affine"])

    a_min = args.a_min
    a_max = args.a_max
    if args.auto_intensity_percentiles is not None:
        low, high = args.auto_intensity_percentiles
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("--auto-intensity-percentiles must satisfy 0 <= LOW < HIGH <= 100")
        x_np = x.numpy()
        a_min = float(np.percentile(x_np, low))
        a_max = float(np.percentile(x_np, high))
        print(f"[auto] percentiles ({low}, {high}) -> a_min={a_min:.6g}, a_max={a_max:.6g}")
    else:
        print(f"[manual] a_min={a_min}, a_max={a_max}")

    x = apply_intensity_scaling(x, a_min, a_max)

    print("Preprocessed image shape:", tuple(x.shape))
    print("Using roi_size:", tuple(args.roi_size))

    requested_device = choose_device(args.device)
    try:
        device = requested_device
        model = model.to(device)
        x = x.to(device)

        with torch.no_grad():
            logits = sliding_window_inference(
                x,
                roi_size=tuple(args.roi_size),
                sw_batch_size=args.sw_batch_size,
                predictor=model,
            )
    except Exception as e:
        if requested_device.type == "cuda":
            print(f"CUDA inference failed ({e}). Retrying on CPU.")
            device = torch.device("cpu")
            model = model.to(device)
            x = x.cpu()
            with torch.no_grad():
                logits = sliding_window_inference(
                    x,
                    roi_size=tuple(args.roi_size),
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                )
        else:
            raise

    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    save_nifti(pred, affine, Path(args.output))
    print(f"Saved segmentation to: {args.output}")

    if args.save_probs_npy:
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        Path(args.save_probs_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_probs_npy, probs)
        print(f"Saved probabilities to: {args.save_probs_npy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())