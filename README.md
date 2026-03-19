# 7T MS Lesion Segmentation Models (UNETR + SegFormer3D)

This repository provides pretrained 3D WML lesion-segmentation models for 7t MRI scans and minimal scripts for loading checkpoints, running smoke tests, and applying the models directly to MRI volumes.

## Example Segmentations

![Comparison](figures/figure1.png)

*Comparison of lesion segmentation outputs on 7T FLAIR MRI across multiple methods, including UNETR and SegFormer3D (models provided in this repository). Manual annotations are shown as reference. The figure highlights qualitative differences in lesion detection, including missed lesions, false positives, and sensitivity to cortical boundaries. Adapted from manuscript (under review).*

## Contents
- `load_unetr.py`
- `load_segformer.py`
- `run_mri_inference.py`
- `requirements.txt`

## Weights

Model weights are hosted externally due to file size.

**Download (Google Drive):**  
`https://drive.google.com/file/d/1_GF_WZlDSQppsC42_nMFtWuPXzgB2tWa/view?usp=sharing`

The archive contains:

```text
UNETR_Models/
  05_05_05.ckpt
  10_10_10.ckpt
  15_15_20.ckpt

Segformer_Models/
  05_05_05.ckpt
  10_10_10.ckpt
  15_15_20.ckpt
```

These correspond to models trained at different resolutions:
- `05_05_05` → `0.5 × 0.5 × 0.5 mm³`
- `10_10_10` → `1.0 × 1.0 × 1.0 mm³`
- `15_15_20` → `1.5 × 1.5 × 2.0 mm³`

## Requirements

- Python 3.10
- MONAI 1.2.0
- PyTorch
- NumPy 1.26.x
- `einops`
- `nibabel`
- `scipy`

Install:

```bash
pip install -r requirements_final_pass.txt
```

MONAI 1.2.0 is not compatible with NumPy 2.x for this workflow, so NumPy is pinned below 2.0.

## Quick Start: MRI Inference

Example:

```bash
python run_mri_inference.py \
  --model unetr \
  --ckpt path/to/UNETR_Models/05_05_05.ckpt \
  --image input.nii.gz \
  --output output.nii.gz \
  --device cpu \
  --pixdim 0.5 0.5 0.5 \
  --a-min -175 \
  --a-max 25000
```

Key parameters:
- `--pixdim`: target voxel spacing
- `--a-min`, `--a-max`: input intensity range mapped to `[0, 1]`
- `--pad-size`: optional spatial padding; omit unless you specifically want it

Important:
- MRI intensity ranges are not standardized across scanners or preprocessing pipelines.
- Incorrect intensity scaling is one of the most common reasons for poor predictions.
- You may need to adjust `--pixdim`, `--a-min`, and `--a-max` for your data.
- If your image has already been resampled or normalized, use values that match the file as stored.

For example, for an already normalized image at stored 1 mm spacing:

```bash
python run_mri_inference.py \
  --model unetr \
  --ckpt path/to/UNETR_Models/10_10_10.ckpt \
  --image input.nii.gz \
  --output output.nii.gz \
  --device cpu \
  --pixdim 1 1 1 \
  --a-min 0 \
  --a-max 1
```

Or, to rescale automatically according to 1 and 99th intensity percentiles:

```bash
python run_mri_inference.py \
  --model unetr \
  --ckpt path/to/UNETR_Models/10_10_10.ckpt \
  --image input.nii.gz \
  --output output.nii.gz \
  --device cpu \
  --pixdim 1 1 1 \
  --auto-intensity-percentiles 1 99 \
```

## Sanity Check (Recommended)

Before running on real data, verify that the checkpoints load correctly.

UNETR:
```bash
python load_unetr.py --ckpt path/to.ckpt --device cpu
```

SegFormer:
```bash
python load_segformer.py \
  --ckpt path/to.ckpt \
  --segformer-file /path/to/segformer3d.py \
  --device cpu
```

Expected output includes:

```text
Loaded cleanly with strict=True ✅
Smoke test passed ✅
```

## UNETR

Resolved loading configuration:
- `img_size = (96, 96, 96)`
- `qkv_bias = False`
- `pos_embed = "perceptron"`
- checkpoint key prefix: `model.`

## SegFormer3D

Resolved loading configuration:
- `in_channels = 1`
- `num_classes = 2`
- recommended sliding-window `roi_size = (96, 96, 96)`
- checkpoint key prefix: `_model.`

`segformer3d.py` is **not included**.

Please obtain it from the upstream OSUPCVLab SegFormer3D repository:  
`https://github.com/OSUPCVLab/SegFormer3D/blob/main/architectures/segformer3d.py`

## Smoke Test on Random Input

Both loader scripts can run a random-input sliding-window smoke test.

```bash
python load_unetr.py --ckpt path/to.ckpt --random-shape 256 256 256
python load_segformer.py --ckpt path/to.ckpt --segformer-file segformer3d.py --random-shape 256 256 256
```

`--device auto` can fall back to CPU if CUDA is unavailable or incompatible. Very large random inputs can be slow or memory-intensive on CPU.

## Notes on Resolution

The released models operate on `96 × 96 × 96` patches/windows. Using this scale for inference is recommended for consistency with training.

## Citation

If you use these models, please cite the associated manuscript:

- *Automated Detection of Multiple Sclerosis Lesions on 7-tesla MRI Using U-net and Transformer-based Segmentation*

Current status: preprint (2026), under review at NeuroImage.

Suggested temporary citation until an arXiv version or final publication is available:

```text
Maynord M, Liu M, Fermüller C, Choi S, Zeng Y, Dahal S, Harrison DM.
Automated Detection of Multiple Sclerosis Lesions on 7-tesla MRI Using U-net and Transformer-based Segmentation.
Preprint, 2026.
```

## Practical Notes

- `strict=True` loading was used to confirm checkpoint-to-architecture compatibility exactly.
- The loader scripts are intended as minimal verification examples.
- The MRI inference script is provided for convenience and ease of testing on user data, but preprocessing parameters may need adjustment depending on the input images.

## Acknowledgements / Third-Party Code

The SegFormer release script depends on an external `segformer3d.py` implementation from OSUPCVLab. That file is not bundled here; please obtain it from the upstream repository and review its license/terms before redistribution or packaging.
