# Model Checkpoints

This directory houses the trained PyTorch model weights across Vision Transformers (`ViT-Small`, `DeiT-Small`, `Swin-Tiny`) and Modern CNN (`ConvNeXt-Tiny`) under different fine-tuning regimes (`full_ft`, `unfreeze_last_3`, `unfreeze_last_2`, `linear_probe`, `scratch`).

Due to GitHub's file size limits (>100MB per file), binary `.pth` checkpoint files are excluded from git tracking.

## Training Checkpoints from Scratch
To train all checkpoints locally with seed 42:
```bash
bash run_all.sh
# or on Windows PowerShell:
.\run_all.ps1
```

## Downloading / Pushing Pretrained Weights
Pretrained checkpoints can be synced with Hugging Face Hub using the deployment utility:
```bash
python scripts/push_to_hf.py --token <HF_TOKEN> --model-repo <username>/vit-lulc-checkpoints --upload-weights
```
