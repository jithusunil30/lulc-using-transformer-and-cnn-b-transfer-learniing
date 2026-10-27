"""
scripts/push_to_hf.py - Hugging Face Hub & Spaces Deployment Utility
===================================================================
Uploads trained PyTorch model checkpoints to a Hugging Face Model Repository
and deploys the interactive Gradio demo application to a Hugging Face Space.

Usage:
------
python scripts/push_to_hf.py \
    --token <HF_TOKEN> \
    --model-repo <username>/vit-lulc-checkpoints \
    --space-repo <username>/vit-lulc-demo \
    --upload-weights \
    --upload-app
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("[HF Deploy] Notice: huggingface_hub not installed. Run 'pip install huggingface-hub'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload checkpoints and deploy Gradio app to Hugging Face Hub")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", None),
                        help="Hugging Face API write token (or set HF_TOKEN environment variable)")
    parser.add_argument("--model-repo", type=str, default=None,
                        help="HF Model Repo ID (e.g., 'username/vit-lulc-checkpoints')")
    parser.add_argument("--space-repo", type=str, default=None,
                        help="HF Space Repo ID (e.g., 'username/vit-lulc-demo')")
    parser.add_argument("--upload-weights", action="store_true", default=True,
                        help="Upload checkpoints/ directory to the model repository")
    parser.add_argument("--upload-app", action="store_true", default=True,
                        help="Upload app/, assets/, configs/, results/, src/ to the Space")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform a dry run without uploading")
    return parser.parse_args()


def upload_model_checkpoints(api: "HfApi", repo_id: str, token: str, dry_run: bool = False):
    """Uploads trained model checkpoints to a Hugging Face Model repo."""
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    if not ckpt_dir.exists():
        print(f"[HF Deploy] Warning: Checkpoints directory '{ckpt_dir}' not found. Skipping weights upload.")
        return

    ckpt_files = list(ckpt_dir.glob("*.pth"))
    print(f"[HF Deploy] Discovered {len(ckpt_files)} checkpoint files in {ckpt_dir}:")
    for f in ckpt_files:
        print(f"  - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    if dry_run:
        print(f"[HF Deploy] [DRY RUN] Would create/verify model repo: {repo_id}")
        print(f"[HF Deploy] [DRY RUN] Would upload {len(ckpt_files)} files to {repo_id}")
        return

    print(f"[HF Deploy] Creating/verifying model repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)

    print(f"[HF Deploy] Uploading folder {ckpt_dir} -> {repo_id}...")
    api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Add trained EuroSAT ViT/DeiT/Swin/ConvNeXt checkpoints"
    )
    print(f"[HF Deploy] Successfully uploaded checkpoints to: https://huggingface.co/{repo_id}")


def upload_space_application(api: "HfApi", space_id: str, token: str, dry_run: bool = False):
    """Deploys the Gradio application to a Hugging Face Space."""
    items_to_upload = [
        "app",
        "assets",
        "configs",
        "results",
        "src",
        "requirements.txt",
        "README.md"
    ]

    print(f"[HF Deploy] Preparing app components for Space '{space_id}':")
    for item in items_to_upload:
        p = PROJECT_ROOT / item
        if p.exists():
            print(f"  + {item} ({'DIR' if p.is_dir() else 'FILE'})")
        else:
            print(f"  - {item} (NOT FOUND, skipping)")

    if dry_run:
        print(f"[HF Deploy] [DRY RUN] Would create/verify space: {space_id} (SDK: gradio)")
        print(f"[HF Deploy] [DRY RUN] Would upload application files to {space_id}")
        return

    print(f"[HF Deploy] Creating/verifying Space repo: {space_id}")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        token=token,
        exist_ok=True
    )

    for item in items_to_upload:
        path = PROJECT_ROOT / item
        if not path.exists():
            continue

        if path.is_file():
            print(f"[HF Deploy] Uploading file: {item}...")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=item,
                repo_id=space_id,
                repo_type="space",
                token=token,
                commit_message=f"Upload {item}"
            )
        elif path.is_dir():
            print(f"[HF Deploy] Uploading directory: {item}/...")
            api.upload_folder(
                folder_path=str(path),
                path_in_repo=item,
                repo_id=space_id,
                repo_type="space",
                token=token,
                commit_message=f"Sync {item}/"
            )

    print(f"[HF Deploy] Successfully deployed Space application to: https://huggingface.co/spaces/{space_id}")


def main():
    args = parse_args()

    if not args.dry_run and not args.token:
        print("[HF Deploy] Error: Hugging Face API token is required. Pass --token <TOKEN> or set HF_TOKEN environment variable.")
        sys.exit(1)

    api = HfApi()

    if args.upload_weights:
        if args.model_repo:
            upload_model_checkpoints(api, args.model_repo, args.token, dry_run=args.dry_run)
        else:
            print("[HF Deploy] Notice: --model-repo not provided. Skipping weights upload.")

    if args.upload_app:
        if args.space_repo:
            upload_space_application(api, args.space_repo, args.token, dry_run=args.dry_run)
        else:
            print("[HF Deploy] Notice: --space-repo not provided. Skipping Space deployment.")

    print("\n[HF Deploy] Deployment utility finished.")


if __name__ == "__main__":
    main()
