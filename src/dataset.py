"""
src/dataset.py - Phase 1: EuroSAT RGB Data Pipeline
===================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024), EuroSAT Dataset (Helber et al., 2019)

Phase 1 Requirements:
- torchvision.datasets.EuroSAT(download=True) (RGB, 10 classes, 27,000 images)
- Stratified 70/15/15 train/val/test split with fixed seed (42)
- Zero-overlap assertions and persistence to data/splits.json
- Model-aware transforms: resize 224, ImageNet mean/std resolved from timm pretrained configs
- Train-only light augmentation: RandomHorizontalFlip, RandomVerticalFlip, RandomRot90
- Class counts per split printed and saved to results/phase1_class_counts.csv
- Sample image grid saved to figures/eurosat_samples_grid.png (DPI=200)
- GPU check & DataLoader throughput benchmark (images/sec) saved to results/phase1_data_metrics.json
- Callable factory: get_dataloaders(batch_size, seed, model_name)
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
import timm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg


class RandomRot90:
    """
    Random 90-degree rotations (0, 90, 180, 270 degrees).
    Standard and domain-appropriate for nadir/orthorectified satellite remote sensing.
    """
    def __init__(self, angles: Tuple[int, ...] = (0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice(self.angles)
        if angle == 0:
            return img
        return img.rotate(angle)


_IMAGE_CACHE: Dict[str, Image.Image] = {}

class EuroSATWrapper(Dataset):
    """
    Subclass/wrapper over torchvision.datasets.EuroSAT or Subset
    to apply split-specific transforms and expose file paths for attribution.
    Includes in-memory PIL caching to eliminate repeated disk I/O bottlenecks.
    """
    def __init__(self, base_dataset: Dataset, indices: List[int], transform=None, cache: bool = True):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.cache = cache

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        real_idx = self.indices[idx]
        # base_dataset is a torchvision EuroSAT (ImageFolder subclass)
        path, label = self.base_dataset.samples[real_idx]
        if self.cache:
            img = _IMAGE_CACHE.get(path)
            if img is None:
                img = Image.open(path).convert("RGB")
                _IMAGE_CACHE[path] = img
        else:
            img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


def get_timm_mean_std(model_name: Optional[str] = None) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Resolves exact normalization mean and standard deviation from timm pretrained config.
    """
    if model_name:
        try:
            model_cfg = timm.data.resolve_model_data_config(model_name)
            mean = tuple(model_cfg.get("mean", cfg.DEFAULT_MEAN))
            std = tuple(model_cfg.get("std", cfg.DEFAULT_STD))
            return mean, std
        except Exception as e:
            print(f"[Dataset] Notice: Could not resolve timm config for '{model_name}' ({e}). Falling back to default.")
    return tuple(cfg.DEFAULT_MEAN), tuple(cfg.DEFAULT_STD)


def get_transforms(
    model_name: Optional[str] = None,
    image_size: int = cfg.IMAGE_SIZE
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    Constructs model-aware transforms for:
      1. Train: Resize to 224, RandomHorizontalFlip, RandomVerticalFlip, RandomRot90, ToTensor, Normalize
      2. Val / Test: Resize to 224, ToTensor, Normalize
      3. Raw: Resize to 224, ToTensor (unnormalized for visualization / XAI)
    """
    mean, std = get_timm_mean_std(model_name)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRot90(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    raw_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform, raw_transform


def load_raw_eurosat(data_dir: Path = cfg.DATA_DIR) -> datasets.EuroSAT:
    """
    Loads EuroSAT RGB via torchvision.datasets.EuroSAT.
    Checks existing local directories first to avoid slow network timeouts.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(data_dir.glob("**/2750"))
    if candidates and len(list(candidates[0].iterdir())) == 10:
        target_dir = candidates[0].parent
        return datasets.EuroSAT(root=str(target_dir.parent), download=False)

    try:
        return datasets.EuroSAT(root=str(data_dir), download=False)
    except Exception:
        return datasets.EuroSAT(root=str(data_dir), download=True)


def create_or_load_stratified_split(
    dataset: datasets.EuroSAT,
    seed: int = cfg.PRIMARY_SEED,
    train_ratio: float = cfg.TRAIN_SPLIT,
    val_ratio: float = cfg.VAL_SPLIT,
    test_ratio: float = cfg.TEST_SPLIT,
    splits_path: Path = cfg.SPLITS_PATH,
) -> Dict[str, List[int]]:
    """
    Creates ONE stratified 70/15/15 train/val/test split with a fixed seed.
    Saves to data/splits.json and strictly asserts zero overlap.
    """
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    # If splits.json exists and matches dataset length, reload it
    if splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            data_splits = json.load(f)
        train_idx = data_splits.get("train", [])
        val_idx = data_splits.get("val", [])
        test_idx = data_splits.get("test", [])

        if len(train_idx) + len(val_idx) + len(test_idx) == len(dataset):
            print(f"[Dataset] Loaded existing verified split from {splits_path}")
            # Run assertions on loaded split
            _assert_zero_overlap(train_idx, val_idx, test_idx, len(dataset))
            return {"train": train_idx, "val": val_idx, "test": test_idx}

    targets = np.array(dataset.targets)
    n_samples = len(targets)

    # 1. Split off test set (15% = 4,050 samples)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(n_samples), targets))

    # 2. Split remaining (85%) into train (70%) and val (15%)
    val_relative_size = val_ratio / (train_ratio + val_ratio)  # 0.15 / 0.85 ≈ 0.17647
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=seed)
    train_sub_idx, val_sub_idx = next(sss_val.split(train_val_idx, targets[train_val_idx]))

    train_idx = train_val_idx[train_sub_idx].tolist()
    val_idx = train_val_idx[val_sub_idx].tolist()
    test_idx = test_idx.tolist()

    # Zero overlap assertion
    _assert_zero_overlap(train_idx, val_idx, test_idx, n_samples)

    splits_data = {
        "metadata": {
            "dataset": "EuroSAT_RGB",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_images": n_samples,
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "test_count": len(test_idx),
        },
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    print(f"[Dataset] Generated 70/15/15 stratified split saved to {splits_path}")
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _assert_zero_overlap(train_idx: List[int], val_idx: List[int], test_idx: List[int], total: int):
    """Strictly asserts disjoint index sets and completeness."""
    s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)

    overlap_train_val = s_train & s_val
    overlap_train_test = s_train & s_test
    overlap_val_test = s_val & s_test

    assert len(overlap_train_val) == 0, f"ASSERTION FAILED: Train and Val overlap on {len(overlap_train_val)} indices!"
    assert len(overlap_train_test) == 0, f"ASSERTION FAILED: Train and Test overlap on {len(overlap_train_test)} indices!"
    assert len(overlap_val_test) == 0, f"ASSERTION FAILED: Val and Test overlap on {len(overlap_val_test)} indices!"
    assert len(s_train) + len(s_val) + len(s_test) == total, (
        f"ASSERTION FAILED: Combined split count ({len(s_train) + len(s_val) + len(s_test)}) != Total ({total})"
    )
    print(f"[Dataset] Assertion PASSED: Zero overlap verified across Train ({len(s_train)}), Val ({len(s_val)}), and Test ({len(s_test)}).")


def get_dataloaders(
    batch_size: int = cfg.BATCH_SIZE,
    seed: int = cfg.PRIMARY_SEED,
    model_name: Optional[str] = None,
    num_workers: int = cfg.NUM_WORKERS,
    data_dir: Path = cfg.DATA_DIR,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Creates and returns PyTorch DataLoaders for train, val, and test partitions
    configured with model-specific normalization parameters from timm.
    """
    raw_dataset = load_raw_eurosat(data_dir=data_dir)
    splits = create_or_load_stratified_split(raw_dataset, seed=seed)

    train_tf, eval_tf, _ = get_transforms(model_name=model_name, image_size=cfg.IMAGE_SIZE)

    train_dataset = EuroSATWrapper(raw_dataset, splits["train"], transform=train_tf)
    val_dataset = EuroSATWrapper(raw_dataset, splits["val"], transform=eval_tf)
    test_dataset = EuroSATWrapper(raw_dataset, splits["test"], transform=eval_tf)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda and cfg.PIN_MEMORY,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda and cfg.PIN_MEMORY,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda and cfg.PIN_MEMORY,
        persistent_workers=num_workers > 0,
    )

    metadata = {
        "classes": raw_dataset.classes,
        "class_to_idx": raw_dataset.class_to_idx,
        "total_images": len(raw_dataset),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "test_count": len(test_dataset),
        "seed": seed,
        "batch_size": batch_size,
        "model_name": model_name,
    }

    return train_loader, val_loader, test_loader, metadata


def compute_and_print_class_counts(dataset: datasets.EuroSAT, splits: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Computes and formats class counts across Train, Val, and Test splits.
    Saves to results/phase1_class_counts.csv.
    """
    classes = dataset.classes
    targets = np.array(dataset.targets)

    train_idx = np.array(splits["train"])
    val_idx = np.array(splits["val"])
    test_idx = np.array(splits["test"])

    records = []
    print("\n" + "=" * 80)
    print(f"{'Class ID':<10}{'Class Name':<24}{'Total':<10}{'Train (70%)':<14}{'Val (15%)':<14}{'Test (15%)':<12}")
    print("=" * 80)

    for cid, cname in enumerate(classes):
        c_tot = int((targets == cid).sum())
        c_tr = int((targets[train_idx] == cid).sum())
        c_v = int((targets[val_idx] == cid).sum())
        c_te = int((targets[test_idx] == cid).sum())
        records.append({
            "class_id": cid,
            "class_name": cname,
            "total_count": c_tot,
            "train_count": c_tr,
            "val_count": c_v,
            "test_count": c_te,
            "train_pct": round(c_tr / c_tot * 100, 2),
            "val_pct": round(c_v / c_tot * 100, 2),
            "test_pct": round(c_te / c_tot * 100, 2),
        })
        print(f"{cid:<10}{cname:<24}{c_tot:<10}{c_tr:<14}{c_v:<14}{c_te:<12}")

    print("-" * 80)
    print(f"{'TOTAL':<34}{len(targets):<10}{len(train_idx):<14}{len(val_idx):<14}{len(test_idx):<12}")
    print("=" * 80)

    df = pd.DataFrame(records)
    csv_path = cfg.RESULTS_DIR / "phase1_class_counts.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Dataset] Class counts exported to {csv_path}")
    return df


def plot_sample_grid(dataset: datasets.EuroSAT, output_path: Path = cfg.FIGURES_DIR / "eurosat_samples_grid.png"):
    """
    Visualizes a clean 2x5 grid showing one representative sample from each of the 10 LULC classes.
    Saved to figures/eurosat_samples_grid.png at DPI=200.
    """
    classes = dataset.classes
    targets = np.array(dataset.targets)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    for cid, cname in enumerate(classes):
        idxs = np.where(targets == cid)[0]
        sample_path, _ = dataset.samples[idxs[0]]
        img = Image.open(sample_path).convert("RGB")

        axes[cid].imshow(img)
        axes[cid].set_title(f"Class {cid}: {cname}", fontsize=11, fontweight="bold", pad=8)
        axes[cid].axis("off")

    plt.suptitle("EuroSAT-RGB Land Use / Land Cover (LULC) - 10 Sentinel-2 Benchmark Classes",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[Dataset] Sample image grid saved to {output_path} (DPI={cfg.DPI})")


def benchmark_dataloader_throughput(
    train_loader: DataLoader,
    num_batches: int = 60,
    warmup_batches: int = 10,
) -> Dict[str, Any]:
    """
    Measures the images/sec throughput of the training DataLoader,
    transferring tensors to the target hardware device (GPU if available).
    """
    device = torch.device(cfg.DEVICE)
    print(f"\n[Benchmark] Measuring DataLoader throughput on device '{device}'...")

    total_images = 0
    start_time = None

    for b_idx, (images, targets, _) in enumerate(train_loader):
        # Transfer batch to device to benchmark end-to-end data pipeline delivery
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if b_idx == warmup_batches:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            total_images = 0

        if b_idx >= warmup_batches:
            total_images += images.size(0)

        if b_idx >= warmup_batches + num_batches:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    fps = total_images / elapsed

    bench_results = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "batches_timed": num_batches,
        "images_timed": total_images,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_images_per_sec": round(fps, 2),
    }

    print(f"[Benchmark] Benchmark Complete: {bench_results['throughput_images_per_sec']} images/sec "
          f"({bench_results['batches_timed']} batches x {train_loader.batch_size} on {bench_results['device_name']})")
    return bench_results


def run_phase1():
    """Executes the full Phase 1 verification workflow."""
    print("=" * 80)
    print("PHASE 1: EuroSAT RGB Data Pipeline, Stratified 70/15/15 Splitting & Benchmarking")
    print("=" * 80)

    # 1. Hardware Check
    print("\n--- 1. Hardware & GPU Availability Check ---")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:    {cuda_avail}")
    if cuda_avail:
        print(f"Device Name:       {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability:   {torch.cuda.get_device_capability(0)}")
        print(f"Total VRAM:        {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Mixed Precision:   Supported (torch.autocast)")
    else:
        print("Device Name:       CPU (No CUDA device found)")

    # 2. Dataset Load & Stratified Split
    print("\n--- 2. Dataset Loading & 70/15/15 Stratified Split ---")
    raw_dataset = load_raw_eurosat(cfg.DATA_DIR)
    print(f"EuroSAT Dataset initialized with {len(raw_dataset)} images across {len(raw_dataset.classes)} classes.")

    splits = create_or_load_stratified_split(
        raw_dataset,
        seed=cfg.PRIMARY_SEED,
        train_ratio=cfg.TRAIN_SPLIT,
        val_ratio=cfg.VAL_SPLIT,
        test_ratio=cfg.TEST_SPLIT,
    )

    # 3. Class Counts & Overlap Assertion
    print("\n--- 3. Class Distribution Across Splits ---")
    df_counts = compute_and_print_class_counts(raw_dataset, splits)

    # 4. Generate Sample Image Grid
    print("\n--- 4. Generating Sample Image Grid ---")
    plot_sample_grid(raw_dataset, cfg.FIGURES_DIR / "eurosat_samples_grid.png")

    # 5. Model-aware Transforms & DataLoader Throughput
    print("\n--- 5. DataLoader Creation & Throughput Benchmark ---")
    default_model = cfg.get_model_name("vit")
    mean, std = get_timm_mean_std(default_model)
    print(f"Default ViT Model Backbone: {default_model}")
    print(f"timm Resolved Normalization: Mean={mean}, Std={std}")

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.PRIMARY_SEED,
        model_name=default_model,
        num_workers=cfg.NUM_WORKERS,
    )

    bench = benchmark_dataloader_throughput(train_loader, num_batches=60, warmup_batches=10)

    # 6. Save Complete Phase 1 Metrics to JSON
    phase1_summary = {
        "phase": 1,
        "status": "COMPLETED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": {
            "cuda_available": cuda_avail,
            "device": cfg.DEVICE,
            "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else None,
        },
        "dataset": {
            "name": "EuroSAT_RGB",
            "total_images": len(raw_dataset),
            "classes": raw_dataset.classes,
            "splits": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
            },
            "split_ratios": {"train": cfg.TRAIN_SPLIT, "val": cfg.VAL_SPLIT, "test": cfg.TEST_SPLIT},
            "seed": cfg.PRIMARY_SEED,
            "zero_overlap_verified": True,
            "splits_json_path": str(cfg.SPLITS_PATH),
        },
        "transforms": {
            "image_size": cfg.IMAGE_SIZE,
            "mean": list(mean),
            "std": list(std),
            "train_augmentations": ["Resize(224)", "RandomHorizontalFlip(0.5)", "RandomVerticalFlip(0.5)", "RandomRot90", "ToTensor", "Normalize"],
            "eval_augmentations": ["Resize(224)", "ToTensor", "Normalize"],
        },
        "benchmark": bench,
    }

    metrics_path = cfg.RESULTS_DIR / "phase1_data_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(phase1_summary, f, indent=2)
    print(f"\n[Phase 1] Complete metrics saved to {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_phase1()
