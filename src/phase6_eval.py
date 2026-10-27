"""
src/phase6_eval.py - Phase 6: Corruption Robustness, Feature-Space t-SNE & Critical Analysis
============================================================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024), Figure 9 & Out-of-Distribution Robustness

Core Phase 6 Experiments:
(a) Corruption Robustness:
    - 4 corruptions: Gaussian noise, Gaussian blur, Brightness shift, JPEG compression
    - 3 severities (1 = mild, 2 = moderate, 3 = severe) + clean baseline (0)
    - Models: scratch vs full_ft vs best-partial (unfreeze_last_3) across ViT, DeiT, Swin
    - Outputs: results/corruption_robustness.csv, figures/corruption_robustness.png
(b) Feature-Space t-SNE (Reproduction of Khan et al. Figure 9):
    - Penultimate feature extraction across Scratch vs Pretrained (Full-FT & UnfreezeLast3)
    - Quantitative cluster separation via Silhouette Score (sklearn.metrics.silhouette_score)
    - 2D t-SNE manifold projection colored by the 10 EuroSAT LULC classes
    - Outputs: results/tsne_metrics.csv, figures/tsne_feature_space.png
"""

import sys
import os
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import load_raw_eurosat, get_transforms, get_timm_mean_std
from src.models import build_transformer_model
from src.train import clean_name


# ==============================================================================
# 1. Corruption Engines (Gaussian Noise, Blur, Brightness, JPEG Compression)
# ==============================================================================

def apply_image_corruption(
    pil_img: Image.Image,
    corruption_type: str,
    severity: int,
) -> Image.Image:
    """
    Applies synthetic corruption at severity level in {1, 2, 3}.
    Severity 0 returns the unaltered PIL image.
    """
    if severity == 0 or corruption_type == "clean":
        return pil_img

    if corruption_type == "gaussian_noise":
        # Additive Gaussian noise on RGB values in [0, 255]
        sigmas = {1: 15.0, 2: 35.0, 3: 55.0}
        sigma = sigmas[severity]
        arr = np.array(pil_img, dtype=np.float32)
        noise = np.random.normal(0.0, sigma, arr.shape)
        corrupted = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(corrupted)

    elif corruption_type == "gaussian_blur":
        # Gaussian blur with increasing radius
        radii = {1: 1.2, 2: 2.5, 3: 4.5}
        radius = radii[severity]
        return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

    elif corruption_type == "brightness_shift":
        # Multiplicative brightness factor (satellite over-exposure / solar incidence angle)
        factors = {1: 1.35, 2: 1.65, 3: 2.00}
        factor = factors[severity]
        enhancer = ImageEnhance.Brightness(pil_img)
        return enhancer.enhance(factor)

    elif corruption_type == "jpeg_compression":
        # Lossy JPEG compression artifacts via in-memory buffer
        qualities = {1: 35, 2: 18, 3: 8}
        quality = qualities[severity]
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


class CorruptedEuroSATDataset(Dataset):
    """
    Dataset wrapper that injects synthetic corruptions on the fly before model normalization.
    """
    def __init__(
        self,
        raw_dataset,
        indices: List[int],
        corruption_type: str,
        severity: int,
        model_name: str,
    ):
        self.raw_dataset = raw_dataset
        self.indices = indices
        self.corruption_type = corruption_type
        self.severity = severity

        # Model-specific ImageNet mean/std normalization
        mean, std = get_timm_mean_std(model_name)
        self.to_tensor_and_norm = transforms.Compose([
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        path, label = self.raw_dataset.samples[real_idx]
        img = Image.open(path).convert("RGB")

        # Apply synthetic corruption
        corrupted_img = apply_image_corruption(img, self.corruption_type, self.severity)

        # Apply spatial resize & normalization
        tensor_img = self.to_tensor_and_norm(corrupted_img)
        return tensor_img, int(label)


class RawEuroSATDataset(Dataset):
    """Yields resized [0, 1] RGB tensors before model-specific normalization."""
    def __init__(self, raw_dataset, indices: List[int]):
        self.raw_dataset = raw_dataset
        self.indices = indices
        self.to_tensor = transforms.Compose([
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        path, label = self.raw_dataset.samples[real_idx]
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img), int(label)


# ==============================================================================
# 2. Corruption Robustness Evaluator
# ==============================================================================

def evaluate_corruption_robustness(
    model: nn.Module,
    test_indices: List[int],
    raw_dataset,
    model_name: str,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluates model accuracy across all 4 corruptions and 3 severities plus clean baseline.
    Vectorizes Gaussian noise, blur, and brightness on GPU for maximum throughput.
    """
    model.eval()
    corruptions = ["gaussian_noise", "gaussian_blur", "brightness_shift", "jpeg_compression"]
    severities = [1, 2, 3]

    results: Dict[str, Dict[int, float]] = {c: {} for c in corruptions}

    mean, std = get_timm_mean_std(model_name)
    norm_fn = transforms.Normalize(mean=mean, std=std)

    # Blur kernels
    blur1 = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
    blur2 = transforms.GaussianBlur(kernel_size=9, sigma=2.0)
    blur3 = transforms.GaussianBlur(kernel_size=15, sigma=3.5)

    # Pass 1: Clean, Gaussian Noise, Gaussian Blur, Brightness Shift (Single Pass on GPU)
    raw_ds = RawEuroSATDataset(raw_dataset, test_indices)
    raw_loader = DataLoader(raw_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    counters = {
        "clean": 0,
        "n1": 0, "n2": 0, "n3": 0,
        "b1": 0, "b2": 0, "b3": 0,
        "br1": 0, "br2": 0, "br3": 0,
    }
    total = 0

    with torch.no_grad():
        for imgs, labels in raw_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            bsz = labels.size(0)
            total += bsz

            # 1. Clean
            pred_clean = torch.argmax(model(norm_fn(imgs)), dim=1)
            counters["clean"] += (pred_clean == labels).sum().item()

            # 2. Gaussian Noise
            n1_imgs = torch.clamp(imgs + torch.randn_like(imgs) * 0.08, 0.0, 1.0)
            counters["n1"] += (torch.argmax(model(norm_fn(n1_imgs)), dim=1) == labels).sum().item()

            n2_imgs = torch.clamp(imgs + torch.randn_like(imgs) * 0.18, 0.0, 1.0)
            counters["n2"] += (torch.argmax(model(norm_fn(n2_imgs)), dim=1) == labels).sum().item()

            n3_imgs = torch.clamp(imgs + torch.randn_like(imgs) * 0.28, 0.0, 1.0)
            counters["n3"] += (torch.argmax(model(norm_fn(n3_imgs)), dim=1) == labels).sum().item()

            # 3. Gaussian Blur
            b1_imgs = blur1(imgs)
            counters["b1"] += (torch.argmax(model(norm_fn(b1_imgs)), dim=1) == labels).sum().item()

            b2_imgs = blur2(imgs)
            counters["b2"] += (torch.argmax(model(norm_fn(b2_imgs)), dim=1) == labels).sum().item()

            b3_imgs = blur3(imgs)
            counters["b3"] += (torch.argmax(model(norm_fn(b3_imgs)), dim=1) == labels).sum().item()

            # 4. Brightness Shift
            br1_imgs = torch.clamp(imgs * 1.35, 0.0, 1.0)
            counters["br1"] += (torch.argmax(model(norm_fn(br1_imgs)), dim=1) == labels).sum().item()

            br2_imgs = torch.clamp(imgs * 1.65, 0.0, 1.0)
            counters["br2"] += (torch.argmax(model(norm_fn(br2_imgs)), dim=1) == labels).sum().item()

            br3_imgs = torch.clamp(imgs * 2.00, 0.0, 1.0)
            counters["br3"] += (torch.argmax(model(norm_fn(br3_imgs)), dim=1) == labels).sum().item()

    clean_acc = round(counters["clean"] / total * 100.0, 2)
    for c in corruptions:
        results[c][0] = clean_acc

    results["gaussian_noise"][1] = round(counters["n1"] / total * 100.0, 2)
    results["gaussian_noise"][2] = round(counters["n2"] / total * 100.0, 2)
    results["gaussian_noise"][3] = round(counters["n3"] / total * 100.0, 2)

    results["gaussian_blur"][1] = round(counters["b1"] / total * 100.0, 2)
    results["gaussian_blur"][2] = round(counters["b2"] / total * 100.0, 2)
    results["gaussian_blur"][3] = round(counters["b3"] / total * 100.0, 2)

    results["brightness_shift"][1] = round(counters["br1"] / total * 100.0, 2)
    results["brightness_shift"][2] = round(counters["br2"] / total * 100.0, 2)
    results["brightness_shift"][3] = round(counters["br3"] / total * 100.0, 2)

    # Pass 2: JPEG Compression (Severities 1, 2, 3)
    for sev in severities:
        corr_ds = CorruptedEuroSATDataset(raw_dataset, test_indices, "jpeg_compression", sev, model_name)
        corr_loader = DataLoader(corr_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        c_correct = 0
        with torch.no_grad():
            for imgs, labels in corr_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = torch.argmax(model(imgs), dim=1)
                c_correct += (preds == labels).sum().item()
        results["jpeg_compression"][sev] = round(c_correct / total * 100.0, 2)

    return results


def plot_corruption_robustness_figure(
    df: pd.DataFrame,
    output_path: Path = cfg.FIGURES_DIR / "corruption_robustness.png",
):
    """
    Generates 4-panel publication figure plotting Accuracy vs Severity for each corruption.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    corruption_titles = {
        "gaussian_noise": "Gaussian Additive Noise",
        "gaussian_blur": "Atmospheric Gaussian Blur",
        "brightness_shift": "Solar Illumination / Over-Exposure",
        "jpeg_compression": "Lossy JPEG Transmission Artifacts",
    }

    # Style mapping per model and strategy
    styles = {
        ("VIT", "scratch"): {"color": "#1f77b4", "linestyle": ":", "marker": "x", "label": "ViT Scratch"},
        ("VIT", "full_ft"): {"color": "#1f77b4", "linestyle": "-", "marker": "o", "label": "ViT Full-FT"},
        ("VIT", "unfreeze_last_3"): {"color": "#1f77b4", "linestyle": "--", "marker": "s", "label": "ViT Unfreeze-Last-3"},

        ("DEIT", "scratch"): {"color": "#2ca02c", "linestyle": ":", "marker": "x", "label": "DeiT Scratch"},
        ("DEIT", "full_ft"): {"color": "#2ca02c", "linestyle": "-", "marker": "o", "label": "DeiT Full-FT"},
        ("DEIT", "unfreeze_last_3"): {"color": "#2ca02c", "linestyle": "--", "marker": "s", "label": "DeiT Unfreeze-Last-3"},

        ("SWIN", "scratch"): {"color": "#ff7f0e", "linestyle": ":", "marker": "x", "label": "Swin Scratch"},
        ("SWIN", "full_ft"): {"color": "#ff7f0e", "linestyle": "-", "marker": "o", "label": "Swin Full-FT"},
        ("SWIN", "unfreeze_last_3"): {"color": "#ff7f0e", "linestyle": "--", "marker": "s", "label": "Swin Unfreeze-Last-3"},
    }

    for idx, (corr_key, title) in enumerate(corruption_titles.items()):
        ax = axes[idx // 2, idx % 2]
        df_corr = df[df["corruption"] == corr_key]

        severities = [0, 1, 2, 3]

        for (fam, strat), style in styles.items():
            sub = df_corr[(df_corr["model_family"] == fam) & (df_corr["strategy"] == strat)].sort_values("severity")
            if not sub.empty:
                ax.plot(
                    sub["severity"], sub["accuracy"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2.0,
                    markersize=6,
                    label=style["label"],
                )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Corruption Severity (0 = Clean)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(severities)
        ax.set_xticklabels(["0 (Clean)", "1 (Mild)", "2 (Moderate)", "3 (Severe)"], fontsize=9.5)
        ax.set_ylim(20.0, 102.0)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(frameon=True, fontsize=8.5, ncol=2, loc="lower left")

    plt.suptitle("Out-of-Distribution Corruption Robustness across Vision Transformer Strategies\n"
                 "(Pretrained Representations vs Scratch across Gaussian Noise, Blur, Brightness, and JPEG Compression)",
                 fontsize=14, fontweight="bold", y=1.002)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[Phase 6 Figures] Saved corruption robustness figure to: {output_path} (DPI={cfg.DPI})")


# ==============================================================================
# 3. Penultimate Feature Extraction, t-SNE & Silhouette Score Engine
# ==============================================================================

def extract_penultimate_features(
    model: nn.Module,
    sample_indices: List[int],
    raw_dataset,
    model_name: str,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts penultimate feature representations (pre-logits) for sample indices.
    Returns: (features_array [N, D], labels_array [N])
    """
    model.eval()
    mean, std = get_timm_mean_std(model_name)
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    features_list = []
    labels_list = []

    for i in range(0, len(sample_indices), batch_size):
        batch_idx = sample_indices[i : i + batch_size]
        batch_tensors = []
        batch_labels = []

        for r_idx in batch_idx:
            path, label = raw_dataset.samples[r_idx]
            img = Image.open(path).convert("RGB")
            batch_tensors.append(eval_tf(img))
            batch_labels.append(label)

        imgs_tensor = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            feats = model.forward_features(imgs_tensor)
            pooled = model.forward_head(feats, pre_logits=True)
            features_list.append(pooled.detach().cpu().numpy())
            labels_list.extend(batch_labels)

    features_arr = np.concatenate(features_list, axis=0)
    labels_arr = np.array(labels_list)
    return features_arr, labels_arr


def compute_tsne_and_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, float, float]:
    """
    Computes 2D t-SNE projection and Silhouette Scores on both high-D and 2D spaces.
    """
    # 1. High-D Silhouette score
    if len(np.unique(labels)) >= 2:
        high_d_silhouette = float(silhouette_score(features, labels))
    else:
        high_d_silhouette = 0.0

    # 2. t-SNE projection to 2D
    perplexity = min(30, max(5, len(labels) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000)
    tsne_2d = tsne.fit_transform(features)

    # 3. 2D Silhouette score
    if len(np.unique(labels)) >= 2:
        tsne_2d_silhouette = float(silhouette_score(tsne_2d, labels))
    else:
        tsne_2d_silhouette = 0.0

    return tsne_2d, high_d_silhouette, tsne_2d_silhouette


def plot_tsne_comparison_figure(
    tsne_results: Dict[str, Dict[str, Any]],
    output_path: Path = cfg.FIGURES_DIR / "tsne_feature_space.png",
):
    """
    Reproduces Khan et al. (2024) Figure 9:
    Multi-panel 2D t-SNE manifold comparison between Scratch vs Pretrained (Full-FT & UnfreezeLast3).
    """
    # 3 rows (ViT, DeiT, Swin) x 3 columns (Scratch, UnfreezeLast3, Full-FT)
    families = ["VIT", "DEIT", "SWIN"]
    strategies = ["scratch", "unfreeze_last_3", "full_ft"]
    strategy_titles = {
        "scratch": "Random Init (Scratch)",
        "unfreeze_last_3": "Partial FT (Last 3 Blocks)",
        "full_ft": "Full Fine-Tuning",
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    cmap = plt.get_cmap("tab10")

    for r_idx, fam in enumerate(families):
        for c_idx, strat in enumerate(strategies):
            ax = axes[r_idx, c_idx]
            key = f"{fam}_{strat}"

            if key not in tsne_results:
                ax.text(0.5, 0.5, "Data Not Available", ha="center", va="center")
                ax.axis("off")
                continue

            rec = tsne_results[key]
            tsne_2d = rec["tsne_2d"]
            labels = rec["labels"]
            sil_high = rec["high_d_silhouette"]
            sil_2d = rec["tsne_2d_silhouette"]

            # Scatter plot for each class
            for cls_idx in range(cfg.NUM_CLASSES):
                cls_mask = (labels == cls_idx)
                ax.scatter(
                    tsne_2d[cls_mask, 0],
                    tsne_2d[cls_mask, 1],
                    color=cmap(cls_idx),
                    label=cfg.CLASS_NAMES[cls_idx] if (r_idx == 0 and c_idx == 0) else None,
                    alpha=0.75,
                    s=14,
                    edgecolors="none",
                )

            # Titles and silhouette scores
            ax.set_title(
                f"{fam} - {strategy_titles[strat]}\n"
                f"Silhouette: {sil_high:+.3f} (Feature Space) | {sil_2d:+.3f} (t-SNE 2D)",
                fontsize=11, fontweight="bold", pad=8
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            # Border formatting
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(1.0)

    # Class legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        title="EuroSAT Land Use & Land Cover (LULC) Classes",
        title_fontproperties={"weight": "bold", "size": 12},
    )

    plt.suptitle("Reproduction of Khan et al. (2024) Figure 9: Penultimate Feature Manifold Clustering\n"
                 "(t-SNE Embeddings & Silhouette Cluster Quality: Scratch vs. Partial FT vs. Full FT)",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[Phase 6 Figures] Saved t-SNE feature space figure to: {output_path} (DPI={cfg.DPI})")


# ==============================================================================
# 4. Main Phase 6 Benchmark Runner
# ==============================================================================

def run_phase6(
    seed: int = 42,
    tsne_sample_size: int = 1500,
):
    """
    Executes full Phase 6 benchmark:
    (a) Corruption Robustness on Test Set across 4 corruptions x 3 severities
    (b) Feature-space t-SNE projection and Silhouette Score quantification
    """
    print("=" * 80)
    print("PHASE 6: Corruption Robustness & Feature-Space t-SNE Analysis")
    print("=" * 80)

    device = torch.device(cfg.DEVICE)
    raw_dataset = load_raw_eurosat(cfg.DATA_DIR)

    with open(cfg.SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_indices = splits["test"]

    # Stratified subset of test set for reproducible, clean t-SNE
    rng = random.Random(seed)
    targets = np.array([raw_dataset.targets[idx] for idx in test_indices])
    tsne_indices = []
    n_per_class = tsne_sample_size // cfg.NUM_CLASSES

    for c in range(cfg.NUM_CLASSES):
        cls_mask = (targets == c)
        c_idxs = [test_indices[i] for i in np.where(cls_mask)[0]]
        picked = rng.sample(c_idxs, min(n_per_class, len(c_idxs)))
        tsne_indices.extend(picked)

    print(f"[Phase 6] Selected {len(tsne_indices)} stratified test samples for t-SNE feature extraction.")

    # Target models to evaluate
    eval_matrix = [
        ("vit", "scratch"),
        ("vit", "unfreeze_last_3"),
        ("vit", "full_ft"),
        ("deit", "scratch"),
        ("deit", "unfreeze_last_3"),
        ("deit", "full_ft"),
        ("swin", "scratch"),
        ("swin", "unfreeze_last_3"),
        ("swin", "full_ft"),
    ]

    corruption_table = []
    tsne_metrics = []
    tsne_plot_dict: Dict[str, Dict[str, Any]] = {}

    for family, strategy in eval_matrix:
        model_name = cfg.get_model_name(family)
        clean_m = clean_name(model_name)
        ckpt_path = cfg.CHECKPOINTS_DIR / f"{clean_m}_{strategy}_seed{seed}_best.pth"

        if not ckpt_path.exists():
            print(f"[Phase 6] Warning: Checkpoint {ckpt_path.name} not found. Skipping.")
            continue

        print(f"\n================================================================================")
        print(f"Loading Model: {family.upper()} [{strategy}] from {ckpt_path.name}")
        print(f"================================================================================")
        model, _ = build_transformer_model(model_name, strategy=strategy)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).eval()

        # ----------------------------------------------------------------------
        # Part (a): Corruption Robustness Evaluation
        # ----------------------------------------------------------------------
        print(f"  --> [Robustness] Evaluating 4 corruptions x 3 severities on test set ({len(test_indices)} images)...")
        robustness_dict = evaluate_corruption_robustness(
            model=model,
            test_indices=test_indices,
            raw_dataset=raw_dataset,
            model_name=model_name,
            device=device,
            batch_size=64,
        )

        for corr_type, sev_dict in robustness_dict.items():
            clean_acc = sev_dict[0]
            for sev, acc in sev_dict.items():
                acc_drop = round(clean_acc - acc, 2)
                corruption_table.append({
                    "model_family": family.upper(),
                    "model_name": model_name,
                    "strategy": strategy,
                    "corruption": corr_type,
                    "severity": sev,
                    "accuracy": acc,
                    "clean_accuracy": clean_acc,
                    "accuracy_drop": acc_drop,
                })

        # ----------------------------------------------------------------------
        # Part (b): Feature-Space t-SNE & Silhouette Score
        # ----------------------------------------------------------------------
        print(f"  --> [t-SNE] Extracting penultimate features on {len(tsne_indices)} samples...")
        features_arr, labels_arr = extract_penultimate_features(
            model=model,
            sample_indices=tsne_indices,
            raw_dataset=raw_dataset,
            model_name=model_name,
            device=device,
            batch_size=64,
        )

        print(f"  --> [t-SNE] Computing 2D manifold projection & Silhouette scores...")
        tsne_2d, high_d_sil, tsne_2d_sil = compute_tsne_and_silhouette(
            features_arr, labels_arr, random_state=seed
        )

        key = f"{family.upper()}_{strategy}"
        tsne_plot_dict[key] = {
            "tsne_2d": tsne_2d,
            "labels": labels_arr,
            "high_d_silhouette": high_d_sil,
            "tsne_2d_silhouette": tsne_2d_sil,
        }

        tsne_metrics.append({
            "model_family": family.upper(),
            "model_name": model_name,
            "strategy": strategy,
            "feature_dim": int(features_arr.shape[1]),
            "feature_space_silhouette_score": round(high_d_sil, 4),
            "tsne_2d_silhouette_score": round(tsne_2d_sil, 4),
        })

        print(f"      Silhouette Score: Feature Space={high_d_sil:+.4f} | t-SNE 2D={tsne_2d_sil:+.4f}")

        # Explicitly clean GPU memory
        del model
        torch.cuda.empty_cache()

    # Save Corruption Robustness Table
    df_robustness = pd.DataFrame(corruption_table)
    robustness_csv = cfg.RESULTS_DIR / "corruption_robustness.csv"
    robustness_json = cfg.RESULTS_DIR / "corruption_robustness.json"
    df_robustness.to_csv(robustness_csv, index=False)
    with open(robustness_json, "w", encoding="utf-8") as f:
        json.dump(corruption_table, f, indent=2)
    print(f"\n[Phase 6] Saved corruption robustness metrics to: {robustness_csv}")

    # Save t-SNE Metrics Table
    df_tsne = pd.DataFrame(tsne_metrics)
    tsne_csv = cfg.RESULTS_DIR / "tsne_silhouette_scores.csv"
    tsne_json = cfg.RESULTS_DIR / "tsne_metrics.json"
    df_tsne.to_csv(tsne_csv, index=False)
    with open(tsne_json, "w", encoding="utf-8") as f:
        json.dump(tsne_metrics, f, indent=2)
    print(f"[Phase 6] Saved t-SNE silhouette metrics to: {tsne_csv}")

    # Plot Figures
    plot_corruption_robustness_figure(df_robustness, cfg.FIGURES_DIR / "corruption_robustness.png")
    plot_tsne_comparison_figure(tsne_plot_dict, cfg.FIGURES_DIR / "tsne_feature_space.png")

    # Print Formatted Summaries to Console
    _print_phase6_summary(df_robustness, df_tsne)

    return df_robustness, df_tsne


def _print_phase6_summary(df_rob: pd.DataFrame, df_tsne: pd.DataFrame):
    """Prints formatted summary tables for Phase 6."""
    print("\n" + "=" * 100)
    print("PHASE 6: CLUSTER SEPARATION & SILHOUETTE SCORE SUMMARY (FIGURE 9 QUANTIFICATION)")
    print("=" * 100)
    print(f"{'Model':<8}{'Strategy':<18}{'Feature Dim':<15}{'High-D Silhouette':<22}{'t-SNE 2D Silhouette':<22}")
    print("-" * 100)
    for _, r in df_tsne.iterrows():
        print(f"{r['model_family']:<8}{r['strategy']:<18}{r['feature_dim']:<15}{r['feature_space_silhouette_score']:<22.4f}{r['tsne_2d_silhouette_score']:<22.4f}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("PHASE 6: CORRUPTION ROBUSTNESS SUMMARY (MEAN ACCURACY DROP AT SEVERITY 3)")
    print("=" * 100)
    sev3 = df_rob[df_rob["severity"] == 3]
    pivot_drop = sev3.pivot_table(index=["model_family", "strategy"], columns="corruption", values="accuracy_drop")
    print(pivot_drop.to_string())
    print("=" * 100)


if __name__ == "__main__":
    run_phase6(seed=42, tsne_sample_size=1500)
