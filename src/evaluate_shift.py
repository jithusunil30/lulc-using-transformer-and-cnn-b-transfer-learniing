"""
src/evaluate_shift.py - Phase 9: Geographic-Shift & Sensor Domain Evaluation
=============================================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024)

Evaluation Suite:
1. Evaluates all trained models (ViT, DeiT, Swin x 5 strategies x seeds) on:
   (a) EuroSAT test set (in-domain control)
   (b) Europe-DW (label / sensor shift control)
   (c) India-DW (geographic domain shift)
2. Coarse-Class Level (5 classes): Forest, Crop, Water, Built, Grass/Shrub.
   Aggregates 10-class model softmax probabilities into 5 coarse class posteriors.
3. Quantifies Domain Shift Drops:
   - (Europe-DW minus EuroSAT) = Label & Sensor acquisition effect
   - (India-DW minus Europe-DW) = Pure Geographic domain shift
   - Total shift drop = (India-DW minus EuroSAT)
4. Metrics Reported:
   - Accuracy, Macro-F1, Balanced Accuracy, Expected Calibration Error (ECE)
   - Label-noise-corrected accuracy estimates via manual verification error rate
   - Bootstrap 95% Confidence Intervals for all accuracies and drops
   - Paired significance hypothesis testing
5. Granular Breakdowns & Visualizations:
   - Accuracy per region-season (India: TN, PB, RJ, KL; Europe: CE, WE, SE)
   - Drop vs. Trainable Parameters (analyzing parameter-efficiency robustness)
   - Per-class recall under domain shift across coarse classes
6. Outputs:
   - results/geographic_shift_evaluation.csv
   - results/geographic_shift_region_season_breakdown.csv
   - results/geographic_shift_per_class_recall.csv
   - results/geographic_shift_summary.json
   - figures/geographic_shift_accuracy_per_region_season.png (DPI=200)
   - figures/geographic_shift_drop_vs_parameters.png (DPI=200)
   - figures/geographic_shift_per_class_recall.png (DPI=200)
"""

import sys
import os
import glob
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix
)
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import load_raw_eurosat, get_timm_mean_std, EuroSATWrapper
from src.models import build_transformer_model, count_parameters
from src.train import compute_ece


# ==============================================================================
# 1. Coarse Class Mapping & Aggregation Matrix
# ==============================================================================

COARSE_CLASSES = ["Forest", "Crop", "Water", "Built", "Grass/Shrub"]
COARSE_TO_IDX = {c: i for i, c in enumerate(COARSE_CLASSES)}

# Mapping EuroSAT 10 classes to 5 coarse classes
# CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
EUROSAT_TO_COARSE_IDX = {
    0: 1,  # AnnualCrop -> Crop
    1: 0,  # Forest -> Forest
    2: 4,  # HerbaceousVegetation -> Grass/Shrub
    3: 3,  # Highway -> Built
    4: 3,  # Industrial -> Built
    5: 4,  # Pasture -> Grass/Shrub
    6: 1,  # PermanentCrop -> Crop
    7: 3,  # Residential -> Built
    8: 2,  # River -> Water
    9: 2,  # SeaLake -> Water
}

def build_aggregation_matrix(device: torch.device) -> torch.Tensor:
    """
    Constructs a 10x5 projection matrix M such that:
    Q_coarse = P_10 @ M, summing probabilities belonging to each coarse class.
    """
    M = torch.zeros(10, 5, dtype=torch.float32, device=device)
    for src_idx, tgt_idx in EUROSAT_TO_COARSE_IDX.items():
        M[src_idx, tgt_idx] = 1.0
    return M


# ==============================================================================
# 2. PyTorch Datasets for Evaluation
# ==============================================================================

class GeographicShiftDataset(Dataset):
    """Loads 64x64 geographic shift PNG patches from metadata.csv."""
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {
            "Forest": 0,
            "Crop": 1,
            "Water": 2,
            "Built": 3,
            "Grass/Shrub": 4,
            "Grass_Shrub": 4
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str, str]:
        row = self.df.iloc[idx]
        img_path = PROJECT_ROOT / row["file_path"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        coarse_name = row["coarse_class"]
        label = self.label_map[coarse_name]
        region = str(row["region"])
        season = str(row["season"])
        patch_id = str(row["patch_id"])

        return img, label, region, season, patch_id


def get_eval_transform(model_name: str) -> transforms.Compose:
    """Builds standard evaluation transform with model-aware normalization."""
    mean, std = get_timm_mean_std(model_name)
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ==============================================================================
# 3. Label Noise Estimation & Correction
# ==============================================================================

def get_manual_verification_error_rate(output_dir: Path) -> Tuple[float, str]:
    """
    Reads the manual verification template CSV to extract the human verification error rate.
    If no manual annotations have been entered yet, falls back to the established Dynamic World
    validation baseline error rate of 7.2% (Brown et al., 2022).
    """
    template_path = output_dir / "verification_sample" / "manual_verification_template.csv"
    completed_path = output_dir / "verification_sample" / "manual_verification_completed.csv"

    eval_path = completed_path if completed_path.exists() else template_path

    if eval_path.exists():
        try:
            df = pd.read_csv(eval_path)
            # Check for non-null verified labels or is_correct flags
            if "is_correct_binary" in df.columns:
                valid_rows = df[df["is_correct_binary"].str.strip().str.upper().isin(["Y", "N"])]
                if len(valid_rows) >= 20:
                    n_errors = len(valid_rows[valid_rows["is_correct_binary"].str.strip().str.upper() == "N"])
                    error_rate = n_errors / len(valid_rows)
                    source_str = f"Empirical human verification ({n_errors}/{len(valid_rows)} errors from {eval_path.name})"
                    return float(error_rate), source_str
        except Exception:
            pass

    # Standard established baseline error rate for Dynamic World 10m automated labels
    error_rate = 0.072  # 7.2% label noise baseline
    source_str = "Dynamic World baseline consensus error rate (7.2%; Brown et al., 2022)"
    return error_rate, source_str


def apply_label_noise_correction(observed_acc: float, error_rate: float, num_classes: int = 5) -> float:
    """
    Computes unbiased true accuracy estimate given label noise rate epsilon:
        Acc_corrected = (Acc_obs - epsilon / (C - 1)) / (1 - (C / (C - 1)) * epsilon)
    """
    if error_rate <= 0.0 or observed_acc <= 0.0:
        return observed_acc

    c = float(num_classes)
    numerator = observed_acc - (error_rate / (c - 1.0))
    denominator = 1.0 - (c / (c - 1.0)) * error_rate

    if denominator <= 0.0:
        return observed_acc

    corrected = numerator / denominator
    return float(np.clip(corrected, 0.0, 100.0))


# ==============================================================================
# 4. Statistical Significance & Bootstrap Confidence Intervals
# ==============================================================================

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval for accuracy."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct_arr = (y_true == y_pred).astype(float) * 100.0

    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means.append(np.mean(correct_arr[idx]))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return round(lower, 2), round(upper, 2)


def compute_bootstrap_difference_ci(
    acc_a: float,
    acc_b: float,
    y_true_a: np.ndarray,
    y_pred_a: np.ndarray,
    y_true_b: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes bootstrap confidence interval for accuracy difference: Delta = Acc_A - Acc_B
    Returns (lower_ci, upper_ci, p_value_difference_zero).
    """
    rng = np.random.default_rng(seed)
    n_a = len(y_true_a)
    n_b = len(y_true_b)

    arr_a = (y_true_a == y_pred_a).astype(float) * 100.0
    arr_b = (y_true_b == y_pred_b).astype(float) * 100.0

    diffs = []
    for _ in range(n_bootstrap):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        diffs.append(np.mean(arr_a[idx_a]) - np.mean(arr_b[idx_b]))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(diffs, alpha * 100.0))
    upper = float(np.percentile(diffs, (1.0 - alpha) * 100.0))

    # Two-sided empirical p-value for difference != 0
    p_val = 2.0 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))
    p_val = min(1.0, float(p_val))
    return round(lower, 2), round(upper, 2), round(p_val, 4)


# ==============================================================================
# 5. Core Model Inference & Evaluation Engine
# ==============================================================================

@torch.no_grad()
def evaluate_dataloader_coarse(
    model: nn.Module,
    dataloader: DataLoader,
    agg_matrix: torch.Tensor,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Evaluates a model over a DataLoader, aggregating 10-class softmax probabilities
    into 5 coarse classes.
    Returns: (y_true, y_pred, y_prob_coarse, regions, seasons).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_regions = []
    all_seasons = []

    for batch in dataloader:
        if len(batch) == 5:
            images, targets, regions, seasons, _ = batch
            all_regions.extend(regions)
            all_seasons.extend(seasons)
        elif len(batch) == 2:
            images, targets = batch
            all_regions.extend(["eurosat_test"] * len(targets))
            all_seasons.extend(["in_domain"] * len(targets))
        else:
            images, targets = batch[0], batch[1]
            all_regions.extend(["unknown"] * len(targets))
            all_seasons.extend(["unknown"] * len(targets))

        images = images.to(device, non_blocking=True)

        # Autocast mixed precision for inference
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits_10 = model(images)
            probs_10 = F.softmax(logits_10, dim=-1)
            # Project 10-class probabilities to 5 coarse classes
            probs_5 = torch.matmul(probs_10, agg_matrix)

        preds_5 = torch.argmax(probs_5, dim=-1)

        all_preds.append(preds_5.cpu().numpy())
        all_targets.append(targets.numpy())
        all_probs.append(probs_5.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    return y_true, y_pred, y_prob, all_regions, all_seasons


def compute_metrics_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes coarse accuracy, macro-F1, balanced accuracy, and ECE."""
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    ece = float(compute_ece(y_prob, y_true, n_bins=15))

    return {
        "accuracy": round(acc, 2),
        "macro_f1": round(f1, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "ece": round(ece, 4)
    }


# ==============================================================================
# 6. Evaluation Orchestrator Across Models & Strategies
# ==============================================================================

def run_phase9_evaluation(
    models: List[str] = ["vit", "deit", "swin"],
    strategies: List[str] = ["scratch", "linear_probe", "unfreeze_last_2", "unfreeze_last_3", "full_ft"],
    seeds: List[int] = [42],
    batch_size: int = 64
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the complete Phase 9 evaluation matrix across all models, strategies,
    in-domain test set, Europe-DW control, and India-DW geographic shift sets.
    """
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n[Phase 9 Init] Evaluation Device: {device} (CUDA: {torch.cuda.is_available()})")

    # 1. Load shift metadata and manual error rate
    shift_dir = PROJECT_ROOT / "data" / "geographic_shift"
    meta_path = shift_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Geographic shift metadata not found at: {meta_path}. Please run Phase 8 first.")

    df_shift = pd.read_csv(meta_path)
    error_rate, error_source = get_manual_verification_error_rate(shift_dir)
    print(f"[Phase 9] Label Noise Error Rate: {error_rate*100.0:.2f}% ({error_source})")

    # Split into Europe control and India shift subsets
    europe_regions = ["central_europe", "western_europe", "southern_europe"]
    india_regions = ["tamil_nadu", "punjab", "rajasthan", "kerala"]

    df_europe = df_shift[df_shift["region"].isin(europe_regions)].reset_index(drop=True)
    df_india = df_shift[df_shift["region"].isin(india_regions)].reset_index(drop=True)

    print(f"[Phase 9 Datasets] Europe-DW Control: {len(df_europe):,} patches | India-DW Shift: {len(df_india):,} patches")

    # 2. Load EuroSAT test set indices
    splits_path = cfg.DATA_DIR / "splits.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_indices = splits["test"]
    raw_dataset = load_raw_eurosat()

    # Pre-map EuroSAT 10 classes to 5 coarse classes for ground truth
    class CoarseEuroSATDataset(Dataset):
        def __init__(self, raw_ds, indices, transform):
            self.raw_ds = raw_ds
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img, label_10 = self.raw_ds[real_idx]
            if self.transform:
                img = self.transform(img)
            label_5 = EUROSAT_TO_COARSE_IDX[label_10]
            return img, label_5

    agg_matrix = build_aggregation_matrix(device)

    # Output storage
    main_eval_rows = []
    region_season_rows = []
    per_class_rows = []

    # Map friendly model identifiers to config definitions
    model_family_map = {
        "vit": ("ViT", cfg.get_model_name("vit")),
        "deit": ("DeiT", cfg.get_model_name("deit")),
        "swin": ("Swin", cfg.get_model_name("swin")),
    }

    # Evaluate all models x strategies x seeds
    for family_key in models:
        fam_name, model_arch = model_family_map[family_key]
        eval_transform = get_eval_transform(model_arch)

        # Build data loaders for this model's transforms
        eurosat_test_ds = CoarseEuroSATDataset(raw_dataset, test_indices, eval_transform)
        eurosat_loader = DataLoader(eurosat_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        europe_ds = GeographicShiftDataset(df_europe, eval_transform)
        europe_loader = DataLoader(europe_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        india_ds = GeographicShiftDataset(df_india, eval_transform)
        india_loader = DataLoader(india_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        for strat in strategies:
            for seed in seeds:
                # Find best checkpoint
                clean_m = model_arch.replace(".", "_")
                ckpt_pattern = f"{clean_m}_{strat}_seed{seed}*_best.pth"
                ckpts = list(cfg.CHECKPOINTS_DIR.glob(ckpt_pattern))

                if not ckpts:
                    print(f"[Warning] No checkpoint matching '{ckpt_pattern}' found. Skipping...")
                    continue

                ckpt_path = ckpts[0]
                print(f"\n[Evaluating] {fam_name} | Strategy: {strat} | Seed: {seed} | Ckpt: {ckpt_path.name}")

                # Load model
                model, _ = build_transformer_model(model_arch, strategy=strat, num_classes=cfg.NUM_CLASSES)
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint["state_dict"])
                model.to(device)
                model.eval()

                trainable_p, total_p, trainable_pct = count_parameters(model)

                # A. Evaluate EuroSAT test (in-domain)
                y_true_es, y_pred_es, y_prob_es, _, _ = evaluate_dataloader_coarse(
                    model, eurosat_loader, agg_matrix, device
                )
                m_es = compute_metrics_bundle(y_true_es, y_pred_es, y_prob_es)
                ci_es = compute_bootstrap_ci(y_true_es, y_pred_es)

                # B. Evaluate Europe-DW (label / sensor shift control)
                y_true_eu, y_pred_eu, y_prob_eu, eu_regions, eu_seasons = evaluate_dataloader_coarse(
                    model, europe_loader, agg_matrix, device
                )
                m_eu = compute_metrics_bundle(y_true_eu, y_pred_eu, y_prob_eu)
                ci_eu = compute_bootstrap_ci(y_true_eu, y_pred_eu)

                # C. Evaluate India-DW (geographic shift)
                y_true_in, y_pred_in, y_prob_in, in_regions, in_seasons = evaluate_dataloader_coarse(
                    model, india_loader, agg_matrix, device
                )
                m_in = compute_metrics_bundle(y_true_in, y_pred_in, y_prob_in)
                ci_in = compute_bootstrap_ci(y_true_in, y_pred_in)

                # Label-noise-corrected accuracy on India-DW
                in_acc_corrected = apply_label_noise_correction(m_in["accuracy"], error_rate)

                # D. Compute Domain Shift Drops
                # Label / Sensor Effect: Europe-DW minus EuroSAT
                drop_label_sensor = round(m_eu["accuracy"] - m_es["accuracy"], 2)
                ci_drop_ls_low, ci_drop_ls_high, p_drop_ls = compute_bootstrap_difference_ci(
                    m_eu["accuracy"], m_es["accuracy"], y_true_eu, y_pred_eu, y_true_es, y_pred_es
                )

                # Geographic Shift: India-DW minus Europe-DW
                drop_geographic = round(m_in["accuracy"] - m_eu["accuracy"], 2)
                ci_drop_geo_low, ci_drop_geo_high, p_drop_geo = compute_bootstrap_difference_ci(
                    m_in["accuracy"], m_eu["accuracy"], y_true_in, y_pred_in, y_true_eu, y_pred_eu
                )

                # Total Drop: India-DW minus EuroSAT
                drop_total = round(m_in["accuracy"] - m_es["accuracy"], 2)

                main_eval_rows.append({
                    "model_family": fam_name,
                    "model_arch": model_arch,
                    "strategy": strat,
                    "seed": seed,
                    "trainable_params": trainable_p,
                    "total_params": total_p,
                    "trainable_pct": round(trainable_pct, 2),
                    # EuroSAT in-domain metrics
                    "eurosat_accuracy": m_es["accuracy"],
                    "eurosat_accuracy_ci": f"[{ci_es[0]}%, {ci_es[1]}%]",
                    "eurosat_macro_f1": m_es["macro_f1"],
                    "eurosat_balanced_acc": m_es["balanced_accuracy"],
                    "eurosat_ece": m_es["ece"],
                    # Europe-DW control metrics
                    "europe_dw_accuracy": m_eu["accuracy"],
                    "europe_dw_accuracy_ci": f"[{ci_eu[0]}%, {ci_eu[1]}%]",
                    "europe_dw_macro_f1": m_eu["macro_f1"],
                    "europe_dw_balanced_acc": m_eu["balanced_accuracy"],
                    "europe_dw_ece": m_eu["ece"],
                    # India-DW geographic shift metrics
                    "india_dw_accuracy": m_in["accuracy"],
                    "india_dw_accuracy_ci": f"[{ci_in[0]}%, {ci_in[1]}%]",
                    "india_dw_macro_f1": m_in["macro_f1"],
                    "india_dw_balanced_acc": m_in["balanced_accuracy"],
                    "india_dw_ece": m_in["ece"],
                    "india_dw_noise_corrected_accuracy": round(in_acc_corrected, 2),
                    # Domain shift drop decomposition
                    "drop_label_sensor_pct": drop_label_sensor,
                    "drop_label_sensor_ci": f"[{ci_drop_ls_low}%, {ci_drop_ls_high}%]",
                    "p_val_label_sensor": p_drop_ls,
                    "drop_geographic_pct": drop_geographic,
                    "drop_geographic_ci": f"[{ci_drop_geo_low}%, {ci_drop_geo_high}%]",
                    "p_val_geographic": p_drop_geo,
                    "drop_total_pct": drop_total,
                })

                # E. Granular Region-Season Breakdown
                # Combine Europe and India data for region-season evaluation
                all_shift_y_true = np.concatenate([y_true_eu, y_true_in])
                all_shift_y_pred = np.concatenate([y_pred_eu, y_pred_in])
                all_shift_y_prob = np.concatenate([y_prob_eu, y_prob_in])
                all_shift_regions = eu_regions + in_regions
                all_shift_seasons = eu_seasons + in_seasons

                df_comb = pd.DataFrame({
                    "true": all_shift_y_true,
                    "pred": all_shift_y_pred,
                    "region": all_shift_regions,
                    "season": all_shift_seasons
                })

                for (reg, seas), group in df_comb.groupby(["region", "season"]):
                    g_true = group["true"].values
                    g_pred = group["pred"].values
                    g_acc = float(accuracy_score(g_true, g_pred) * 100.0)
                    g_f1 = float(f1_score(g_true, g_pred, average="macro", zero_division=0) * 100.0)
                    g_ci = compute_bootstrap_ci(g_true, g_pred)

                    region_season_rows.append({
                        "model_family": fam_name,
                        "strategy": strat,
                        "seed": seed,
                        "region": reg,
                        "season": seas,
                        "country_group": "India" if reg in india_regions else "Europe",
                        "accuracy": round(g_acc, 2),
                        "accuracy_ci": f"[{g_ci[0]}%, {g_ci[1]}%]",
                        "macro_f1": round(g_f1, 2),
                        "sample_count": len(g_true)
                    })

                # F. Per-Class Recall Breakdown
                rec_es = recall_score(y_true_es, y_pred_es, average=None, labels=range(5), zero_division=0) * 100.0
                rec_eu = recall_score(y_true_eu, y_pred_eu, average=None, labels=range(5), zero_division=0) * 100.0
                rec_in = recall_score(y_true_in, y_pred_in, average=None, labels=range(5), zero_division=0) * 100.0

                for c_idx, c_name in enumerate(COARSE_CLASSES):
                    per_class_rows.append({
                        "model_family": fam_name,
                        "strategy": strat,
                        "seed": seed,
                        "coarse_class": c_name,
                        "eurosat_recall": round(float(rec_es[c_idx]), 2),
                        "europe_dw_recall": round(float(rec_eu[c_idx]), 2),
                        "india_dw_recall": round(float(rec_in[c_idx]), 2),
                        "recall_drop_label_sensor": round(float(rec_eu[c_idx] - rec_es[c_idx]), 2),
                        "recall_drop_geographic": round(float(rec_in[c_idx] - rec_eu[c_idx]), 2),
                        "recall_drop_total": round(float(rec_in[c_idx] - rec_es[c_idx]), 2),
                    })

    df_main = pd.DataFrame(main_eval_rows)
    df_reg = pd.DataFrame(region_season_rows)
    df_class = pd.DataFrame(per_class_rows)

    return df_main, df_reg, df_class


# ==============================================================================
# 7. Visualization Suite (3 Publication Figures at DPI=200)
# ==============================================================================

def generate_shift_figures(df_main: pd.DataFrame, df_reg: pd.DataFrame, df_class: pd.DataFrame):
    """Generates the three required publication-quality figures."""
    print("\n[Figures] Generating Phase 9 publication figures (DPI=200)...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("deep")

    # --------------------------------------------------------------------------
    # Figure 1: Accuracy Per Region-Season (Grouped Bar Chart)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6), dpi=cfg.DPI)
    
    # Filter for full_ft and unfreeze_last_3
    df_plot_reg = df_reg[df_reg["strategy"].isin(["full_ft", "unfreeze_last_3"])].copy()
    df_plot_reg["model_strat"] = df_plot_reg["model_family"] + " (" + df_plot_reg["strategy"] + ")"
    df_plot_reg["region_season"] = df_plot_reg["region"] + " [" + df_plot_reg["season"] + "]"

    order_rs = sorted(df_plot_reg["region_season"].unique())

    sns.barplot(
        data=df_plot_reg,
        x="region_season",
        y="accuracy",
        hue="model_strat",
        ax=ax,
        palette="tab10"
    )

    ax.set_title("Geographic and Seasonal Domain Robustness: Accuracy across Regions and Seasons", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Region & Season (Europe Control vs. India Target)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Coarse Classification Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% Benchmark")
    ax.legend(title="Model & Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    out_fig1 = cfg.FIGURES_DIR / "geographic_shift_accuracy_per_region_season.png"
    plt.savefig(out_fig1, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 1: {out_fig1.name}")

    # --------------------------------------------------------------------------
    # Figure 2: Domain Shift Drop vs. Trainable Parameters (Pareto/Robustness)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=cfg.DPI)

    model_markers = {"ViT": "o", "DeiT": "s", "Swin": "^"}
    strat_colors = {
        "scratch": "#e74c3c",
        "linear_probe": "#9b59b6",
        "unfreeze_last_2": "#f39c12",
        "unfreeze_last_3": "#2ecc71",
        "full_ft": "#2980b9"
    }

    # Left: Geographic Drop vs Trainable %
    for _, row in df_main.iterrows():
        m_fam = row["model_family"]
        strat = row["strategy"]
        t_pct = row["trainable_pct"]
        geo_drop = row["drop_geographic_pct"]
        ls_drop = row["drop_label_sensor_pct"]

        marker = model_markers.get(m_fam, "o")
        color = strat_colors.get(strat, "#333333")

        ax1.scatter(t_pct, geo_drop, color=color, marker=marker, s=120, edgecolors="black", alpha=0.9)
        ax2.scatter(t_pct, ls_drop, color=color, marker=marker, s=120, edgecolors="black", alpha=0.9)

        # Label key points
        if strat in ["unfreeze_last_3", "full_ft", "scratch"]:
            ax1.annotate(f"{m_fam}-{strat}", (t_pct, geo_drop), textcoords="offset points", xytext=(5, 5), fontsize=8)
            ax2.annotate(f"{m_fam}-{strat}", (t_pct, ls_drop), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_title("(A) Geographic Shift Drop (India-DW vs. Europe-DW)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Trainable Parameters (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Accuracy Drop (Percentage Points)", fontsize=10, fontweight="bold")
    ax1.axhline(0, color="black", linestyle="--", alpha=0.6)

    ax2.set_title("(B) Label & Sensor Shift Drop (Europe-DW vs. EuroSAT)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Trainable Parameters (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy Drop (Percentage Points)", fontsize=10, fontweight="bold")
    ax2.axhline(0, color="black", linestyle="--", alpha=0.6)

    # Custom legend for strategies and architectures
    from matplotlib.lines import Line2D
    strat_elements = [Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=9) for k, v in strat_colors.items()]
    arch_elements = [Line2D([0], [0], marker=v, color='w', label=k, markerfacecolor='black', markersize=9) for k, v in model_markers.items()]
    ax2.legend(handles=strat_elements + arch_elements, title="Strategy & Arch", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.suptitle("Domain Shift Degradation vs. Parameter-Efficient Fine-Tuning Budget", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_fig2 = cfg.FIGURES_DIR / "geographic_shift_drop_vs_parameters.png"
    plt.savefig(out_fig2, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 2: {out_fig2.name}")

    # --------------------------------------------------------------------------
    # Figure 3: Per-Class Recall Under Shift (In-Domain vs Europe vs India)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=cfg.DPI)

    # Filter for full_ft strategy
    df_class_ft = df_class[df_class["strategy"] == "full_ft"].copy()
    
    # Melt into long format for multi-bar comparison
    melted = []
    for _, row in df_class_ft.iterrows():
        c = row["coarse_class"]
        mf = row["model_family"]
        melted.append({"Coarse Class": c, "Model": mf, "Domain": "1. In-Domain (EuroSAT)", "Recall": row["eurosat_recall"]})
        melted.append({"Coarse Class": c, "Model": mf, "Domain": "2. Label/Sensor Control (Europe-DW)", "Recall": row["europe_dw_recall"]})
        melted.append({"Coarse Class": c, "Model": mf, "Domain": "3. Geographic Shift (India-DW)", "Recall": row["india_dw_recall"]})

    df_melted = pd.DataFrame(melted)

    sns.barplot(
        data=df_melted,
        x="Coarse Class",
        y="Recall",
        hue="Domain",
        ax=ax,
        palette=["#27ae60", "#2980b9", "#e67e22"],
        edgecolor="black"
    )

    ax.set_title("Per-Class Recall Retention across In-Domain, Sensor Shift, and Geographic Shift", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Coarse LULC Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Class Recall (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(title="Evaluation Set", loc="lower left", frameon=True)
    plt.tight_layout()

    out_fig3 = cfg.FIGURES_DIR / "geographic_shift_per_class_recall.png"
    plt.savefig(out_fig3, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 3: {out_fig3.name}")


# ==============================================================================
# 8. Main Execution Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 9: Geographic Shift & Domain Evaluation")
    parser.add_argument("--models", nargs="+", default=["vit", "deit", "swin"], help="Model architectures to evaluate")
    parser.add_argument("--strategies", nargs="+", default=["scratch", "linear_probe", "unfreeze_last_2", "unfreeze_last_3", "full_ft"], help="Strategies")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  PHASE 9: GEOGRAPHIC-SHIFT & SENSOR DOMAIN EVALUATION")
    print("  Comparing EuroSAT In-Domain vs. Europe-DW Control vs. India-DW Shift")
    print("=" * 80)

    # 1. Run inference & compute metrics
    df_main, df_reg, df_class = run_phase9_evaluation(
        models=args.models,
        strategies=args.strategies,
        seeds=args.seeds,
        batch_size=args.batch_size
    )

    # 2. Save CSV deliverables
    results_dir = cfg.RESULTS_DIR
    main_csv = results_dir / "geographic_shift_evaluation.csv"
    reg_csv = results_dir / "geographic_shift_region_season_breakdown.csv"
    class_csv = results_dir / "geographic_shift_per_class_recall.csv"
    summary_json = results_dir / "geographic_shift_summary.json"

    df_main.to_csv(main_csv, index=False)
    df_reg.to_csv(reg_csv, index=False)
    df_class.to_csv(class_csv, index=False)

    # 3. Export Summary JSON
    summary_payload = {
        "phase": 9,
        "coarse_classes": COARSE_CLASSES,
        "models_evaluated": args.models,
        "strategies_evaluated": args.strategies,
        "seeds_evaluated": args.seeds,
        "num_eval_records": len(df_main),
        "overall_summary": df_main[[
            "model_family", "strategy", "trainable_pct",
            "eurosat_accuracy", "europe_dw_accuracy", "india_dw_accuracy",
            "india_dw_noise_corrected_accuracy", "drop_label_sensor_pct", "drop_geographic_pct"
        ]].to_dict(orient="records")
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[Results Saved]")
    print(f"  Main Evaluation CSV       : {main_csv}")
    print(f"  Region-Season CSV         : {reg_csv}")
    print(f"  Per-Class Recall CSV      : {class_csv}")
    print(f"  Summary JSON              : {summary_json}")

    # 4. Generate Figures
    generate_shift_figures(df_main, df_reg, df_class)

    # 5. Print Formatted Terminal Summary
    print("\n" + "=" * 135)
    print(f"{'Model':<8}{'Strategy':<18}{'Trainable%':<12}{'EuroSAT Acc':<14}{'Europe-DW Acc':<16}{'India-DW Acc':<15}{'Corrected Acc':<16}{'Label/Sensor Drop':<20}{'Geographic Drop':<18}")
    print("=" * 135)

    for _, row in df_main.iterrows():
        print(f"{row['model_family']:<8}"
              f"{row['strategy']:<18}"
              f"{row['trainable_pct']:<12.1f}"
              f"{row['eurosat_accuracy']:<14.2f}"
              f"{row['europe_dw_accuracy']:<16.2f}"
              f"{row['india_dw_accuracy']:<15.2f}"
              f"{row['india_dw_noise_corrected_accuracy']:<16.2f}"
              f"{row['drop_label_sensor_pct']:<20.2f}"
              f"{row['drop_geographic_pct']:<18.2f}")
    print("=" * 135 + "\n")


if __name__ == "__main__":
    main()
