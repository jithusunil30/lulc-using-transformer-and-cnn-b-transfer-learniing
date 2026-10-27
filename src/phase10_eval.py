"""
src/phase10_eval.py - Phase 10: Model Calibration, Selective Prediction, and Out-of-Domain XAI Faithfulness
===========================================================================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024), Guo et al. (ICML 2017), Geifman & El-Yaniv (NeurIPS 2017)

Phase 10 Components:
(a) Calibration:
    - Reliability diagrams and Expected Calibration Error (ECE) in-domain vs. shifted.
    - Temperature scaling fitted on the EuroSAT validation set only (minimizing NLL).
    - Uncalibrated vs. Temperature-calibrated ECE, MCE, NLL, and Brier score.
    - Coarse-class level projection (Q = P M) for apples-to-apples in-domain vs shifted comparison.
(b) Selective Prediction:
    - Accuracy-coverage curves using max-softmax confidence.
    - Area Under Risk-Coverage Curve (AURC) and Area Under Accuracy-Coverage Curve (AUACC).
    - Empirical coverage achieved at target accuracies (90%, 95%) and accuracy at fixed coverages (90%, 80%, 50%).
(c) Re-run Phase 5 XAI on Shifted Images:
    - Methods: Integrated Gradients (black baseline, blurred baseline, SmoothGrad), Grad-CAM, Attention Rollout.
    - Stratified sample of shifted images (India-DW across coarse classes, regions, and seasons) vs. in-domain EuroSAT.
    - Quantitative Deletion AUC and Insertion AUC.
    - Statistical paired tests (Wilcoxon signed-rank and paired t-test) testing whether partial fine-tuning
      changes attribution faithfulness compared to full fine-tuning.

Outputs:
- results/phase10_calibration_summary.csv
- results/phase10_selective_prediction.csv
- results/phase10_xai_shifted_faithfulness.csv
- results/phase10_summary.json
- figures/phase10_reliability_diagrams.png (DPI=200)
- figures/phase10_selective_prediction_curves.png (DPI=200)
- figures/phase10_xai_faithfulness_shift_comparison.png (DPI=200)
"""

import sys
import os
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
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import auc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Captum & Grad-CAM
from captum.attr import IntegratedGradients, NoiseTunnel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import (
    load_raw_eurosat,
    create_or_load_stratified_split,
    get_transforms,
    EuroSATWrapper,
    get_timm_mean_std,
)
from src.models import build_transformer_model, count_parameters
from src.train import clean_name
from src.evaluate_shift import (
    COARSE_CLASSES,
    COARSE_TO_IDX,
    EUROSAT_TO_COARSE_IDX,
    build_aggregation_matrix,
    GeographicShiftDataset,
    get_eval_transform,
)
from src.xai_eval import (
    get_gradcam_target_layers,
    get_gradcam_reshape_tf,
    compute_attention_rollout,
    compute_attribution,
    compute_insertion_deletion_auc,
)


# ==============================================================================
# 1. Temperature Scaling Engine (Guo et al., ICML 2017)
# ==============================================================================

class TemperatureScaler:
    """
    Fits and applies a scalar temperature parameter T > 0 on validation set logits
    by minimizing negative log-likelihood (cross-entropy loss).
    """
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Fits optimal temperature T on validation set logits using L-BFGS-B.
        logits: (N, C) numpy array
        labels: (N,) numpy array of integer labels
        """
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        from scipy.optimize import minimize_scalar

        def loss_fn(t: float) -> float:
            if t <= 0:
                return 1e9
            scaled_logits = logits_tensor / float(t)
            loss = criterion(scaled_logits, labels_tensor)
            return float(loss.item())

        res = minimize_scalar(
            loss_fn,
            bounds=(0.05, 10.0),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 100},
        )

        self.temperature = float(res.x)
        return self.temperature

    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    def predict_probs(self, logits: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
        t = temperature if temperature is not None else self.temperature
        scaled = logits / t
        # Stable softmax
        exp_logits = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


# ==============================================================================
# 2. Calibration Metrics & Reliability Diagram Calculator
# ==============================================================================

def compute_calibration_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    Negative Log-Likelihood (NLL), and Brier Score, alongside per-bin statistics.
    """
    n_samples, n_classes = probs.shape
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(float)

    # NLL
    eps = 1e-12
    clipped_probs = np.clip(probs, eps, 1.0)
    target_probs = clipped_probs[np.arange(n_samples), targets]
    nll = float(-np.mean(np.log(target_probs)))

    # Brier score
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), targets] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Equal-width bins partition [0, 1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)

            diff = abs(bin_acc - bin_conf)
            ece += (count / n_samples) * diff
            if diff > mce:
                mce = diff
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((low + high) / 2.0))

    return {
        "accuracy": float(np.mean(accuracies) * 100.0),
        "ece": float(ece),
        "mce": float(mce),
        "nll": float(nll),
        "brier": float(brier),
        "avg_confidence": float(np.mean(confidences)),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges.tolist(),
    }


# ==============================================================================
# 3. Selective Prediction Engine (Geifman & El-Yaniv, 2017)
# ==============================================================================

def compute_selective_prediction(
    probs: np.ndarray,
    targets: np.ndarray,
    n_points: int = 101,
) -> Dict[str, Any]:
    """
    Computes accuracy-coverage curves and selective risk metrics using max-softmax confidence.
    """
    n_samples = len(targets)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == targets).astype(float)

    # Sort descending by confidence
    sorted_indices = np.argsort(-confidences)
    sorted_correctness = correctness[sorted_indices]
    sorted_confidences = confidences[sorted_indices]

    coverages = np.linspace(0.01, 1.0, n_points)
    accuracies = []
    risks = []

    for cov in coverages:
        k = max(1, int(np.ceil(cov * n_samples)))
        sel_acc = float(np.mean(sorted_correctness[:k]))
        accuracies.append(sel_acc)
        risks.append(1.0 - sel_acc)

    # Area Under Risk-Coverage Curve (AURC)
    aurc = float(auc(coverages, risks))
    auacc = float(auc(coverages, accuracies))

    # Optimal theoretical AURC (if confident samples were perfectly sorted)
    n_correct = int(np.sum(correctness))
    opt_risks = []
    for cov in coverages:
        k = max(1, int(np.ceil(cov * n_samples)))
        correct_in_k = min(k, n_correct)
        opt_risks.append(1.0 - (correct_in_k / k))
    opt_aurc = float(auc(coverages, opt_risks))
    excess_aurc = max(0.0, aurc - opt_aurc)

    # Key threshold evaluations
    def acc_at_coverage(target_cov: float) -> float:
        k = max(1, int(np.ceil(target_cov * n_samples)))
        return float(np.mean(sorted_correctness[:k]) * 100.0)

    def coverage_at_acc(target_acc: float) -> Optional[float]:
        # Search from highest coverage to lowest
        for cov, acc in zip(reversed(coverages), reversed(accuracies)):
            if acc >= target_acc:
                return float(cov * 100.0)
        return None

    return {
        "aurc": aurc,
        "auacc": auacc,
        "excess_aurc": excess_aurc,
        "acc_cov_100": acc_at_coverage(1.0),
        "acc_cov_90": acc_at_coverage(0.90),
        "acc_cov_80": acc_at_coverage(0.80),
        "acc_cov_50": acc_at_coverage(0.50),
        "cov_for_90acc": coverage_at_acc(0.90),
        "cov_for_95acc": coverage_at_acc(0.95),
        "coverages": coverages.tolist(),
        "accuracies": accuracies,
        "risks": risks,
    }


# ==============================================================================
# 4. Feature / Logit Extractor
# ==============================================================================

def extract_model_logits(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_shift_dataset: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts raw model output logits and targets from a DataLoader.
    """
    model.eval()
    all_logits = []
    all_targets = []
    metadata = []

    with torch.no_grad():
        for batch in dataloader:
            if is_shift_dataset:
                images, labels, regions, seasons, patch_ids = batch
                for r, s, p in zip(regions, seasons, patch_ids):
                    metadata.append({"region": r, "season": s, "patch_id": p})
            else:
                images, labels, paths = batch
                for p in paths:
                    metadata.append({"path": p})

            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    return logits_arr, targets_arr, metadata


# ==============================================================================
# 5. Stratified Shift Sample Selector for XAI Faithfulness Evaluation
# ==============================================================================

def select_stratified_shift_xai_sample(
    df_shift: pd.DataFrame,
    n_per_class: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Selects a balanced sample of shifted images (100 total: 20 per coarse class)
    across India-DW regions and seasons.
    """
    rng = random.Random(seed)
    selected = []

    for c_name in COARSE_CLASSES:
        cls_df = df_shift[df_shift["coarse_class"] == c_name]
        if len(cls_df) == 0:
            continue
        sample_size = min(n_per_class, len(cls_df))
        sampled_indices = rng.sample(list(cls_df.index), sample_size)
        for idx in sampled_indices:
            row = cls_df.loc[idx]
            selected.append({
                "patch_id": str(row["patch_id"]),
                "file_path": str(PROJECT_ROOT / row["file_path"]),
                "coarse_class": str(row["coarse_class"]),
                "coarse_label": COARSE_TO_IDX[row["coarse_class"]],
                "region": str(row["region"]),
                "season": str(row["season"]),
            })

    print(f"[Phase 10 XAI] Selected {len(selected)} stratified shifted samples ({n_per_class} per coarse class across 5 classes).")
    return selected


# ==============================================================================
# 6. Evaluation Pipelines: Calibration, Selective Risk, and XAI Faithfulness
# ==============================================================================

def run_phase10_evaluation(
    device_str: str = cfg.DEVICE,
    batch_size: int = cfg.BATCH_SIZE,
    seed: int = cfg.PRIMARY_SEED,
    xai_sample_per_class: int = 20,
):
    print("=" * 80)
    print("PHASE 10: Model Calibration, Selective Prediction, and Shifted XAI Faithfulness")
    print("=" * 80)

    device = torch.device(device_str)
    print(f"[Phase 10] Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load EuroSAT Raw Dataset & Splits
    raw_eurosat = load_raw_eurosat(cfg.DATA_DIR)
    splits = create_or_load_stratified_split(raw_eurosat, seed=seed)

    # Load Geographic Shift Dataset (Metadata)
    shift_meta_path = cfg.DATA_DIR / "geographic_shift" / "metadata.csv"
    if not shift_meta_path.exists():
        raise FileNotFoundError(f"Geographic shift metadata not found at {shift_meta_path}. Run Phase 8 first.")
    df_shift = pd.read_csv(shift_meta_path)

    # Isolate India-DW (Shift target) and Europe-DW (Control)
    india_regions = ["tamil_nadu", "punjab", "rajasthan", "kerala"]
    europe_regions = ["central_europe", "western_europe", "southern_europe"]
    df_india = df_shift[df_shift["region"].isin(india_regions)].copy()
    df_europe = df_shift[df_shift["region"].isin(europe_regions)].copy()

    # Model targets: 3 architectures x 5 strategies
    model_families = ["vit", "deit", "swin"]
    strategies = ["scratch", "linear_probe", "unfreeze_last_2", "unfreeze_last_3", "full_ft"]

    # Storage structures
    calibration_records = []
    selective_records = []
    reliability_diagram_data = {}
    selective_curves_data = {}

    M_proj = build_aggregation_matrix(device).cpu().numpy()  # 10x5 projection matrix

    print("\n--- Running Part (a) Calibration & Part (b) Selective Prediction ---")

    for family in model_families:
        model_name = cfg.get_model_name(family)
        clean_m = clean_name(model_name)
        eval_tf = get_eval_transform(model_name)

        # EuroSAT Validation and Test Loaders
        val_dataset = EuroSATWrapper(raw_eurosat, splits["val"], transform=eval_tf)
        test_dataset = EuroSATWrapper(raw_eurosat, splits["test"], transform=eval_tf)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)

        # Geographic Shift Loaders
        india_dataset = GeographicShiftDataset(df_india, transform=eval_tf)
        india_loader = DataLoader(india_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)

        europe_dataset = GeographicShiftDataset(df_europe, transform=eval_tf)
        europe_loader = DataLoader(europe_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)

        for strategy in strategies:
            ckpt_name = f"{clean_m}_{strategy}_seed{seed}_best.pth"
            ckpt_path = cfg.CHECKPOINTS_DIR / ckpt_name

            if not ckpt_path.exists():
                print(f"[Phase 10] Warning: Checkpoint {ckpt_name} not found. Skipping.")
                continue

            print(f"\n[Evaluating Model] {family.upper()} | Strategy: `{strategy}` | Checkpoint: {ckpt_name}")

            # Build and load model (pretrained=False prevents redundant HF Hub downloads)
            model, _ = build_transformer_model(model_name, strategy=strategy, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model = model.to(device).eval()

            # 1. Extract validation logits to fit Temperature Scaling (EuroSAT Val only)
            val_logits, val_targets_10, _ = extract_model_logits(model, val_loader, device)

            scaler = TemperatureScaler()
            optimal_temp = scaler.fit(val_logits, val_targets_10)
            print(f"  --> Fitted Temperature T* on EuroSAT Val: {optimal_temp:.4f}")

            # 2. Extract Test Logits (In-Domain EuroSAT Test)
            test_logits, test_targets_10, _ = extract_model_logits(model, test_loader, device)
            test_targets_coarse = np.array([EUROSAT_TO_COARSE_IDX[int(t)] for t in test_targets_10])

            # In-Domain Probabilities (10-class)
            test_probs_10_uncal = scaler.predict_probs(test_logits, temperature=1.0)
            test_probs_10_cal = scaler.predict_probs(test_logits, temperature=optimal_temp)

            # In-Domain Coarse Probabilities (5-class projection: Q = P @ M)
            test_probs_coarse_uncal = test_probs_10_uncal @ M_proj
            test_probs_coarse_cal = test_probs_10_cal @ M_proj

            # 3. Extract Shifted Logits (India-DW Target)
            india_logits, india_targets_coarse, _ = extract_model_logits(model, india_loader, device, is_shift_dataset=True)
            india_probs_10_uncal = scaler.predict_probs(india_logits, temperature=1.0)
            india_probs_10_cal = scaler.predict_probs(india_logits, temperature=optimal_temp)
            india_probs_coarse_uncal = india_probs_10_uncal @ M_proj
            india_probs_coarse_cal = india_probs_10_cal @ M_proj

            # 4. Extract Europe-DW Logits (Control)
            europe_logits, europe_targets_coarse, _ = extract_model_logits(model, europe_loader, device, is_shift_dataset=True)
            europe_probs_10_uncal = scaler.predict_probs(europe_logits, temperature=1.0)
            europe_probs_10_cal = scaler.predict_probs(europe_logits, temperature=optimal_temp)
            europe_probs_coarse_uncal = europe_probs_10_uncal @ M_proj
            europe_probs_coarse_cal = europe_probs_10_cal @ M_proj

            # ------------------------------------------------------------------
            # Compute Calibration Metrics
            # ------------------------------------------------------------------
            # A. In-Domain EuroSAT Test (Coarse-level for direct comparison)
            cal_es_uncal = compute_calibration_metrics(test_probs_coarse_uncal, test_targets_coarse)
            cal_es_cal = compute_calibration_metrics(test_probs_coarse_cal, test_targets_coarse)

            # B. In-Domain EuroSAT Test (10-class standard)
            cal_es_10_uncal = compute_calibration_metrics(test_probs_10_uncal, test_targets_10)
            cal_es_10_cal = compute_calibration_metrics(test_probs_10_cal, test_targets_10)

            # C. Shifted India-DW (Coarse-level)
            cal_in_uncal = compute_calibration_metrics(india_probs_coarse_uncal, india_targets_coarse)
            cal_in_cal = compute_calibration_metrics(india_probs_coarse_cal, india_targets_coarse)

            # D. Europe-DW Control (Coarse-level)
            cal_eu_uncal = compute_calibration_metrics(europe_probs_coarse_uncal, europe_targets_coarse)
            cal_eu_cal = compute_calibration_metrics(europe_probs_coarse_cal, europe_targets_coarse)

            # Record Calibration Summary
            trainable_params, total_params, trainable_pct = count_parameters(model)
            cal_rec = {
                "model_family": family.upper(),
                "model_arch": model_name,
                "strategy": strategy,
                "seed": seed,
                "trainable_params": trainable_params,
                "trainable_pct": round(trainable_pct, 2),
                "optimal_temperature": round(optimal_temp, 4),
                # EuroSAT Coarse
                "eurosat_coarse_acc": round(cal_es_uncal["accuracy"], 2),
                "eurosat_coarse_ece_uncal": round(cal_es_uncal["ece"], 4),
                "eurosat_coarse_ece_cal": round(cal_es_cal["ece"], 4),
                "eurosat_coarse_nll_uncal": round(cal_es_uncal["nll"], 4),
                "eurosat_coarse_nll_cal": round(cal_es_cal["nll"], 4),
                "eurosat_coarse_brier_uncal": round(cal_es_uncal["brier"], 4),
                "eurosat_coarse_brier_cal": round(cal_es_cal["brier"], 4),
                # EuroSAT 10-Class
                "eurosat_10class_acc": round(cal_es_10_uncal["accuracy"], 2),
                "eurosat_10class_ece_uncal": round(cal_es_10_uncal["ece"], 4),
                "eurosat_10class_ece_cal": round(cal_es_10_cal["ece"], 4),
                # Europe-DW Control
                "europe_dw_acc": round(cal_eu_uncal["accuracy"], 2),
                "europe_dw_ece_uncal": round(cal_eu_uncal["ece"], 4),
                "europe_dw_ece_cal": round(cal_eu_cal["ece"], 4),
                # India-DW Shift
                "india_dw_acc": round(cal_in_uncal["accuracy"], 2),
                "india_dw_ece_uncal": round(cal_in_uncal["ece"], 4),
                "india_dw_ece_cal": round(cal_in_cal["ece"], 4),
                "india_dw_nll_uncal": round(cal_in_uncal["nll"], 4),
                "india_dw_nll_cal": round(cal_in_cal["nll"], 4),
                "india_dw_brier_uncal": round(cal_in_uncal["brier"], 4),
                "india_dw_brier_cal": round(cal_in_cal["brier"], 4),
                # ECE Deltas
                "ece_improvement_in_domain": round(cal_es_uncal["ece"] - cal_es_cal["ece"], 4),
                "ece_delta_under_shift": round(cal_in_uncal["ece"] - cal_in_cal["ece"], 4),
            }
            calibration_records.append(cal_rec)

            model_key = f"{family.upper()}-{strategy}"
            reliability_diagram_data[model_key] = {
                "in_domain_uncal": cal_es_uncal,
                "in_domain_cal": cal_es_cal,
                "shift_uncal": cal_in_uncal,
                "shift_cal": cal_in_cal,
                "optimal_temp": optimal_temp,
            }

            # ------------------------------------------------------------------
            # Compute Selective Prediction (Risk-Coverage)
            # ------------------------------------------------------------------
            sel_es = compute_selective_prediction(test_probs_coarse_cal, test_targets_coarse)
            sel_in = compute_selective_prediction(india_probs_coarse_cal, india_targets_coarse)

            sel_rec = {
                "model_family": family.upper(),
                "model_arch": model_name,
                "strategy": strategy,
                "seed": seed,
                # In-domain EuroSAT
                "in_domain_aurc": round(sel_es["aurc"], 4),
                "in_domain_auacc": round(sel_es["auacc"], 4),
                "in_domain_acc_cov100": round(sel_es["acc_cov_100"], 2),
                "in_domain_acc_cov90": round(sel_es["acc_cov_90"], 2),
                "in_domain_acc_cov80": round(sel_es["acc_cov_80"], 2),
                "in_domain_acc_cov50": round(sel_es["acc_cov_50"], 2),
                "in_domain_cov_at_95acc": round(sel_es["cov_for_95acc"], 2) if sel_es["cov_for_95acc"] else None,
                # Shifted India-DW
                "shifted_aurc": round(sel_in["aurc"], 4),
                "shifted_auacc": round(sel_in["auacc"], 4),
                "shifted_acc_cov100": round(sel_in["acc_cov_100"], 2),
                "shifted_acc_cov90": round(sel_in["acc_cov_90"], 2),
                "shifted_acc_cov80": round(sel_in["acc_cov_80"], 2),
                "shifted_acc_cov50": round(sel_in["acc_cov_50"], 2),
                "shifted_cov_at_90acc": round(sel_in["cov_for_90acc"], 2) if sel_in["cov_for_90acc"] else None,
                "shifted_cov_at_95acc": round(sel_in["cov_for_95acc"], 2) if sel_in["cov_for_95acc"] else None,
            }
            selective_records.append(sel_rec)

            selective_curves_data[model_key] = {
                "in_domain": sel_es,
                "shifted": sel_in,
            }

    # Save Calibration & Selective Risk CSVs
    df_cal = pd.DataFrame(calibration_records)
    cal_csv_path = cfg.RESULTS_DIR / "phase10_calibration_summary.csv"
    df_cal.to_csv(cal_csv_path, index=False)
    print(f"\n[Saved] Calibration summary exported to: {cal_csv_path}")

    df_sel = pd.DataFrame(selective_records)
    sel_csv_path = cfg.RESULTS_DIR / "phase10_selective_prediction.csv"
    df_sel.to_csv(sel_csv_path, index=False)
    print(f"[Saved] Selective prediction metrics exported to: {sel_csv_path}")

    # ==========================================================================
    # Part (c): Re-Run XAI Evaluation on Shifted Images
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"--- Running Part (c) XAI Evaluation on Shifted Images vs. In-Domain ---")
    print("=" * 80)

    # 1. Select Stratified Shifted Sample (100 images: 20 per coarse class)
    shift_samples = select_stratified_shift_xai_sample(df_india, n_per_class=xai_sample_per_class, seed=seed)

    # 2. Select Matched In-Domain Sample (matched sample size across 5 coarse classes)
    in_domain_indices = splits["test"]
    raw_targets = np.array([raw_eurosat.targets[idx] for idx in in_domain_indices])
    rng = random.Random(seed)
    indomain_samples = []
    n_per_eurosat_cls = max(1, xai_sample_per_class // 2)
    for cls_idx in range(cfg.NUM_CLASSES):
        cls_mask = (raw_targets == cls_idx)
        matching = [in_domain_indices[i] for i in np.where(cls_mask)[0]]
        picked = rng.sample(matching, min(n_per_eurosat_cls, len(matching)))
        for idx in picked:
            p, lbl = raw_eurosat.samples[idx]
            indomain_samples.append({
                "patch_id": Path(p).name,
                "file_path": str(p),
                "eurosat_class": cfg.CLASS_NAMES[lbl],
                "eurosat_label": int(lbl),
                "coarse_class": COARSE_CLASSES[EUROSAT_TO_COARSE_IDX[int(lbl)]],
                "coarse_label": EUROSAT_TO_COARSE_IDX[int(lbl)],
            })

    # XAI Targets to evaluate: ViT, DeiT, Swin across 4 strategies:
    # `unfreeze_last_3` (best partial FT), `full_ft`, `scratch`, `linear_probe`
    xai_strategies = ["unfreeze_last_3", "full_ft", "scratch", "linear_probe"]
    xai_methods = ["ig_black", "ig_blur", "ig_smoothgrad", "grad_cam", "rollout"]

    xai_records = []
    pairwise_records = []

    for family in model_families:
        model_name = cfg.get_model_name(family)
        clean_m = clean_name(model_name)
        _, eval_tf, _ = get_transforms(model_name=model_name)

        for strategy in xai_strategies:
            ckpt_name = f"{clean_m}_{strategy}_seed{seed}_best.pth"
            ckpt_path = cfg.CHECKPOINTS_DIR / ckpt_name
            if not ckpt_path.exists():
                continue

            print(f"\n[XAI Evaluation] Model: {family.upper()} | Strategy: `{strategy}`")
            model, _ = build_transformer_model(model_name, strategy=strategy, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model = model.to(device).eval()

            available_methods = [m for m in xai_methods if not (m == "rollout" and family.lower() == "swin")]

            for method in available_methods:
                del_in_domain = []
                ins_in_domain = []
                del_shifted = []
                ins_shifted = []

                # A. Evaluate on In-Domain Samples
                for item in tqdm(indomain_samples, desc=f"{family}-{strategy} {method} [In-Domain]", leave=False):
                    pil_img = Image.open(item["file_path"]).convert("RGB")
                    t_img = eval_tf(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(t_img)
                        pred_class = int(torch.argmax(logits, dim=1).item())

                    try:
                        attr = compute_attribution(method, model, t_img[0], pred_class, family, device)
                        del_auc, ins_auc, _, _ = compute_insertion_deletion_auc(
                            model, t_img[0], pred_class, attr, device, n_steps=10
                        )
                        del_in_domain.append(del_auc)
                        ins_in_domain.append(ins_auc)
                    except Exception as e:
                        continue

                # B. Evaluate on Shifted Samples
                for item in tqdm(shift_samples, desc=f"{family}-{strategy} {method} [Shifted]", leave=False):
                    pil_img = Image.open(item["file_path"]).convert("RGB")
                    t_img = eval_tf(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(t_img)
                        pred_class = int(torch.argmax(logits, dim=1).item())

                    try:
                        attr = compute_attribution(method, model, t_img[0], pred_class, family, device)
                        del_auc, ins_auc, _, _ = compute_insertion_deletion_auc(
                            model, t_img[0], pred_class, attr, device, n_steps=10
                        )
                        del_shifted.append(del_auc)
                        ins_shifted.append(ins_auc)
                    except Exception as e:
                        continue

                if len(del_in_domain) > 0 and len(del_shifted) > 0:
                    mean_del_id = float(np.mean(del_in_domain))
                    mean_ins_id = float(np.mean(ins_in_domain))
                    mean_del_sh = float(np.mean(del_shifted))
                    mean_ins_sh = float(np.mean(ins_shifted))

                    # Statistical test between in-domain and shifted faithfulness
                    stat_del, p_del = stats.mannwhitneyu(del_in_domain, del_shifted, alternative="two-sided")
                    stat_ins, p_ins = stats.mannwhitneyu(ins_in_domain, ins_shifted, alternative="two-sided")

                    xai_records.append({
                        "model_family": family.upper(),
                        "strategy": strategy,
                        "xai_method": method,
                        "n_samples_indomain": len(del_in_domain),
                        "n_samples_shifted": len(del_shifted),
                        "deletion_auc_indomain": round(mean_del_id, 4),
                        "deletion_auc_shifted": round(mean_del_sh, 4),
                        "deletion_auc_delta": round(mean_del_sh - mean_del_id, 4),
                        "deletion_shift_p_val": round(float(p_del), 4),
                        "insertion_auc_indomain": round(mean_ins_id, 4),
                        "insertion_auc_shifted": round(mean_ins_sh, 4),
                        "insertion_auc_delta": round(mean_ins_sh - mean_ins_id, 4),
                        "insertion_shift_p_val": round(float(p_ins), 4),
                        "raw_del_indomain": del_in_domain,
                        "raw_ins_indomain": ins_in_domain,
                        "raw_del_shifted": del_shifted,
                        "raw_ins_shifted": ins_shifted,
                    })

    # Save XAI CSV (without raw arrays)
    df_xai = pd.DataFrame(xai_records)
    clean_df_xai = df_xai.drop(columns=["raw_del_indomain", "raw_ins_indomain", "raw_del_shifted", "raw_ins_shifted"], errors="ignore")
    xai_csv_path = cfg.RESULTS_DIR / "phase10_xai_shifted_faithfulness.csv"
    clean_df_xai.to_csv(xai_csv_path, index=False)
    print(f"\n[Saved] XAI shifted faithfulness exported to: {xai_csv_path}")

    # --------------------------------------------------------------------------
    # Paired Statistical Tests: unfreeze_last_3 vs. full_ft on Shifted Images
    # --------------------------------------------------------------------------
    print("\n--- Testing Whether Partial Fine-Tuning Changes Attribution Faithfulness ---")
    strategy_tests = []
    for family in model_families:
        rec_u3 = [r for r in xai_records if r["model_family"] == family.upper() and r["strategy"] == "unfreeze_last_3"]
        rec_fft = [r for r in xai_records if r["model_family"] == family.upper() and r["strategy"] == "full_ft"]

        for u3_item in rec_u3:
            m = u3_item["xai_method"]
            fft_item = next((f for f in rec_fft if f["xai_method"] == m), None)
            if not fft_item:
                continue

            del_u3 = np.array(u3_item["raw_del_shifted"])
            del_fft = np.array(fft_item["raw_del_shifted"])
            ins_u3 = np.array(u3_item["raw_ins_shifted"])
            ins_fft = np.array(fft_item["raw_ins_shifted"])

            min_len = min(len(del_u3), len(del_fft))
            del_u3, del_fft = del_u3[:min_len], del_fft[:min_len]
            ins_u3, ins_fft = ins_u3[:min_len], ins_fft[:min_len]

            # Wilcoxon signed rank test
            w_del, p_del = stats.wilcoxon(del_u3, del_fft) if not np.all(del_u3 == del_fft) else (0.0, 1.0)
            w_ins, p_ins = stats.wilcoxon(ins_u3, ins_fft) if not np.all(ins_u3 == ins_fft) else (0.0, 1.0)

            # Cohen's d effect size
            def cohen_d(x, y):
                nx, ny = len(x), len(y)
                s_pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
                return float((np.mean(x) - np.mean(y)) / (s_pooled + 1e-8))

            d_del = cohen_d(del_u3, del_fft)
            d_ins = cohen_d(ins_u3, ins_fft)

            strategy_tests.append({
                "model_family": family.upper(),
                "xai_method": m,
                "del_u3_shifted": round(float(np.mean(del_u3)), 4),
                "del_fft_shifted": round(float(np.mean(del_fft)), 4),
                "del_diff (u3 - fft)": round(float(np.mean(del_u3) - np.mean(del_fft)), 4),
                "del_p_val": round(float(p_del), 4),
                "del_cohen_d": round(d_del, 3),
                "ins_u3_shifted": round(float(np.mean(ins_u3)), 4),
                "ins_fft_shifted": round(float(np.mean(ins_fft)), 4),
                "ins_diff (u3 - fft)": round(float(np.mean(ins_u3) - np.mean(ins_fft)), 4),
                "ins_p_val": round(float(p_ins), 4),
                "ins_cohen_d": round(d_ins, 3),
                "faithfulness_verdict": (
                    "U3 Significantly More Faithful" if (np.mean(del_u3) < np.mean(del_fft) and p_del < 0.05) or (np.mean(ins_u3) > np.mean(ins_fft) and p_ins < 0.05)
                    else ("Full-FT Significantly More Faithful" if (np.mean(del_fft) < np.mean(del_u3) and p_del < 0.05) or (np.mean(ins_fft) > np.mean(ins_u3) and p_ins < 0.05)
                    else "Statistically Equivalent")
                )
            })

    df_strat_tests = pd.DataFrame(strategy_tests)
    print("\n--- Strategy Faithfulness Comparison on Shifted Images (unfreeze_last_3 vs. full_ft) ---")
    print(df_strat_tests[["model_family", "xai_method", "del_u3_shifted", "del_fft_shifted", "del_p_val", "ins_u3_shifted", "ins_fft_shifted", "ins_p_val", "faithfulness_verdict"]].to_string())

    # ==========================================================================
    # Visualizations
    # ==========================================================================
    print("\n--- Generating Publication-Quality Figures (DPI=200) ---")

    # 1. Reliability Diagrams
    plot_reliability_diagrams(reliability_diagram_data, cfg.FIGURES_DIR / "phase10_reliability_diagrams.png")

    # 2. Selective Prediction Curves
    plot_selective_prediction_curves(selective_curves_data, cfg.FIGURES_DIR / "phase10_selective_prediction_curves.png")

    # 3. XAI Faithfulness Shift Comparison
    plot_xai_faithfulness_comparison(clean_df_xai, cfg.FIGURES_DIR / "phase10_xai_faithfulness_shift_comparison.png")

    # Save summary JSON
    summary_json_path = cfg.RESULTS_DIR / "phase10_summary.json"
    summary_data = {
        "phase": 10,
        "calibration_records_count": len(calibration_records),
        "selective_records_count": len(selective_records),
        "xai_records_count": len(clean_df_xai),
        "calibration_top_performers": df_cal.sort_values(by="india_dw_ece_cal").head(5).to_dict(orient="records"),
        "selective_top_performers": df_sel.sort_values(by="shifted_aurc").head(5).to_dict(orient="records"),
        "strategy_faithfulness_tests": strategy_tests,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Saved] Phase 10 summary JSON saved to: {summary_json_path}")

    print("\n[Phase 10] Complete execution finished successfully!")


# ==============================================================================
# 7. Visualization Plotting Functions
# ==============================================================================

def plot_reliability_diagrams(
    diagram_data: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    """
    Plots a multi-panel reliability diagram (10 bins) comparing:
    - Top row: In-Domain EuroSAT Test (Uncalibrated vs. Temperature Scaled)
    - Bottom row: Shifted India-DW (Uncalibrated vs. In-Domain Temperature Scaled)
    For Swin unfreeze_last_3, Swin full_ft, ViT unfreeze_last_3, and ViT full_ft.
    """
    models_to_plot = ["SWIN-unfreeze_last_3", "SWIN-full_ft", "VIT-unfreeze_last_3", "VIT-full_ft"]
    models_to_plot = [m for m in models_to_plot if m in diagram_data]

    if not models_to_plot:
        return

    n_models = len(models_to_plot)
    fig, axes = plt.subplots(4, n_models, figsize=(4.5 * n_models, 15))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=1)

    bin_centers = np.linspace(0.05, 0.95, 10)
    width = 0.08

    row_titles = [
        "In-Domain EuroSAT (Uncalibrated, T=1.0)",
        "In-Domain EuroSAT (Calibrated, T=T*)",
        "Shifted India-DW (Uncalibrated, T=1.0)",
        "Shifted India-DW (Calibrated with EuroSAT T*)",
    ]

    for col_idx, m_key in enumerate(models_to_plot):
        data = diagram_data[m_key]
        t_star = data["optimal_temp"]
        subsets = [
            (data["in_domain_uncal"], "#1f77b4", "In-Domain Uncal"),
            (data["in_domain_cal"], "#2ca02c", f"In-Domain Cal (T={t_star:.2f})"),
            (data["shift_uncal"], "#d62728", "Shift Uncal"),
            (data["shift_cal"], "#ff7f0e", f"Shift Cal (T={t_star:.2f})"),
        ]

        for row_idx, (subset_data, color, label) in enumerate(subsets):
            ax = axes[row_idx, col_idx]

            bin_edges = subset_data.get("bin_edges", np.linspace(0.0, 1.0, 11).tolist())
            n_b = len(bin_edges) - 1
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2.0 for i in range(n_b)]
            bar_w = [(bin_edges[i+1] - bin_edges[i]) * 0.82 for i in range(n_b)]

            accs = subset_data["bin_accuracies"][:n_b]
            confs = subset_data["bin_confidences"][:n_b]
            counts = subset_data["bin_counts"][:n_b]

            # Diagonal perfect calibration
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Perfect Calibration")

            # Calibration bars
            ax.bar(bin_centers, accs, width=bar_w, color=color, alpha=0.75, edgecolor="black", label="Outputs")

            # Gap to diagonal
            for bc, acc in zip(bin_centers, accs):
                if acc > 0:
                    ax.plot([bc, bc], [acc, bc], color="red", linewidth=1.8, alpha=0.8)

            ece = subset_data["ece"]
            mce = subset_data["mce"]
            nll = subset_data["nll"]

            ax.text(
                0.05, 0.78,
                f"ECE = {ece:.3f}\nMCE = {mce:.3f}\nNLL = {nll:.3f}",
                transform=ax.transAxes,
                fontsize=9.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"),
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=":", alpha=0.5)

            if row_idx == 0:
                ax.set_title(f"{m_key}", fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{row_titles[row_idx]}\n\nObserved Accuracy", fontsize=10, fontweight="bold")
            if row_idx == 3:
                ax.set_xlabel("Confidence", fontsize=10, fontweight="bold")

    plt.suptitle("Phase 10: Reliability Diagrams & Expected Calibration Error (In-Domain vs. Shifted)\n"
                 "(Temperature Scaling Fitted Exclusively on EuroSAT Validation Set)",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[Figure] Saved reliability diagrams to: {output_path} (DPI={cfg.DPI})")


def plot_selective_prediction_curves(
    curves_data: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    """
    Plots Accuracy-Coverage Curves:
    - Panel A: In-Domain EuroSAT (comparing strategies)
    - Panel B: Shifted India-DW (comparing strategies)
    - Panel C: Cross-domain comparison for top architectures
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    strat_colors = {
        "scratch": "#7f7f7f",
        "linear_probe": "#17becf",
        "unfreeze_last_2": "#ff7f0e",
        "unfreeze_last_3": "#2ca02c",
        "full_ft": "#d62728",
    }
    strat_labels = {
        "scratch": "Scratch",
        "linear_probe": "Linear Probe",
        "unfreeze_last_2": "Unfreeze Last 2",
        "unfreeze_last_3": "Unfreeze Last 3 (Best Partial)",
        "full_ft": "Full Fine-Tuning",
    }

    # Panel A: In-Domain EuroSAT (Swin)
    for strat in ["scratch", "linear_probe", "unfreeze_last_2", "unfreeze_last_3", "full_ft"]:
        k = f"SWIN-{strat}"
        if k in curves_data:
            covs = curves_data[k]["in_domain"]["coverages"]
            accs = [a * 100.0 for a in curves_data[k]["in_domain"]["accuracies"]]
            aurc = curves_data[k]["in_domain"]["aurc"]
            ax1.plot(covs, accs, label=f"{strat_labels[strat]} (AURC={aurc:.3f})", color=strat_colors[strat], linewidth=2.2)

    ax1.set_title("(A) In-Domain EuroSAT: Selective Accuracy vs. Coverage", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Coverage (Fraction of Retained Predictions)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Selective Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0.0, 1.02)
    ax1.set_ylim(80, 101)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9, loc="lower left")

    # Panel B: Shifted India-DW (Swin)
    for strat in ["scratch", "linear_probe", "unfreeze_last_2", "unfreeze_last_3", "full_ft"]:
        k = f"SWIN-{strat}"
        if k in curves_data:
            covs = curves_data[k]["shifted"]["coverages"]
            accs = [a * 100.0 for a in curves_data[k]["shifted"]["accuracies"]]
            aurc = curves_data[k]["shifted"]["aurc"]
            ax2.plot(covs, accs, label=f"{strat_labels[strat]} (AURC={aurc:.3f})", color=strat_colors[strat], linewidth=2.2)

    ax2.set_title("(B) Shifted India-DW: Selective Accuracy vs. Coverage", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Coverage (Fraction of Retained Predictions)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Selective Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_xlim(0.0, 1.02)
    ax2.set_ylim(30, 101)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9, loc="lower left")

    # Panel C: Direct Comparison of Key Models In-Domain vs. Shifted
    key_pairs = [
        ("SWIN-unfreeze_last_3", "#2ca02c", "Swin U3"),
        ("SWIN-full_ft", "#d62728", "Swin Full-FT"),
        ("VIT-unfreeze_last_3", "#1f77b4", "ViT U3"),
        ("VIT-full_ft", "#9467bd", "ViT Full-FT"),
    ]

    for k, c, lbl in key_pairs:
        if k in curves_data:
            covs = curves_data[k]["in_domain"]["coverages"]
            acc_id = [a * 100.0 for a in curves_data[k]["in_domain"]["accuracies"]]
            acc_sh = [a * 100.0 for a in curves_data[k]["shifted"]["accuracies"]]

            ax3.plot(covs, acc_id, linestyle="--", color=c, alpha=0.7, label=f"{lbl} (In-Domain)")
            ax3.plot(covs, acc_sh, linestyle="-", color=c, linewidth=2.4, label=f"{lbl} (Shifted)")

    ax3.set_title("(C) In-Domain (Dashed) vs. Shifted (Solid)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Coverage", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Selective Accuracy (%)", fontsize=10, fontweight="bold")
    ax3.set_xlim(0.0, 1.02)
    ax3.set_ylim(25, 101)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(frameon=True, fontsize=8.5, loc="lower left")

    plt.suptitle("Selective Prediction: Accuracy-Coverage Trade-offs Under Max-Softmax Confidence",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[Figure] Saved selective prediction curves to: {output_path} (DPI={cfg.DPI})")


def plot_xai_faithfulness_comparison(
    df_xai: pd.DataFrame,
    output_path: Path,
):
    """
    Plots Deletion AUC and Insertion AUC comparing In-Domain vs. Shifted
    across XAI methods and fine-tuning strategies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    methods = df_xai["xai_method"].unique()
    x = np.arange(len(methods))
    width = 0.20

    # Aggregate by strategy
    df_u3 = df_xai[df_xai["strategy"] == "unfreeze_last_3"].groupby("xai_method").mean(numeric_only=True).reindex(methods)
    df_fft = df_xai[df_xai["strategy"] == "full_ft"].groupby("xai_method").mean(numeric_only=True).reindex(methods)

    # Panel 1: Deletion AUC (Lower = More Faithful)
    ax1.bar(x - width*1.5, df_u3["deletion_auc_indomain"], width, label="U3 (In-Domain)", color="#2ca02c", alpha=0.7)
    ax1.bar(x - width*0.5, df_u3["deletion_auc_shifted"], width, label="U3 (Shifted)", color="#2ca02c", edgecolor="black", hatch="//")
    ax1.bar(x + width*0.5, df_fft["deletion_auc_indomain"], width, label="Full-FT (In-Domain)", color="#d62728", alpha=0.7)
    ax1.bar(x + width*1.5, df_fft["deletion_auc_shifted"], width, label="Full-FT (Shifted)", color="#d62728", edgecolor="black", hatch="//")

    ax1.set_title("Deletion AUC (Lower AUC = More Faithful)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Deletion AUC Score", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax1.set_ylim(0.0, 0.65)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    # Panel 2: Insertion AUC (Higher = More Faithful)
    ax2.bar(x - width*1.5, df_u3["insertion_auc_indomain"], width, label="U3 (In-Domain)", color="#2ca02c", alpha=0.7)
    ax2.bar(x - width*0.5, df_u3["insertion_auc_shifted"], width, label="U3 (Shifted)", color="#2ca02c", edgecolor="black", hatch="//")
    ax2.bar(x + width*0.5, df_fft["insertion_auc_indomain"], width, label="Full-FT (In-Domain)", color="#d62728", alpha=0.7)
    ax2.bar(x + width*1.5, df_fft["insertion_auc_shifted"], width, label="Full-FT (Shifted)", color="#d62728", edgecolor="black", hatch="//")

    ax2.set_title("Insertion AUC (Higher AUC = More Faithful)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Insertion AUC Score", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax2.set_ylim(0.0, 0.75)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)

    plt.suptitle("XAI Explanation Faithfulness Under Domain Shift: In-Domain vs. Shifted\n"
                 "(Testing Partial Fine-Tuning `unfreeze_last_3` vs. Full Fine-Tuning `full_ft`)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[Figure] Saved XAI faithfulness comparison to: {output_path} (DPI={cfg.DPI})")


# ==============================================================================
# 8. Main Entrypoint
# ==============================================================================

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Phase 10: Calibration, Selective Prediction, and Shifted XAI")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for logit extraction")
    parser.add_argument("--seed", type=int, default=cfg.PRIMARY_SEED, help="Random seed")
    parser.add_argument("--xai-samples-per-class", type=int, default=10, help="Number of samples per class for XAI evaluation")
    args = parser.parse_args()

    run_phase10_evaluation(
        batch_size=args.batch_size,
        seed=args.seed,
        xai_sample_per_class=args.xai_samples_per_class,
    )
