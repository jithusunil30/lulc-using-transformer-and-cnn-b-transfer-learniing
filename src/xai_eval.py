"""
src/xai_eval.py - Phase 5: Quantitative Explainable AI (XAI) & Critical Evaluation
==================================================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024), Figures 11, 12, 13 & Table 1

XAI Methods Evaluated:
(a) Captum IntegratedGradients with n_steps=50 and two baselines (black, blurred)
(b) IG + NoiseTunnel (SmoothGrad)
(c) Grad-CAM via pytorch-grad-cam with custom reshape_transform for ViT, DeiT, and Swin
(d) Attention Rollout (Abnar & Zuidema, 2020) for ViT and DeiT

Quantitative Evaluation Suite:
1. Deletion Curve & AUC (lower AUC = more faithful explanation)
2. Insertion Curve & AUC (higher AUC = more faithful explanation)
3. Model Parameter Randomization Sanity Check (Adebayo et al., NeurIPS 2018)
4. Attribution Stability under Gaussian Noise
5. Paper's 95th-Percentile Colormap Reproduction (qualitative figures)
6. Misclassification Study (attribution maps for top 20 confident errors and common confusions)
"""

import sys
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Captum interpretability library
from captum.attr import IntegratedGradients, NoiseTunnel

# Pytorch Grad-CAM library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import load_raw_eurosat, get_transforms, EuroSATWrapper
from src.models import build_transformer_model
from src.train import clean_name


# ==============================================================================
# 1. Reshape Transforms & Target Layer Resolvers for Grad-CAM
# ==============================================================================

def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Reshape transform for ViT and DeiT: strips [CLS] token and maps patches to spatial grid."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def swin_reshape_transform(tensor: torch.Tensor, height: int = 7, width: int = 7) -> torch.Tensor:
    """Reshape transform for Swin Transformer final stage blocks."""
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def get_gradcam_target_layers(model: nn.Module, model_family: str) -> List[nn.Module]:
    """Retrieves appropriate normalization layer of the final block for Grad-CAM."""
    fam = model_family.lower()
    if fam in ["vit", "deit"]:
        return [model.blocks[-1].norm1]
    elif fam == "swin":
        return [model.layers[-1].blocks[-1].norm1]
    else:
        raise ValueError(f"Unsupported model family for Grad-CAM target layer: {model_family}")


def get_gradcam_reshape_tf(model_family: str):
    fam = model_family.lower()
    if fam in ["vit", "deit"]:
        return vit_reshape_transform
    elif fam == "swin":
        return swin_reshape_transform
    return None


# ==============================================================================
# 2. Attention Rollout for Vision Transformers (Abnar & Zuidema, 2020)
# ==============================================================================

def compute_attention_rollout(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Computes Attention Rollout across all transformer blocks.
    Returns 2D attention map of shape (224, 224) normalized to [0, 1].
    """
    model.eval()
    attentions = []

    def get_attn_hook(module, inp, out):
        x = inp[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(module, "q_norm") and module.q_norm is not None:
            q, k = module.q_norm(q), module.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attentions.append(attn.detach())

    hooks = []
    if hasattr(model, "blocks"):
        for blk in model.blocks:
            hooks.append(blk.attn.register_forward_hook(get_attn_hook))

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    if not attentions:
        return np.ones((224, 224), dtype=np.float32) * 0.5

    # Attention Rollout: R_0 = I, R_l = (0.5 * A_l + 0.5 * I) @ R_{l-1}
    num_tokens = attentions[0].size(-1)
    eye = torch.eye(num_tokens, device=input_tensor.device).unsqueeze(0)

    joint_attn = eye.clone()
    for attn_layer in attentions:
        head_avg = attn_layer.mean(dim=1)  # [B, N, N]
        head_avg = 0.5 * head_avg + 0.5 * eye
        head_avg = head_avg / head_avg.sum(dim=-1, keepdim=True)
        joint_attn = torch.matmul(head_avg, joint_attn)

    cls_to_patches = joint_attn[0, 0, 1:]  # shape: [196]
    grid_size = int(np.sqrt(cls_to_patches.size(0)))
    rollout_map = cls_to_patches.reshape(1, 1, grid_size, grid_size)

    rollout_224 = nn.functional.interpolate(rollout_map, size=(224, 224), mode="bilinear", align_corners=False)
    rollout_arr = rollout_224.squeeze().cpu().numpy()

    rollout_norm = (rollout_arr - rollout_arr.min()) / (rollout_arr.max() - rollout_arr.min() + 1e-8)
    return rollout_norm


# ==============================================================================
# 3. Stratified Test Sample Selector (200 Images: 20 per Class)
# ==============================================================================

def select_stratified_test_sample(n_per_class: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Selects a balanced, reproducible stratified sample of 200 images (20 per class) from test split.
    """
    raw_dataset = load_raw_eurosat(cfg.DATA_DIR)
    with open(cfg.SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_indices = splits["test"]
    targets = np.array([raw_dataset.targets[idx] for idx in test_indices])

    rng = random.Random(seed)
    selected_samples = []

    for class_id in range(cfg.NUM_CLASSES):
        cls_mask = (targets == class_id)
        matching_indices = [test_indices[i] for i in np.where(cls_mask)[0]]
        picked = rng.sample(matching_indices, min(n_per_class, len(matching_indices)))

        for idx in picked:
            img_path, label = raw_dataset.samples[idx]
            selected_samples.append({
                "dataset_idx": idx,
                "path": str(img_path),
                "label": int(label),
                "class_name": cfg.CLASS_NAMES[label],
            })

    print(f"[XAI] Selected {len(selected_samples)} stratified test samples ({n_per_class} per class across 10 classes).")
    return selected_samples


# ==============================================================================
# 4. Attribution Computations
# ==============================================================================

def compute_attribution(
    method_name: str,
    model: nn.Module,
    img_tensor: torch.Tensor,
    target_class: int,
    model_family: str,
    device: torch.device,
) -> np.ndarray:
    """
    Generates a 2D attribution heatmap of shape (224, 224) normalized to [0, 1].
    Uses architecture-tuned batch sizes inside IntegratedGradients to avoid VRAM paging.
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    fam_lower = model_family.lower()
    ig_batch_size = 10 if fam_lower == "swin" else 25
    sg_batch_size = 10 if fam_lower == "swin" else 25

    if method_name == "ig_black":
        ig = IntegratedGradients(model)
        baseline = torch.zeros_like(img_tensor)
        attr = ig.attribute(img_tensor, baselines=baseline, target=target_class, n_steps=50, internal_batch_size=ig_batch_size)
        attr_2d = torch.sum(torch.abs(attr), dim=1).squeeze().detach().cpu().numpy()

    elif method_name == "ig_blur":
        ig = IntegratedGradients(model)
        blur_tf = transforms.GaussianBlur(kernel_size=15, sigma=5.0)
        baseline = blur_tf(img_tensor)
        attr = ig.attribute(img_tensor, baselines=baseline, target=target_class, n_steps=50, internal_batch_size=ig_batch_size)
        attr_2d = torch.sum(torch.abs(attr), dim=1).squeeze().detach().cpu().numpy()

    elif method_name == "ig_smoothgrad":
        ig = IntegratedGradients(model)
        nt = NoiseTunnel(ig)
        baseline = torch.zeros_like(img_tensor)
        attr = nt.attribute(img_tensor, baselines=baseline, target=target_class,
                            nt_type="smoothgrad", nt_samples=3, stdevs=0.15, n_steps=15, internal_batch_size=sg_batch_size)
        attr_2d = torch.sum(torch.abs(attr), dim=1).squeeze().detach().cpu().numpy()

    elif method_name == "grad_cam":
        target_layers = get_gradcam_target_layers(model, model_family)
        reshape_tf = get_gradcam_reshape_tf(model_family)
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_tf)
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        attr_2d = grayscale_cam[0]

    elif method_name == "rollout":
        if fam_lower in ["vit", "deit"]:
            attr_2d = compute_attention_rollout(model, img_tensor)
        else:
            target_layers = get_gradcam_target_layers(model, "swin")
            reshape_tf = get_gradcam_reshape_tf("swin")
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_tf)
            attr_2d = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(target_class)])[0]

    else:
        raise ValueError(f"Unknown XAI method: {method_name}")

    # Min-max normalization with numerical stability guard
    attr_min, attr_max = attr_2d.min(), attr_2d.max()
    if attr_max - attr_min > 1e-8:
        norm_attr = (attr_2d - attr_min) / (attr_max - attr_min)
    else:
        norm_attr = np.zeros_like(attr_2d)
    return norm_attr


# ==============================================================================
# 5. Quantitative Metric Engines (Insertion, Deletion, Sanity Check, Stability)
# ==============================================================================

def compute_insertion_deletion_auc(
    model: nn.Module,
    img_tensor: torch.Tensor,
    target_class: int,
    attribution_map: np.ndarray,
    device: torch.device,
    n_steps: int = 10,
) -> Tuple[float, float, List[float], List[float]]:
    """
    Computes Deletion AUC and Insertion AUC by masking / inserting top-attributed pixels
    using batched forward passes for maximum GPU performance.
    """
    model.eval()
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    flat_attr = attribution_map.flatten()
    sorted_indices = np.argsort(-flat_attr)  # Descending order of saliency
    total_pixels = len(flat_attr)

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    del_batch = []
    ins_batch = []
    base_zero = torch.zeros_like(img_tensor[0])

    for frac in fractions:
        n_replace = int(frac * total_pixels)

        # 1. Deletion: Start with image, mask top n_replace pixels with baseline 0
        del_img = img_tensor[0].clone()
        if n_replace > 0:
            mask = np.zeros(total_pixels, dtype=bool)
            mask[sorted_indices[:n_replace]] = True
            mask_2d = torch.tensor(mask.reshape(224, 224), device=device, dtype=torch.bool)
            del_img[:, mask_2d] = 0.0
        del_batch.append(del_img)

        # 2. Insertion: Start with baseline 0, insert top n_replace pixels from image
        ins_img = base_zero.clone()
        if n_replace > 0:
            mask = np.zeros(total_pixels, dtype=bool)
            mask[sorted_indices[:n_replace]] = True
            mask_2d = torch.tensor(mask.reshape(224, 224), device=device, dtype=torch.bool)
            ins_img[:, mask_2d] = img_tensor[0, :, mask_2d]
        ins_batch.append(ins_img)

    del_batch_tensor = torch.stack(del_batch, dim=0)  # Shape: [n_steps+1, 3, 224, 224]
    ins_batch_tensor = torch.stack(ins_batch, dim=0)

    with torch.no_grad():
        logits_del = model(del_batch_tensor)
        probs_del = torch.softmax(logits_del, dim=1)[:, target_class].cpu().numpy().tolist()

        logits_ins = model(ins_batch_tensor)
        probs_ins = torch.softmax(logits_ins, dim=1)[:, target_class].cpu().numpy().tolist()

    del_auc = float(auc(fractions, probs_del))
    ins_auc = float(auc(fractions, probs_ins))
    return del_auc, ins_auc, probs_del, probs_ins


def compute_randomization_sanity_check(
    attr_trained: np.ndarray,
    random_model: nn.Module,
    method_name: str,
    img_tensor: torch.Tensor,
    target_class: int,
    model_family: str,
    device: torch.device,
) -> float:
    """
    Adebayo et al. (2018) Parameter Randomization Sanity Check.
    Returns Pearson correlation between attributions of trained and randomized model.
    """
    attr_random = compute_attribution(method_name, random_model, img_tensor, target_class, model_family, device)

    if np.std(attr_trained) < 1e-7 or np.std(attr_random) < 1e-7:
        return 0.0

    r_val, _ = pearsonr(attr_trained.flatten(), attr_random.flatten())
    return float(r_val) if not np.isnan(r_val) else 0.0


def compute_noise_stability(
    attr_clean: np.ndarray,
    model: nn.Module,
    method_name: str,
    img_tensor: torch.Tensor,
    target_class: int,
    model_family: str,
    device: torch.device,
    noise_std: float = 0.05,
) -> float:
    """
    Measures Pearson correlation between attribution of clean image vs slightly noisy image.
    """
    noisy_tensor = img_tensor + torch.randn_like(img_tensor) * noise_std
    attr_noisy = compute_attribution(method_name, model, noisy_tensor, target_class, model_family, device)

    if np.std(attr_clean) < 1e-7 or np.std(attr_noisy) < 1e-7:
        return 1.0

    r_val, _ = pearsonr(attr_clean.flatten(), attr_noisy.flatten())
    return float(r_val) if not np.isnan(r_val) else 1.0


# ==============================================================================
# 6. Qualitative Figures (Khan et al. 95th Percentile Reproduction)
# ==============================================================================

def plot_paper_95th_percentile_figure(
    qualitative_records: Dict[str, List[Dict[str, Any]]],
    output_path: Path = cfg.FIGURES_DIR / "xai_95th_percentile_comparison.png",
):
    """
    Reproduces Figures 11 and 12 in Khan et al. (2024):
    Overlays Integrated Gradients thresholded at the 95th percentile across multiple classes and models.
    """
    model_keys = list(qualitative_records.keys())
    if not model_keys:
        return

    n_samples = len(qualitative_records[model_keys[0]])
    n_cols = len(model_keys) + 1  # 1 for raw RGB + models

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3.8 * n_cols, 3.4 * n_samples))

    for r_idx in range(n_samples):
        first_rec = qualitative_records[model_keys[0]][r_idx]
        raw_arr = first_rec["raw_img"]

        # Col 0: Original satellite image
        axes[r_idx, 0].imshow(raw_arr)
        axes[r_idx, 0].set_title(f"Class: {first_rec['class_name']}\n(Sentinel-2 RGB)", fontsize=10, fontweight="bold")
        axes[r_idx, 0].axis("off")

        for c_idx, m_label in enumerate(model_keys, start=1):
            rec = qualitative_records[m_label][r_idx]
            attr_map = rec["attr_map"]
            pred_name = rec["pred_name"]
            prob = rec["prob"]

            # 95th percentile thresholding as in Khan et al. 2024
            p95 = np.percentile(attr_map, 95)
            thresholded_attr = np.where(attr_map >= p95, attr_map, 0.0)

            axes[r_idx, c_idx].imshow(raw_arr)
            masked_overlay = np.ma.masked_where(thresholded_attr < p95, thresholded_attr)
            axes[r_idx, c_idx].imshow(masked_overlay, cmap="turbo", alpha=0.75, vmin=p95, vmax=1.0)
            axes[r_idx, c_idx].set_title(f"{m_label}\nPred: {pred_name} ({prob*100:.1f}%)",
                                         fontsize=10, fontweight="bold")
            axes[r_idx, c_idx].axis("off")

    plt.suptitle("Reproduction of Khan et al. (2024) 95th-Percentile Attribution Maps across Architectures & Strategies\n"
                 "(Salient features with Integrated Gradients attribution >= 95th percentile overlaid on Sentinel-2 RGB)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[XAI Figures] Saved 95th percentile reproduction figure to: {output_path} (DPI={cfg.DPI})")


# ==============================================================================
# 7. Misclassification Study (Top 20 Confident Errors & Common Confusions)
# ==============================================================================

def run_misclassification_study(
    model: nn.Module,
    model_family: str,
    device: torch.device,
    top_errors_path: Path = cfg.FIGURES_DIR / "misclassifications_attribution_study.png",
    confusions_path: Path = cfg.FIGURES_DIR / "common_confusions_study.png",
):
    """
    Finds the 20 most confident test errors and the 3 classic LULC confusion pairs,
    generating publication figures with predicted vs true class attributions.
    Uses batched DataLoader evaluation for maximum throughput.
    """
    print("\n[XAI Misclassification] Fast batch evaluating test set to locate errors and confusion pairs...")
    raw_dataset = load_raw_eurosat(cfg.DATA_DIR)
    _, eval_tf, raw_tf = get_transforms()

    with open(cfg.SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_indices = splits["test"]

    test_ds = EuroSATWrapper(raw_dataset, test_indices, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    errors = []
    common_pairs = {
        ("AnnualCrop", "PermanentCrop"): [],
        ("PermanentCrop", "AnnualCrop"): [],
        ("Pasture", "HerbaceousVegetation"): [],
        ("HerbaceousVegetation", "Pasture"): [],
        ("River", "Highway"): [],
        ("Highway", "River"): [],
    }

    model.eval()
    with torch.no_grad():
        for imgs, targets, paths in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_cpu = preds.cpu().tolist()
            targets_cpu = targets.tolist()

            for i in range(len(preds_cpu)):
                p_lbl = preds_cpu[i]
                t_lbl = targets_cpu[i]
                if p_lbl != t_lbl:
                    p_name = cfg.CLASS_NAMES[p_lbl]
                    t_name = cfg.CLASS_NAMES[t_lbl]
                    conf = float(probs[i, p_lbl].item())
                    item = {
                        "path": paths[i],
                        "true_label": t_lbl,
                        "pred_label": p_lbl,
                        "true_name": t_name,
                        "pred_name": p_name,
                        "confidence": conf,
                    }
                    errors.append(item)

                    if (t_name, p_name) in common_pairs:
                        common_pairs[(t_name, p_name)].append(item)

    errors.sort(key=lambda x: -x["confidence"])
    print(f"[XAI Misclassification] Found {len(errors)} total test errors. Top error confidence: {errors[0]['confidence']*100:.2f}%.")

    # --------------------------------------------------------------------------
    # Figure 1: 20 Most Confident Errors (10 rows x 6 columns: 2 complete error triplets per row)
    # --------------------------------------------------------------------------
    top20_errors = errors[:20]
    fig, axes = plt.subplots(10, 6, figsize=(20, 28))

    for idx, err in enumerate(top20_errors):
        row = idx // 2
        col_offset = (idx % 2) * 3

        pil_img = Image.open(err["path"]).convert("RGB")
        tensor_img = eval_tf(pil_img).to(device)
        raw_arr = raw_tf(pil_img).permute(1, 2, 0).numpy()

        attr_pred = compute_attribution("grad_cam", model, tensor_img, err["pred_label"], model_family, device)
        attr_true = compute_attribution("grad_cam", model, tensor_img, err["true_label"], model_family, device)

        # 1. Raw image
        axes[row, col_offset].imshow(raw_arr)
        axes[row, col_offset].set_title(f"#{idx+1} Ground Truth: {err['true_name']}", fontsize=10, fontweight="bold")
        axes[row, col_offset].axis("off")

        # 2. Predicted class heatmap
        axes[row, col_offset + 1].imshow(raw_arr)
        axes[row, col_offset + 1].imshow(attr_pred, cmap="jet", alpha=0.6)
        axes[row, col_offset + 1].set_title(f"Mistake: {err['pred_name']}\n(Conf: {err['confidence']*100:.1f}%)",
                                            fontsize=10, fontweight="bold", color="darkred")
        axes[row, col_offset + 1].axis("off")

        # 3. Ground truth counterfactual heatmap
        axes[row, col_offset + 2].imshow(raw_arr)
        axes[row, col_offset + 2].imshow(attr_true, cmap="viridis", alpha=0.6)
        axes[row, col_offset + 2].set_title(f"Evidence for True: {err['true_name']}",
                                            fontsize=10, fontweight="bold", color="darkgreen")
        axes[row, col_offset + 2].axis("off")

    plt.suptitle("Analysis of the 20 Most Confident Model Misclassifications\n"
                 "(Raw Image | Saliency Driving Incorrect Prediction | Saliency Supporting Ground Truth)",
                 fontsize=14, fontweight="bold", y=1.002)
    plt.tight_layout()
    top_errors_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(top_errors_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[XAI Figures] Saved 20 confident errors attribution study to: {top_errors_path} (DPI={cfg.DPI})")

    # --------------------------------------------------------------------------
    # Figure 2: Common Confusion Pairs Study (AnnualCrop/PermanentCrop, Pasture/Herbaceous, River/Highway)
    # --------------------------------------------------------------------------
    confusion_definitions = [
        (("AnnualCrop", "PermanentCrop"), ("PermanentCrop", "AnnualCrop"),
         "Agricultural Confusion: Crop field textures confused with tree canopy spacing / orchards"),
        (("Pasture", "HerbaceousVegetation"), ("HerbaceousVegetation", "Pasture"),
         "Vegetation Confusion: Grazing pastures confused with wild herbaceous grassland meadows"),
        (("River", "Highway"), ("Highway", "River"),
         "Linear Geometry Confusion: Meandering water channels vs linear paved asphalt corridors"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for row_idx, (dir1, dir2, rationale) in enumerate(confusion_definitions):
        candidates = common_pairs[dir1] + common_pairs[dir2]
        if candidates:
            sample = candidates[0]
            pil_img = Image.open(sample["path"]).convert("RGB")
            tensor_img = eval_tf(pil_img).to(device)
            raw_arr = raw_tf(pil_img).permute(1, 2, 0).numpy()

            attr_pred = compute_attribution("grad_cam", model, tensor_img, sample["pred_label"], model_family, device)
            attr_true = compute_attribution("grad_cam", model, tensor_img, sample["true_label"], model_family, device)

            axes[row_idx, 0].imshow(raw_arr)
            axes[row_idx, 0].set_title(f"Actual: {sample['true_name']}\n(Sentinel-2 RGB)", fontsize=11, fontweight="bold")
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(raw_arr)
            axes[row_idx, 1].imshow(attr_pred, cmap="jet", alpha=0.6)
            axes[row_idx, 1].set_title(f"Model Predicted: {sample['pred_name']}\n(Conf: {sample['confidence']*100:.1f}%)",
                                       fontsize=11, fontweight="bold", color="darkred")
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(raw_arr)
            axes[row_idx, 2].imshow(attr_true, cmap="viridis", alpha=0.6)
            axes[row_idx, 2].set_title(f"Attributed to Actual: {sample['true_name']}\n{rationale}",
                                       fontsize=9.5, fontweight="bold", color="darkgreen")
            axes[row_idx, 2].axis("off")
        else:
            for c in range(3):
                axes[row_idx, c].text(0.5, 0.5, f"No test errors between\n{dir1[0]} and {dir1[1]}",
                                      ha="center", va="center", fontsize=11)
                axes[row_idx, c].axis("off")

    plt.suptitle("Common LULC Confusion Pairs: Spatial Feature Confusion Analysis", fontsize=14, fontweight="bold", y=1.002)
    plt.tight_layout()
    confusions_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(confusions_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[XAI Figures] Saved common confusions attribution study to: {confusions_path} (DPI={cfg.DPI})")


# ==============================================================================
# 8. Deletion & Insertion Curves Plot & Quantitative Summary Plot
# ==============================================================================

def plot_deletion_insertion_curves(
    curves_dict: Dict[str, Dict[str, Tuple[List[float], List[float]]]],
    output_path: Path = cfg.FIGURES_DIR / "xai_deletion_insertion_curves.png",
):
    """
    Plots the progressive Deletion and Insertion curves across fractions [0.0 to 1.0].
    """
    fractions = np.linspace(0.0, 1.0, 11)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {
        "ig_black": "#1f77b4",
        "ig_blur": "#ff7f0e",
        "ig_smoothgrad": "#2ca02c",
        "grad_cam": "#d62728",
        "rollout": "#9467bd",
    }
    labels = {
        "ig_black": "IG (Black Baseline)",
        "ig_blur": "IG (Blurred Baseline)",
        "ig_smoothgrad": "IG + NoiseTunnel (SmoothGrad)",
        "grad_cam": "Grad-CAM",
        "rollout": "Attention Rollout",
    }

    method_dels: Dict[str, List[np.ndarray]] = {}
    method_inss: Dict[str, List[np.ndarray]] = {}

    for m_label, methods in curves_dict.items():
        for m_name, (d_curve, i_curve) in methods.items():
            if m_name not in method_dels:
                method_dels[m_name] = []
                method_inss[m_name] = []
            method_dels[m_name].append(np.array(d_curve))
            method_inss[m_name].append(np.array(i_curve))

    for m_name in method_dels.keys():
        mean_del = np.mean(method_dels[m_name], axis=0)
        mean_ins = np.mean(method_inss[m_name], axis=0)
        c = colors.get(m_name, "black")
        lbl = labels.get(m_name, m_name)

        ax1.plot(fractions, mean_del, marker="o", linewidth=2.2, label=lbl, color=c)
        ax2.plot(fractions, mean_ins, marker="s", linewidth=2.2, label=lbl, color=c)

    ax1.set_title("Deletion Curve (Lower AUC = More Faithful)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Fraction of Top-Attributed Pixels Removed", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted Class Probability", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9.5)

    ax2.set_title("Insertion Curve (Higher AUC = More Faithful)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Fraction of Top-Attributed Pixels Inserted", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Class Probability", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9.5)

    plt.suptitle("Quantitative Evaluation of Explanation Faithfulness: Deletion & Insertion Curves",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[XAI Figures] Saved deletion/insertion curves to: {output_path} (DPI={cfg.DPI})")


def plot_xai_quantitative_summary(
    df: pd.DataFrame,
    out_path: Path = cfg.FIGURES_DIR / "xai_quantitative_tradeoffs.png",
):
    """Generates 3-panel publication figure comparing Insertion/Deletion AUC, Sanity Check, and Stability."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    methods = df["xai_method"].unique()
    x = np.arange(len(methods))
    width = 0.35

    df_full = df[df["strategy"] == "full_ft"].groupby("xai_method").mean(numeric_only=True).reindex(methods)
    df_part = df[df["strategy"] == "unfreeze_last_3"].groupby("xai_method").mean(numeric_only=True).reindex(methods)

    # Panel 1: Insertion vs Deletion AUC (Faithfulness)
    ax1.bar(x - width/2, df_full["insertion_auc"], width, label="Insertion AUC (Higher is Better)", color="#2ca02c", alpha=0.85)
    ax1.bar(x + width/2, df_full["deletion_auc"], width, label="Deletion AUC (Lower is Better)", color="#d62728", alpha=0.85)
    ax1.set_ylabel("AUC Score", fontsize=11, fontweight="bold")
    ax1.set_title("Explanation Faithfulness (Insertion vs. Deletion AUC)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 2: Model Parameter Randomization Sanity Check (Adebayo et al.)
    colors = ["#2ca02c" if p else "#d62728" for p in df_full["passes_adebayo_sanity_check"]]
    ax2.bar(x, df_full["randomization_correlation"], width=0.5, color=colors, alpha=0.85)
    ax2.axhline(0.20, linestyle="--", color="gray", label="Sanity Threshold (r < 0.20)")
    ax2.set_ylabel("Correlation (Trained vs. Random Weights)", fontsize=11, fontweight="bold")
    ax2.set_title("Adebayo Sanity Check: Sensitivity to Learned Weights", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 3: Stability under Input Noise
    ax3.bar(x - width/2, df_full["noise_stability_correlation"], width, label="Full Fine-Tuning", color="#1f77b4", alpha=0.85)
    ax3.bar(x + width/2, df_part["noise_stability_correlation"], width, label="Partial FT (Last 3 Blocks)", color="#ff7f0e", alpha=0.85)
    ax3.set_ylabel("Pearson Correlation r", fontsize=11, fontweight="bold")
    ax3.set_title("Attribution Stability under Input Noise (r)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
    ax3.set_ylim(0.0, 1.1)
    ax3.legend(frameon=True)
    ax3.grid(axis="y", linestyle="--", alpha=0.6)

    plt.suptitle("Quantitative Evaluation of Transformer Explainability (Captum vs. Grad-CAM vs. Rollout)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[XAI Figures] Saved quantitative trade-offs figure to: {out_path} (DPI={cfg.DPI})")


# ==============================================================================
# 9. Main Phase 5 Benchmark Runner
# ==============================================================================

def run_phase5(
    n_samples_per_class: int = 20,
    seed: int = 42,
):
    """
    Executes complete quantitative XAI benchmark across methods, models, and strategies.
    Evaluates 200 stratified test images with full memory management and high GPU throughput.
    """
    print("=" * 80)
    print(f"PHASE 5: Quantitative XAI Benchmark on Stratified Test Sample ({n_samples_per_class * 10} Images)")
    print("=" * 80)

    device = torch.device(cfg.DEVICE)
    _, eval_tf, raw_tf = get_transforms()

    # 1. Stratified sample of 200 images (20 per class)
    sample_items = select_stratified_test_sample(n_per_class=n_samples_per_class, seed=seed)

    # Pick 5 diverse samples across classes for Khan et al. 95th-percentile qualitative figure
    qualitative_samples = []
    seen_classes = set()
    for item in sample_items:
        cls_name = item["class_name"]
        if cls_name not in seen_classes and cls_name in ["Highway", "River", "Forest", "PermanentCrop", "Residential"]:
            seen_classes.add(cls_name)
            qualitative_samples.append(item)
        if len(qualitative_samples) >= 5:
            break

    # 2. Models to evaluate: Best strategy (unfreeze_last_3) + full_ft for ViT, DeiT, Swin
    eval_targets = [
        ("vit", "full_ft"),
        ("vit", "unfreeze_last_3"),
        ("deit", "full_ft"),
        ("deit", "unfreeze_last_3"),
        ("swin", "full_ft"),
        ("swin", "unfreeze_last_3"),
    ]

    xai_methods = ["ig_black", "ig_blur", "ig_smoothgrad", "grad_cam", "rollout"]

    results_table = []
    curves_dict: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}
    qualitative_records: Dict[str, List[Dict[str, Any]]] = {}

    # Pre-build randomized models for Adebayo sanity check
    print("\n[XAI Setup] Initializing randomized reference models for Adebayo sanity check...")
    random_models: Dict[str, nn.Module] = {}
    for fam in ["vit", "deit", "swin"]:
        m_name = cfg.get_model_name(fam)
        rand_m, _ = build_transformer_model(m_name, strategy="scratch")
        random_models[fam] = rand_m.to(device).eval()

    for family, strategy in eval_targets:
        model_name = cfg.get_model_name(family)
        clean_m = clean_name(model_name)
        ckpt_path = cfg.CHECKPOINTS_DIR / f"{clean_m}_{strategy}_seed{seed}_best.pth"

        if not ckpt_path.exists():
            print(f"[XAI] Warning: Checkpoint {ckpt_path.name} not found. Skipping.")
            continue

        print(f"\n================================================================================")
        print(f"Loading Model: {family.upper()} [{strategy}] from {ckpt_path.name}")
        print(f"================================================================================")
        model, _ = build_transformer_model(model_name, strategy=strategy)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).eval()

        label_key = f"{family.upper()}-{strategy}"
        curves_dict[label_key] = {}
        qualitative_records[label_key] = []

        rand_ref_model = random_models[family]

        # 1. Collect qualitative 95th-percentile attributions for this model
        for q_item in qualitative_samples:
            pil_img = Image.open(q_item["path"]).convert("RGB")
            t_img = eval_tf(pil_img).unsqueeze(0).to(device)
            raw_arr = raw_tf(pil_img).permute(1, 2, 0).numpy()

            with torch.no_grad():
                logits = model(t_img)
                pred_cls = int(torch.argmax(logits, dim=1).item())
                prob = float(torch.softmax(logits, dim=1)[0, pred_cls].item())

            attr_ig = compute_attribution("ig_black", model, t_img, pred_cls, family, device)
            qualitative_records[label_key].append({
                "class_name": q_item["class_name"],
                "raw_img": raw_arr,
                "pred_name": cfg.CLASS_NAMES[pred_cls],
                "prob": prob,
                "attr_map": attr_ig,
            })

        # 2. If this is ViT Full-FT, run the misclassification study immediately while in memory
        if family == "vit" and strategy == "full_ft":
            run_misclassification_study(model, family, device)

        # 3. Evaluate XAI methods
        for method in xai_methods:
            if method == "rollout" and family == "swin":
                continue

            print(f"\n  [Evaluating XAI Method] {method.upper()} on {family.upper()} [{strategy}] across {len(sample_items)} images...")

            del_aucs = []
            ins_aucs = []
            del_curves_all = []
            ins_curves_all = []
            sanity_corrs = []
            stabilities = []

            for s_idx, item in enumerate(tqdm(sample_items, desc=f"    {method}", ncols=80)):
                pil_img = Image.open(item["path"]).convert("RGB")
                tensor_img = eval_tf(pil_img).to(device)

                attr = compute_attribution(method, model, tensor_img, item["label"], family, device)

                # 1. Insertion & Deletion AUC (vectorized GPU evaluation)
                d_auc, i_auc, d_curve, i_curve = compute_insertion_deletion_auc(
                    model, tensor_img, item["label"], attr, device, n_steps=10
                )
                del_aucs.append(d_auc)
                ins_aucs.append(i_auc)
                del_curves_all.append(d_curve)
                ins_curves_all.append(i_curve)

                # 2. Adebayo Sanity Check (evaluate on first 30 samples for speed)
                if s_idx < 30:
                    corr = compute_randomization_sanity_check(
                        attr, rand_ref_model, method, tensor_img, item["label"], family, device
                    )
                    sanity_corrs.append(corr)

                # 3. Noise Stability (evaluate on first 30 samples)
                if s_idx < 30:
                    stab = compute_noise_stability(
                        attr, model, method, tensor_img, item["label"], family, device, noise_std=0.05
                    )
                    stabilities.append(stab)

            mean_del = float(np.mean(del_aucs))
            mean_ins = float(np.mean(ins_aucs))
            mean_sanity = float(np.mean(sanity_corrs)) if sanity_corrs else 0.0
            mean_stab = float(np.mean(stabilities)) if stabilities else 1.0

            passes_sanity = bool(abs(mean_sanity) < 0.25)

            curves_dict[label_key][method] = (
                np.mean(del_curves_all, axis=0).tolist(),
                np.mean(ins_curves_all, axis=0).tolist(),
            )

            results_table.append({
                "model_family": family.upper(),
                "model_name": model_name,
                "strategy": strategy,
                "xai_method": method,
                "deletion_auc": round(mean_del, 4),
                "insertion_auc": round(mean_ins, 4),
                "faithfulness_ratio": round(mean_ins / (mean_del + 1e-6), 3),
                "randomization_correlation": round(mean_sanity, 4),
                "passes_adebayo_sanity_check": passes_sanity,
                "noise_stability_correlation": round(mean_stab, 4),
            })

            print(f"    --> {method:<14}: Del AUC={mean_del:.4f} | Ins AUC={mean_ins:.4f} | Sanity r={mean_sanity:+.3f} (Pass={passes_sanity}) | Stability={mean_stab:.4f}")

        # Explicitly destroy model and empty GPU cache
        del model
        torch.cuda.empty_cache()

    # Clean up randomized models
    del random_models
    torch.cuda.empty_cache()

    # Save quantitative metrics to CSV and JSON
    df_xai = pd.DataFrame(results_table)
    csv_path = cfg.RESULTS_DIR / "xai_metrics.csv"
    json_path = cfg.RESULTS_DIR / "xai_metrics.json"

    df_xai.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    print(f"\n[XAI] Successfully saved quantitative metrics table to: {csv_path}")

    # Plot Figures (DPI=200)
    plot_deletion_insertion_curves(curves_dict, cfg.FIGURES_DIR / "xai_deletion_insertion_curves.png")
    plot_xai_quantitative_summary(df_xai, cfg.FIGURES_DIR / "xai_quantitative_tradeoffs.png")
    plot_paper_95th_percentile_figure(qualitative_records, cfg.FIGURES_DIR / "xai_95th_percentile_comparison.png")

    # Print Formatted Table to Console
    _print_xai_summary_table(df_xai)

    return df_xai


def _print_xai_summary_table(df: pd.DataFrame):
    """Prints a clean formatted summary of quantitative XAI metrics."""
    print("\n" + "=" * 115)
    print(f"{'Model':<8}{'Strategy':<18}{'Method':<16}{'Deletion AUC':<15}{'Insertion AUC':<16}{'Sanity Check r':<18}{'Noise Stability':<16}")
    print("=" * 115)

    for _, r in df.iterrows():
        sanity_str = f"{r['randomization_correlation']:+.3f} ({'PASS' if r['passes_adebayo_sanity_check'] else 'FAIL'})"
        print(f"{r['model_family']:<8}"
              f"{r['strategy']:<18}"
              f"{r['xai_method']:<16}"
              f"{r['deletion_auc']:<15.4f}"
              f"{r['insertion_auc']:<16.4f}"
              f"{sanity_str:<18}"
              f"{r['noise_stability_correlation']:<16.4f}")
    print("=" * 115)


if __name__ == "__main__":
    run_phase5(n_samples_per_class=20, seed=42)
