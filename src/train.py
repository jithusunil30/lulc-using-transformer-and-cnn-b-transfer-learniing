"""
src/train.py - Phase 2: Training & Evaluation Framework
======================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024)

Features:
- Strategies: scratch, linear_probe, full_ft, unfreeze_last_k (k in {1, 2, 3})
- SGD Optimizer (momentum 0.9, lr 0.001), CrossEntropyLoss, mixed precision (torch.autocast)
- Early stopping on VALIDATION loss with patience 5 (max epochs 15)
- Saves best-val checkpoint to checkpoints/<model>_<strategy>_<seed>_best.pth
- Per-run logging: trainable/total params, peak GPU memory, seconds/epoch, epochs to best val,
  total wall time, inference throughput (img/s)
- Single comprehensive test evaluation: accuracy, macro precision/recall/F1, balanced accuracy,
  test loss, per-class recall, confusion matrix, Expected Calibration Error (ECE)
- Saves everything to results/<model>_<strategy>_<seed>.json
- Smoke test mode: 1 epoch on a 2000-image subset
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import time
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import get_dataloaders
from src.models import build_transformer_model, count_parameters


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width confidence bins.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def clean_name(model_name: str) -> str:
    """Creates a clean string identifier for file naming."""
    return model_name.replace("/", "_").replace(".", "_")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    measure_throughput: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """
    Evaluates model on a dataloader. Computes loss, accuracy, predictions,
    probabilities, and optionally measures inference throughput (images/sec).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []

    throughput_img_per_sec = None
    start_time = None
    timed_samples = 0
    warmup_batches = 3

    with torch.no_grad():
        for b_idx, (images, targets, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            b_size = images.size(0)

            # Measure throughput during evaluation
            if measure_throughput and b_idx == warmup_batches:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                timed_samples = 0

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, targets)

            if measure_throughput and b_idx >= warmup_batches:
                timed_samples += b_size

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * b_size
            total_samples += b_size

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if measure_throughput and start_time is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        if elapsed > 0:
            throughput_img_per_sec = round(timed_samples / elapsed, 2)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    avg_loss = float(total_loss / total_samples)

    metrics = {
        "loss": avg_loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    return metrics, y_true, y_pred, y_prob, throughput_img_per_sec


def train_and_evaluate(
    model_name: str,
    strategy: str = "unfreeze_last_3",
    seed: int = cfg.PRIMARY_SEED,
    batch_size: int = cfg.BATCH_SIZE,
    max_epochs: int = 15,
    patience: int = 5,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    smoke_test: bool = False,
    num_workers: int = cfg.NUM_WORKERS,
) -> Dict[str, Any]:
    """
    Executes complete training and evaluation pipeline for a model under a specific strategy.
    """
    cfg.ensure_directories()

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device(cfg.DEVICE)
    use_amp = cfg.USE_AMP

    clean_m = clean_name(model_name)
    run_tag = f"{clean_m}_{strategy}_seed{seed}"
    if smoke_test:
        run_tag = f"smoke_{run_tag}"

    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {run_tag}")
    print(f"Model: {model_name} | Strategy: {strategy} | Seed: {seed} | Batch Size: {batch_size}")
    print(f"Device: {device} | Mixed Precision (AMP): {use_amp} | Max Epochs: {max_epochs}")
    print("=" * 80)

    # 1. Prepare DataLoaders
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        batch_size=batch_size,
        seed=seed,
        model_name=model_name,
        num_workers=num_workers,
    )

    if smoke_test:
        print("[Smoke Test] Subsetting to 2,000 train images, 400 val images, 400 test images.")
        train_sub = Subset(train_loader.dataset, range(min(2000, len(train_loader.dataset))))
        val_sub = Subset(val_loader.dataset, range(min(400, len(val_loader.dataset))))
        test_sub = Subset(test_loader.dataset, range(min(400, len(test_loader.dataset))))

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        max_epochs = 1

    # 2. Build Model with Layer Freezing Strategy
    model, model_meta = build_transformer_model(
        model_name=model_name,
        strategy=strategy,
        num_classes=cfg.NUM_CLASSES,
    )
    model = model.to(device)

    trainable_p, total_p, pct_tr = count_parameters(model)
    print(f"Parameters: {trainable_p:,} trainable / {total_p:,} total ({pct_tr:.2f}% active)")

    # 3. Setup Optimizer, Loss, and Scaler
    # Paper explicitly specifies SGD with momentum 0.9 and lr 0.001
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and torch.cuda.is_available())

    # 4. Training Loop with Early Stopping on Validation Loss
    best_val_loss = float("inf")
    best_epoch = -1
    best_checkpoint_path = cfg.CHECKPOINTS_DIR / f"{run_tag}_best.pth"
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_seconds": [],
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    wall_clock_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()

        running_loss = 0.0
        running_correct = 0
        total_train_samples = 0

        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            b_size = images.size(0)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * b_size
            preds = torch.argmax(outputs, dim=1)
            running_correct += (preds == targets).sum().item()
            total_train_samples += b_size

        epoch_seconds = time.perf_counter() - epoch_start
        train_loss = running_loss / total_train_samples
        train_acc = running_correct / total_train_samples

        # Validation evaluation (NEVER test)
        val_metrics, _, _, _, _ = evaluate_model(model, val_loader, criterion, device, use_amp=use_amp)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc * 100, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc * 100, 2))
        history["epoch_seconds"].append(round(epoch_seconds, 2))

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}% | "
              f"Time: {epoch_seconds:5.2f}s")

        # Check Early Stopping & Save Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "strategy": strategy,
                "seed": seed,
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, best_checkpoint_path)
            print(f"  --> [*] New Best Validation Loss: {val_loss:.4f} (Saved to {best_checkpoint_path.name})")
        else:
            patience_counter += 1
            print(f"  --> [!] Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}! Reached patience limit of {patience}.")
                break

    total_wall_time = time.perf_counter() - wall_clock_start

    # Peak GPU memory
    peak_gpu_mb = None
    if torch.cuda.is_available():
        peak_gpu_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

    # 5. Load Best Checkpoint ONCE and Evaluate on TEST Set
    print("\n" + "-" * 60)
    print(f"Loading best checkpoint from Epoch {best_epoch} for Test Evaluation...")
    print("-" * 60)
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    test_metrics_base, y_true, y_pred, y_prob, test_throughput = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        use_amp=use_amp,
        measure_throughput=True,
    )

    # Calculate comprehensive test metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()
    ece = compute_ece(y_prob, y_true, n_bins=15)

    # Per-class recall breakdown
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall_dict = {
        cfg.CLASS_NAMES[i]: round(float(per_class_rec[i]) * 100, 2)
        for i in range(len(cfg.CLASS_NAMES))
    }

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION RESULTS:")
    print("=" * 80)
    print(f"  Test Loss:                  {test_metrics_base['loss']:.4f}")
    print(f"  Test Accuracy:              {acc * 100:.2f}%")
    print(f"  Macro Precision:            {macro_prec * 100:.2f}%")
    print(f"  Macro Recall:               {macro_rec * 100:.2f}%")
    print(f"  Macro F1-Score:             {macro_f1 * 100:.2f}%")
    print(f"  Balanced Accuracy:          {balanced_acc * 100:.2f}%")
    print(f"  Expected Calibration (ECE): {ece:.4f}")
    print(f"  Inference Throughput:       {test_throughput} img/s")
    print(f"  Peak GPU Memory:            {peak_gpu_mb} MB")
    print(f"  Total Wall Time:            {total_wall_time:.2f} s ({total_wall_time/60:.2f} min)")
    print(f"  Epochs to Best Val:         {best_epoch}")
    print("=" * 80)

    # 6. Save Detailed Results to results/<model>_<strategy>_<seed>.json
    result_data = {
        "experiment": run_tag,
        "model_name": model_name,
        "clean_model_name": clean_m,
        "strategy": strategy,
        "seed": seed,
        "smoke_test": smoke_test,
        "parameters": {
            "trainable_params": trainable_p,
            "total_params": total_p,
            "trainable_pct": pct_tr,
            "total_blocks": model_meta["total_blocks"],
            "k_blocks_unfrozen": model_meta["k_blocks_unfrozen"],
        },
        "hardware_efficiency": {
            "peak_gpu_memory_mb": peak_gpu_mb,
            "total_wall_time_seconds": round(total_wall_time, 2),
            "total_wall_time_minutes": round(total_wall_time / 60, 2),
            "seconds_per_epoch": history["epoch_seconds"],
            "avg_seconds_per_epoch": round(float(np.mean(history["epoch_seconds"])), 2) if history["epoch_seconds"] else 0.0,
            "epochs_to_best_val": best_epoch,
            "total_epochs_trained": len(history["epoch"]),
            "inference_throughput_img_per_sec": test_throughput,
        },
        "test_metrics": {
            "test_loss": round(test_metrics_base["loss"], 4),
            "accuracy": round(acc * 100, 2),
            "macro_precision": round(macro_prec * 100, 2),
            "macro_recall": round(macro_rec * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "balanced_accuracy": round(balanced_acc * 100, 2),
            "expected_calibration_error": round(ece, 4),
            "per_class_recall": per_class_recall_dict,
            "confusion_matrix": cm,
        },
        "predictions": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        },
        "training_history": history,
        "checkpoint_path": str(best_checkpoint_path),
    }

    result_json_path = cfg.RESULTS_DIR / f"{run_tag}.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    print(f"[Results] Detailed run metrics saved to: {result_json_path}")

    # 7. Generate & Save High-Resolution Figures (DPI=200)
    # Figure A: Training & Validation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = history["epoch"]
    ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.axvline(best_epoch, linestyle="--", color="gray", label=f"Best Val (Epoch {best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_title(f"Loss Curves: {model_name} ({strategy})", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs_range, history["train_acc"], "o-", label="Train Acc (%)", color="#2ca02c")
    ax2.plot(epochs_range, history["val_acc"], "s-", label="Val Acc (%)", color="#d62728")
    ax2.axvline(best_epoch, linestyle="--", color="gray", label=f"Best Val (Epoch {best_epoch})")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Accuracy Curves: {model_name} ({strategy})", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curve_fig_path = cfg.FIGURES_DIR / f"{run_tag}_curves.png"
    plt.savefig(curve_fig_path, dpi=cfg.DPI)
    plt.close()

    # Figure B: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(cfg.CLASS_NAMES)),
        yticks=np.arange(len(cfg.CLASS_NAMES)),
        xticklabels=cfg.CLASS_NAMES,
        yticklabels=cfg.CLASS_NAMES,
        title=f"EuroSAT Confusion Matrix: {model_name} [{strategy}]\nAccuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%",
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate numbers
    thresh = np.array(cm).max() / 2.0
    for i in range(len(cfg.CLASS_NAMES)):
        for j in range(len(cfg.CLASS_NAMES)):
            ax.text(j, i, format(cm[i][j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black",
                    fontsize=9)
    plt.tight_layout()
    cm_fig_path = cfg.FIGURES_DIR / f"{run_tag}_cm.png"
    plt.savefig(cm_fig_path, dpi=cfg.DPI)
    plt.close()

    print(f"[Figures] Saved curves to: {curve_fig_path} (DPI={cfg.DPI})")
    print(f"[Figures] Saved confusion matrix to: {cm_fig_path} (DPI={cfg.DPI})")

    return result_data


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Training Framework for Vision Transformers on EuroSAT")
    parser.add_argument("--model", type=str, default="vit",
                        help="Model family ('vit', 'deit', 'swin') or explicit timm name")
    parser.add_argument("--strategy", type=str, default="unfreeze_last_3",
                        choices=["scratch", "linear_probe", "full_ft", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"],
                        help="Fine-tuning strategy")
    parser.add_argument("--seed", type=int, default=cfg.PRIMARY_SEED, help="Random seed (42, 43, 44)")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="Batch size (paper=16)")
    parser.add_argument("--max_epochs", type=int, default=cfg.MAX_EPOCHS, help="Max epochs (default=15)")
    parser.add_argument("--patience", type=int, default=cfg.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE, help="SGD learning rate (default=0.001)")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-epoch smoke test on a 2000-image subset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Map family to model string if family key provided
    if args.model.lower() in ["vit", "deit", "swin"]:
        model_name = cfg.get_model_name(args.model.lower())
    else:
        model_name = args.model

    train_and_evaluate(
        model_name=model_name,
        strategy=args.strategy,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.lr,
        smoke_test=args.smoke_test,
    )
