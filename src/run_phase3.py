"""
src/run_phase3.py - Phase 3 Reproduction, Benchmarking & Statistical Testing
=============================================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024), Table 5

Key Capabilities:
- Runs all models (ViT, DeiT, Swin) x strategies (scratch, full_ft, unfreeze_last_k, linear_probe)
- Step 1: Executes Seed 42 first, displays comparative analysis, and generates seed 42 outputs
- Resumes automatically from existing results/*.json files (smart caching)
- Compares directly against Khan et al. (2024) Table 5 paper-reported values
- Performs paired statistical significance testing vs full_ft:
  * McNemar's test (with continuity correction or exact binomial)
  * Bootstrap 95% Confidence Interval for paired accuracy differences
- Generates publication-ready Table 5 CSV/JSON and figures (confusion matrices & bar charts at 200 DPI)
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.train import train_and_evaluate, clean_name


# Paper-reported values from Khan et al. (Scientific Reports 2024) Table 5
PAPER_REPORTED = {
    "vit": {
        "scratch": {"paper_acc": 79.26, "paper_loss": 0.5866, "paper_f1": 78.95, "paper_bal_acc": 78.52, "paper_params": 86567656},
        "full_ft": {"paper_acc": 98.67, "paper_loss": 0.0417, "paper_f1": 98.67, "paper_bal_acc": 98.60, "paper_params": 86567656},
        "unfreeze_last_3": {"paper_acc": 97.83, "paper_loss": 0.0625, "paper_f1": 97.84, "paper_bal_acc": 97.77, "paper_params": 21263616},
        "unfreeze_last_2": {"paper_acc": 97.65, "paper_loss": 0.0693, "paper_f1": 97.64, "paper_bal_acc": 97.40, "paper_params": 14175744},
        "unfreeze_last_1": {"paper_acc": None, "paper_loss": None, "paper_f1": None, "paper_bal_acc": None, "paper_params": None},
        "linear_probe": {"paper_acc": None, "paper_loss": None, "paper_f1": None, "paper_bal_acc": None, "paper_params": None},
    },
    "deit": {
        "scratch": {"paper_acc": 85.67, "paper_loss": 0.4231, "paper_f1": 85.58, "paper_bal_acc": 85.30, "paper_params": 86567656},
        "full_ft": {"paper_acc": 98.31, "paper_loss": 0.0502, "paper_f1": 98.32, "paper_bal_acc": 98.32, "paper_params": 86567656},
        "unfreeze_last_3": {"paper_acc": 96.76, "paper_loss": 0.1012, "paper_f1": 96.77, "paper_bal_acc": 96.67, "paper_params": 21263616},
        "unfreeze_last_2": {"paper_acc": 95.69, "paper_loss": 0.1427, "paper_f1": 95.70, "paper_bal_acc": 95.56, "paper_params": 14175744},
        "unfreeze_last_1": {"paper_acc": None, "paper_loss": None, "paper_f1": None, "paper_bal_acc": None, "paper_params": None},
        "linear_probe": {"paper_acc": None, "paper_loss": None, "paper_f1": None, "paper_bal_acc": None, "paper_params": None},
    },
    "swin": {
        "scratch": {"paper_acc": 74.91, "paper_loss": 0.7090, "paper_f1": 74.52, "paper_bal_acc": 73.68, "paper_params": 49728418},
        "full_ft": {"paper_acc": 98.28, "paper_loss": 0.0516, "paper_f1": 98.28, "paper_bal_acc": 98.23, "paper_params": 49728418},
        "unfreeze_last_3": {"paper_acc": 97.98, "paper_loss": 0.0564, "paper_f1": 97.98, "paper_bal_acc": 97.92, "paper_params": 48723348},
        "unfreeze_last_2": {"paper_acc": 98.17, "paper_loss": 0.0567, "paper_f1": 98.16, "paper_bal_acc": 97.90, "paper_params": 47750664},
        "unfreeze_last_1": {"paper_acc": None, "paper_loss": None, "paper_f1": None, "paper_bal_acc": None, "paper_params": None},
        "linear_probe": {"paper_acc": None, "paper_loss": None, "paper_f1": None, "paper_bal_acc": None, "paper_params": None},
    },
}


def mcnemar_test(y_true: np.ndarray, y_pred_strat: np.ndarray, y_pred_full: np.ndarray) -> Dict[str, Any]:
    """
    Computes McNemar's paired test comparing a strategy vs full_ft.
    b: strategy is correct, full_ft is wrong
    c: strategy is wrong, full_ft is correct
    """
    correct_strat = (y_pred_strat == y_true)
    correct_full = (y_pred_full == y_true)

    b = int(np.sum(correct_strat & ~correct_full))
    c = int(np.sum(~correct_strat & correct_full))

    if b + c == 0:
        statistic = 0.0
        p_val = 1.0
    elif b + c < 25:
        # Exact binomial test for small sample discordant pairs
        res = stats.binomtest(b, b + c, 0.5)
        p_val = float(res.pvalue)
        statistic = float(abs(b - c))
    else:
        # Edwards continuity correction
        statistic = float((abs(b - c) - 1.0) ** 2 / (b + c))
        p_val = float(1.0 - stats.chi2.cdf(statistic, df=1))

    return {
        "b_strat_correct_full_wrong": b,
        "c_strat_wrong_full_correct": c,
        "mcnemar_stat": round(statistic, 4),
        "p_value": round(p_val, 5),
        "is_significant": bool(p_val < 0.05),
    }


def bootstrap_accuracy_difference(
    y_true: np.ndarray,
    y_pred_strat: np.ndarray,
    y_pred_full: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes bootstrap 95% Confidence Interval for (Accuracy_Strategy - Accuracy_Full_FT) in percentage points.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    correct_strat = (y_pred_strat == y_true).astype(float)
    correct_full = (y_pred_full == y_true).astype(float)

    for _ in range(n_bootstraps):
        idxs = rng.randint(0, n, size=n)
        acc_strat = np.mean(correct_strat[idxs])
        acc_full = np.mean(correct_full[idxs])
        diffs.append((acc_strat - acc_full) * 100.0)

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_diff": round(mean_diff, 2),
        "ci_lower_95": round(ci_lower, 2),
        "ci_upper_95": round(ci_upper, 2),
        "ci_str": f"[{ci_lower:+.2f}%, {ci_upper:+.2f}%]",
    }


def run_or_load_experiment(
    family: str,
    strategy: str,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
) -> Dict[str, Any]:
    """Runs the training experiment or loads existing result JSON if already computed."""
    model_name = cfg.get_model_name(family)
    clean_m = clean_name(model_name)
    json_path = cfg.RESULTS_DIR / f"{clean_m}_{strategy}_seed{seed}.json"

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verify data has necessary fields
            if "test_metrics" in data and "parameters" in data:
                print(f"[Phase 3 Cache] Loaded existing results from: {json_path.name}")
                return data
        except Exception as e:
            print(f"[Phase 3 Cache] Warning: could not load {json_path.name} ({e}). Re-running...")

    print(f"\n[Phase 3 Run] Executing {family.upper()} ({model_name}) | Strategy: {strategy} | Seed: {seed}")
    data = train_and_evaluate(
        model_name=model_name,
        strategy=strategy,
        seed=seed,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=lr,
        smoke_test=False,
    )
    return data


def run_phase3(
    models: List[str] = ["vit", "deit", "swin"],
    strategies: List[str] = ["scratch", "full_ft", "unfreeze_last_3", "unfreeze_last_2", "unfreeze_last_1", "linear_probe"],
    seeds: List[int] = [42],
    batch_size: int = cfg.BATCH_SIZE,
    max_epochs: int = cfg.MAX_EPOCHS,
    patience: int = cfg.EARLY_STOPPING_PATIENCE,
    lr: float = cfg.LEARNING_RATE,
) -> pd.DataFrame:
    """
    Executes Phase 3 benchmark matrix across models, strategies, and seeds.
    """
    cfg.ensure_directories()
    all_runs: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {m: {s: {} for s in strategies} for m in models}

    print("=" * 80)
    print("PHASE 3: Full Reproduction & Benchmark Matrix (Khan et al. 2024 Table 5)")
    print(f"Models: {models}")
    print(f"Strategies: {strategies}")
    print(f"Seeds: {seeds}")
    print(f"Batch Size: {batch_size} | Max Epochs: {max_epochs} | Early Stopping Patience: {patience}")
    print("=" * 80)

    # 1. Run / Load all experiments
    for family in models:
        for strat in strategies:
            for seed in seeds:
                res = run_or_load_experiment(
                    family=family,
                    strategy=strat,
                    seed=seed,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    lr=lr,
                )
                all_runs[family][strat][seed] = res

    # 2. Build Structured Table Laid Out Like Paper's Table 5
    records = []
    best_strategy_per_model = {}

    for family in models:
        model_name = cfg.get_model_name(family)
        # Find full_ft baseline for each seed to perform paired significance test
        best_acc = -1.0
        best_strat = None

        for strat in strategies:
            # Collect metrics across seeds
            accs = [all_runs[family][strat][sd]["test_metrics"]["accuracy"] for sd in seeds]
            losses = [all_runs[family][strat][sd]["test_metrics"]["test_loss"] for sd in seeds]
            f1s = [all_runs[family][strat][sd]["test_metrics"]["macro_f1"] for sd in seeds]
            bal_accs = [all_runs[family][strat][sd]["test_metrics"]["balanced_accuracy"] for sd in seeds]
            eces = [all_runs[family][strat][sd]["test_metrics"]["expected_calibration_error"] for sd in seeds]
            times = [all_runs[family][strat][sd]["hardware_efficiency"]["total_wall_time_minutes"] for sd in seeds]
            fps_list = [all_runs[family][strat][sd]["hardware_efficiency"]["inference_throughput_img_per_sec"] for sd in seeds if all_runs[family][strat][sd]["hardware_efficiency"].get("inference_throughput_img_per_sec")]

            # Parameter counts from primary run
            primary_data = all_runs[family][strat][seeds[0]]
            params_trainable = primary_data["parameters"]["trainable_params"]
            params_total = primary_data["parameters"]["total_params"]
            pct_trainable = primary_data["parameters"]["trainable_pct"]

            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs)) if len(accs) > 1 else 0.0

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_strat = strat

            # Paper reported values
            paper_info = PAPER_REPORTED.get(family, {}).get(strat, {})
            paper_acc = paper_info.get("paper_acc", None)
            diff_vs_paper = round(mean_acc - paper_acc, 2) if paper_acc is not None else None

            # Significance testing vs full_ft on primary seed (seed 42)
            p_val = None
            ci_str = "-"
            diff_vs_full = None
            is_sig = None

            if "predictions" in primary_data and "predictions" in all_runs[family]["full_ft"][seeds[0]]:
                y_true = np.array(primary_data["predictions"]["y_true"])
                y_pred_strat = np.array(primary_data["predictions"]["y_pred"])
                y_pred_full = np.array(all_runs[family]["full_ft"][seeds[0]]["predictions"]["y_pred"])

                acc_full_primary = all_runs[family]["full_ft"][seeds[0]]["test_metrics"]["accuracy"]
                diff_vs_full = round(primary_data["test_metrics"]["accuracy"] - acc_full_primary, 2)

                if strat != "full_ft":
                    mcn = mcnemar_test(y_true, y_pred_strat, y_pred_full)
                    boot = bootstrap_accuracy_difference(y_true, y_pred_strat, y_pred_full, seed=seeds[0])
                    p_val = mcn["p_value"]
                    is_sig = mcn["is_significant"]
                    ci_str = boot["ci_str"]
                else:
                    ci_str = "[0.00%, 0.00%]"
                    diff_vs_full = 0.0
                    p_val = 1.0
                    is_sig = False

            rec = {
                "Model_Family": family.upper(),
                "Architecture": model_name,
                "Strategy": strat,
                "Trainable_Params": params_trainable,
                "Total_Params": params_total,
                "Trainable_Pct": pct_trainable,
                "Test_Loss": round(float(np.mean(losses)), 4),
                "Test_Accuracy": round(mean_acc, 2),
                "Test_Accuracy_Std": round(std_acc, 2),
                "Paper_Reported_Accuracy": paper_acc,
                "Diff_vs_Paper": diff_vs_paper,
                "Delta_vs_Full_FT": diff_vs_full,
                "Bootstrap_95_CI": ci_str,
                "McNemar_p_val": p_val,
                "Significant_vs_Full_FT": is_sig,
                "Macro_F1": round(float(np.mean(f1s)), 2),
                "Balanced_Accuracy": round(float(np.mean(bal_accs)), 2),
                "ECE": round(float(np.mean(eces)), 4),
                "Wall_Time_Minutes": round(float(np.mean(times)), 2),
                "Inference_Throughput_fps": round(float(np.mean(fps_list)), 2) if fps_list else None,
            }
            records.append(rec)

        best_strategy_per_model[family] = best_strat

    df_table = pd.DataFrame(records)

    # 3. Save Summary Tables
    is_single_seed = (len(seeds) == 1 and seeds[0] == 42)
    csv_filename = "seed42_table.csv" if is_single_seed else "main_table.csv"
    json_filename = "seed42_table.json" if is_single_seed else "main_table.json"

    csv_path = cfg.RESULTS_DIR / csv_filename
    json_path = cfg.RESULTS_DIR / json_filename

    df_table.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\n[Phase 3] Saved benchmark table to: {csv_path} and {json_path}")

    # 4. Generate Visualizations (DPI=200)
    # Figure A: Bar Chart of Accuracy with Comparisons (and Error Bars if multi-seed)
    _plot_accuracy_comparison(df_table, is_single_seed, seeds)

    # Figure B: Confusion Matrices for each Model's Best Performing Strategy
    _plot_best_confusion_matrices(all_runs, best_strategy_per_model, seeds[0])

    # 5. Print Formatted Table 5 to Console
    _print_table5_summary(df_table)

    return df_table


def _plot_accuracy_comparison(df: pd.DataFrame, is_single_seed: bool, seeds: List[int]):
    """Plots clean bar chart of accuracy across strategies and models with error bars."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(14, 6))

    families = df["Model_Family"].unique()
    strategies = df["Strategy"].unique()

    x = np.arange(len(families))
    width = 0.8 / len(strategies)

    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for i, strat in enumerate(strategies):
        sub = df[df["Strategy"] == strat]
        means = sub["Test_Accuracy"].values
        stds = sub["Test_Accuracy_Std"].values if not is_single_seed else None

        offset = (i - len(strategies) / 2 + 0.5) * width
        rects = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                       label=strat, color=colors[i % len(colors)], alpha=0.9)

        # Label accuracy on top of bar
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=35)

    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    title_suffix = f"(Seed {seeds[0]})" if is_single_seed else f"(Mean ± Std across seeds {seeds})"
    ax.set_title(f"LULC Classification Accuracy Across Transformer Architectures & Freezing Strategies {title_suffix}",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=12, fontweight="bold")
    ax.set_ylim(60.0, 102.0)
    ax.legend(title="Fine-Tuning Strategy", frameon=True, fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig_name = "accuracy_comparison_seed42.png" if is_single_seed else "main_accuracy_tradeoffs_with_errorbars.png"
    out_path = cfg.FIGURES_DIR / fig_name
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[Phase 3 Figures] Saved accuracy bar chart to: {out_path} (DPI={cfg.DPI})")


def _plot_best_confusion_matrices(all_runs: Dict[str, Any], best_strategies: Dict[str, str], seed: int):
    """Plots side-by-side confusion matrices for each model's best performing strategy."""
    families = list(best_strategies.keys())
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5.5))
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        strat = best_strategies[fam]
        run_data = all_runs[fam][strat][seed]
        cm = np.array(run_data["test_metrics"]["confusion_matrix"])
        acc = run_data["test_metrics"]["accuracy"]
        f1 = run_data["test_metrics"]["macro_f1"]

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{fam.upper()}: Best Strategy [{strat}]\nAcc: {acc:.2f}% | F1: {f1:.2f}%",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(len(cfg.CLASS_NAMES)))
        ax.set_yticks(range(len(cfg.CLASS_NAMES)))
        ax.set_xticklabels(cfg.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cfg.CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = cfg.FIGURES_DIR / "best_strategies_confusion_matrices.png"
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"[Phase 3 Figures] Saved best strategies confusion matrices to: {out_path} (DPI={cfg.DPI})")


def _print_table5_summary(df: pd.DataFrame):
    """Prints a formatted reproduction table laid out like Khan et al. Table 5."""
    print("\n" + "=" * 125)
    print(f"{'Model':<8}{'Strategy':<18}{'Trainable':<12}{'Params (%)':<12}{'Test Loss':<11}{'Ours Acc (%)':<14}{'Paper Acc (%)':<15}{'Delta Full (%)':<16}{'McNemar p':<11}{'Significant':<12}")
    print("=" * 125)

    for _, row in df.iterrows():
        p_val_str = f"{row['McNemar_p_val']:.4f}" if row['McNemar_p_val'] is not None else "N/A"
        sig_str = "YES (*)" if row['Significant_vs_Full_FT'] else ("NO" if row['Significant_vs_Full_FT'] is not None else "Baseline")
        paper_str = f"{row['Paper_Reported_Accuracy']:.2f}%" if row['Paper_Reported_Accuracy'] is not None else "-"
        delta_str = f"{row['Delta_vs_Full_FT']:+.2f}%" if row['Delta_vs_Full_FT'] is not None else "-"

        print(f"{row['Model_Family']:<8}"
              f"{row['Strategy']:<18}"
              f"{row['Trainable_Params']:<12,}"
              f"{row['Trainable_Pct']:<12.1f}"
              f"{row['Test_Loss']:<11.4f}"
              f"{row['Test_Accuracy']:<14.2f}"
              f"{paper_str:<15}"
              f"{delta_str:<16}"
              f"{p_val_str:<11}"
              f"{sig_str:<12}")
    print("=" * 125)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Full Reproduction & Benchmarking Matrix")
    parser.add_argument("--models", nargs="+", default=["vit", "deit", "swin"], help="Model families to evaluate")
    parser.add_argument("--strategies", nargs="+",
                        default=["scratch", "full_ft", "unfreeze_last_3", "unfreeze_last_2", "unfreeze_last_1", "linear_probe"],
                        help="Strategies to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds (e.g. 42 or 42 43 44)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default 32 for efficiency)")
    parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs (paper used 10)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
    args = parser.parse_args()

    run_phase3(
        models=args.models,
        strategies=args.strategies,
        seeds=args.seeds,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
    )
