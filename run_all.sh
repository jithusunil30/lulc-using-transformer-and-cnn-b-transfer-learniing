#!/usr/bin/env bash
# ==============================================================================
# run_all.sh - Complete End-to-End Replication & Evaluation Pipeline
# ==============================================================================
# Research Project:
# "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
# for LULC, with Quantitative Evaluation of Explanations"
#
# Reference Paper:
# Khan et al., "Comparative analysis of vision transformers and CNNs for land
# use and land cover classification", Scientific Reports 14, 16744 (2024).
#
# Execution Pipeline:
#   Phase 1: Dataset download, stratified 70/15/15 split, transforms & benchmark
#   Phase 2 & 3: Model training matrix (ViT, DeiT, Swin x 6 strategies)
#   Phase 4: Efficiency analysis, Pareto frontiers & transfer learning critique
#   Phase 5: Quantitative XAI (IG, SmoothGrad, Grad-CAM, Rollout, Faithfulness, Sanity)
#   Phase 6: Corruption robustness (4 corruptions x 3 severities) & t-SNE manifolds
#   Phase 7: Programmatic report skeleton generation (report/report_skeleton.md)
# ==============================================================================

set -e # Exit immediately if a command fails

# Color output definitions
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}==============================================================================${RESET}"
echo -e "${BOLD}${CYAN}  Vision Transformers for LULC: End-to-End Scientific Replication Pipeline   ${RESET}"
echo -e "${BOLD}${CYAN}  Reference: Khan et al. (Scientific Reports 2024)                            ${RESET}"
echo -e "${BOLD}${CYAN}==============================================================================${RESET}"

# Default parameters
SEEDS="42"
BATCH_SIZE="32"
SKIP_TRAIN=false
PYTHON_CMD="python"

# Parse optional command line flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seeds) SEEDS="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --skip-train) SKIP_TRAIN=true ;;
        --python) PYTHON_CMD="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seeds <SEEDS>        Space-separated seeds (default: '42', e.g. '42 43 44')"
            echo "  --batch-size <N>       Training batch size (default: 32)"
            echo "  --skip-train           Skip training if checkpoints and result JSONs already exist"
            echo "  --python <CMD>         Python binary path (default: python)"
            echo "  -h, --help             Show this help message and exit"
            exit 0
            ;;
        *) echo -e "${RED}Unknown parameter: $1${RESET}"; exit 1 ;;
    esac
    shift
done

START_TIME=$(date +%s)

# Helper function to print phase headers
print_phase_header() {
    local phase_num="$1"
    local phase_title="$2"
    echo ""
    echo -e "${BOLD}${BLUE}------------------------------------------------------------------------------${RESET}"
    echo -e "${BOLD}${GREEN}[Phase ${phase_num}] ${phase_title}${RESET}"
    echo -e "${BOLD}${BLUE}------------------------------------------------------------------------------${RESET}"
    echo -e "${CYAN}Timestamp:${RESET} $(date '+%Y-%m-%d %H:%M:%S')"
}

# ------------------------------------------------------------------------------
# Pre-Flight Environment Checks
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${YELLOW}[Pre-Flight] Verifying Python environment and GPU acceleration...${RESET}"
$PYTHON_CMD -c "
import sys
import torch
print(f'Python version : {sys.version.split()[0]}')
print(f'PyTorch version: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
print(f'CUDA Available : {cuda_avail}')
if cuda_avail:
    print(f'Device Name    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM Total     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"

# ------------------------------------------------------------------------------
# Phase 1: Data Preparation & Stratified Splitting
# ------------------------------------------------------------------------------
print_phase_header "1" "EuroSAT Dataset Preparation, Stratified Split & Dataloader Benchmark"
$PYTHON_CMD src/dataset.py

# ------------------------------------------------------------------------------
# Phase 2 & 3: Model Training Matrix & Statistical Significance Testing
# ------------------------------------------------------------------------------
print_phase_header "2 & 3" "Model Training Matrix (ViT, DeiT, Swin x 6 Strategies) & Table 5 Reproduction"
if [ "$SKIP_TRAIN" = true ]; then
    echo -e "${YELLOW}Notice: --skip-train specified. Evaluating / formatting existing results...${RESET}"
fi

# Run Phase 3 orchestration (auto-resumes / skips completed runs)
$PYTHON_CMD src/run_phase3.py --seeds $SEEDS --batch_size $BATCH_SIZE

# ------------------------------------------------------------------------------
# Phase 4: Computational Efficiency & Pareto Frontiers
# ------------------------------------------------------------------------------
print_phase_header "4" "Computational Efficiency Benchmarks, Pareto Frontiers & Cost Critique"
$PYTHON_CMD src/efficiency_eval.py

# ------------------------------------------------------------------------------
# Phase 5: Quantitative Explainable AI (XAI) Suite
# ------------------------------------------------------------------------------
print_phase_header "5" "Quantitative Explainable AI (IG, SmoothGrad, Grad-CAM, Rollout & Misclassifications)"
$PYTHON_CMD src/xai_eval.py

# ------------------------------------------------------------------------------
# Phase 6: Out-of-Distribution Robustness & Representation Clustering
# ------------------------------------------------------------------------------
print_phase_header "6" "Corruption Robustness (4 Corruptions x 3 Severities) & t-SNE Manifold Clustering"
$PYTHON_CMD src/phase6_eval.py

# ------------------------------------------------------------------------------
# Phase 7: Programmatic Publication Report Skeleton Generation
# ------------------------------------------------------------------------------
print_phase_header "7" "Programmatic Report Skeleton Generation (report/report_skeleton.md)"
$PYTHON_CMD src/generate_report.py

# ------------------------------------------------------------------------------
# Phase 8: Geographic-Shift Dataset Construction (GEE & Dynamic World)
# ------------------------------------------------------------------------------
print_phase_header "8" "Geographic-Shift Dataset Construction & Verification Sample"
$PYTHON_CMD src/build_shift_dataset.py --target_per_class 300

# ------------------------------------------------------------------------------
# Phase 9: Geographic-Shift & Sensor Domain Evaluation
# ------------------------------------------------------------------------------
print_phase_header "9" "Geographic-Shift & Sensor Domain Evaluation (EuroSAT vs Europe vs India)"
$PYTHON_CMD src/evaluate_shift.py

# ------------------------------------------------------------------------------
# Phase 10: Model Calibration, Selective Prediction & Shifted XAI Faithfulness
# ------------------------------------------------------------------------------
print_phase_header "10" "Model Calibration, Selective Prediction & Shifted XAI Faithfulness"
$PYTHON_CMD src/phase10_eval.py --batch-size 64

# ------------------------------------------------------------------------------
# Phase 11: Scientific Rigor (Sensitivity Grid, CNN Baseline, LoRA & Reproducibility)
# ------------------------------------------------------------------------------
print_phase_header "11" "Methodological Rigor, Sensitivity, CNN Baseline, LoRA & Reproducibility"
$PYTHON_CMD src/phase11_rigor.py --batch-size 32

# Regenerate final publication report skeleton with all phases (1 through 11)
$PYTHON_CMD src/generate_report.py

# ------------------------------------------------------------------------------
# Pipeline Completion Summary
# ------------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_DURATION / 60))
TOTAL_SEC=$((TOTAL_DURATION % 60))

echo ""
echo -e "${BOLD}${CYAN}==============================================================================${RESET}"
echo -e "${BOLD}${GREEN}  REPLICATION PIPELINE SUCCESSFULLY COMPLETED!                               ${RESET}"
echo -e "${BOLD}${CYAN}==============================================================================${RESET}"
echo -e "${BOLD}Total Execution Time:${RESET} ${TOTAL_MIN}m ${TOTAL_SEC}s (${TOTAL_DURATION} seconds)"
echo -e "${BOLD}Key Generated Deliverables:${RESET}"
echo -e "  - ${CYAN}Final Publication Report:${RESET}  report/report_skeleton.md"
echo -e "  - ${CYAN}Table 5 Benchmark Data:${RESET}    results/seed42_table.csv"
echo -e "  - ${CYAN}Efficiency & Cost Audit:${RESET}   results/transfer_learning_cost_critique.csv"
echo -e "  - ${CYAN}Quantitative XAI Scores:${RESET}   results/xai_metrics.csv"
echo -e "  - ${CYAN}Corruption Robustness:${RESET}     results/corruption_robustness.csv"
echo -e "  - ${CYAN}t-SNE Silhouette Scores:${RESET}   results/tsne_silhouette_scores.csv"
echo -e "  - ${CYAN}Geographic Shift Dataset:${RESET}  data/geographic_shift/metadata.csv"
echo -e "  - ${CYAN}Manual Verification Sample:${RESET}data/geographic_shift/verification_sample/manual_verification_template.csv"
echo -e "  - ${CYAN}Calibration & ECE Matrix:${RESET}  results/phase10_calibration_summary.csv"
echo -e "  - ${CYAN}Selective Risk Metrics:${RESET}    results/phase10_selective_prediction.csv"
echo -e "  - ${CYAN}Shifted XAI Faithfulness:${RESET}  results/phase10_xai_shifted_faithfulness.csv"
echo -e "  - ${CYAN}Publication Figures:${RESET}       figures/*.png (DPI=200)"
echo -e "${BOLD}${CYAN}==============================================================================${RESET}"
