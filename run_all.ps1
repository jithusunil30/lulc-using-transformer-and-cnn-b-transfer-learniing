<#
.SYNOPSIS
run_all.ps1 - Native PowerShell Runner for the Complete Replication Pipeline

.DESCRIPTION
Executes Phases 1 through 7 sequentially in Windows PowerShell:
  Phase 1: EuroSAT dataset download, 70/15/15 stratified split, dataloader benchmark
  Phase 2 & 3: Model training matrix (ViT, DeiT, Swin x 6 strategies)
  Phase 4: Efficiency analysis, Pareto frontiers & transfer learning cost critique
  Phase 5: Quantitative XAI suite (IG, SmoothGrad, Grad-CAM, Rollout, Deletion/Insertion)
  Phase 6: Corruption robustness (4 corruptions x 3 severities) & t-SNE manifold clustering
  Phase 7: Programmatic report skeleton generation (report/report_skeleton.md)

.PARAMETER Seeds
Space-separated seeds (default: "42")

.PARAMETER BatchSize
Training batch size (default: 32)
#>

param(
    [string]$Seeds = "42",
    [int]$BatchSize = 32,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  Vision Transformers for LULC: End-to-End Scientific Replication Pipeline   " -ForegroundColor Cyan
Write-Host "  Reference: Khan et al. (Scientific Reports 2024)                            " -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

$StartTime = Get-Date

# Environment check
Write-Host "`n[Pre-Flight] Verifying Python and PyTorch GPU environment..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Phase 1
Write-Host "`n[Phase 1] EuroSAT Dataset Preparation & Split Verification..." -ForegroundColor Green
python src/dataset.py

# Phase 2 & 3
Write-Host "`n[Phase 2 & 3] Model Training Matrix & Table 5 Reproduction..." -ForegroundColor Green
python src/run_phase3.py --seeds $Seeds --batch_size $BatchSize

# Phase 4
Write-Host "`n[Phase 4] Computational Efficiency Benchmarks & Pareto Frontiers..." -ForegroundColor Green
python src/efficiency_eval.py

# Phase 5
Write-Host "`n[Phase 5] Quantitative Explainable AI (XAI) Suite..." -ForegroundColor Green
python src/xai_eval.py

# Phase 6
Write-Host "`n[Phase 6] Corruption Robustness & t-SNE Manifold Clustering..." -ForegroundColor Green
python src/phase6_eval.py

# Phase 7
Write-Host "`n[Phase 7] Generating Final Publication Report Skeleton..." -ForegroundColor Green
python src/generate_report.py

# Phase 8
Write-Host "`n[Phase 8] Geographic-Shift Dataset Construction & Verification Sample..." -ForegroundColor Green
python src/build_shift_dataset.py --target_per_class 300

# Phase 9
Write-Host "`n[Phase 9] Geographic-Shift & Sensor Domain Evaluation..." -ForegroundColor Green
python src/evaluate_shift.py

# Phase 10
Write-Host "`n[Phase 10] Model Calibration, Selective Prediction & Shifted XAI Faithfulness..." -ForegroundColor Green
python src/phase10_eval.py --batch-size 64

# Phase 11
Write-Host "`n[Phase 11] Methodological Rigor: Sensitivity Grid, CNN Baseline, LoRA & Reproducibility Check..." -ForegroundColor Green
python src/phase11_rigor.py --batch-size 32

# Regenerate report with all empirical findings
Write-Host "`n[Final Report] Compiling Final Publication Report Skeleton with All 11 Phases..." -ForegroundColor Green
python src/generate_report.py

$EndTime = Get-Date
$Duration = $EndTime - $StartTime
Write-Host "`n==============================================================================" -ForegroundColor Cyan
Write-Host "  REPLICATION PIPELINE COMPLETED IN $($Duration.Minutes)m $($Duration.Seconds)s!  " -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
