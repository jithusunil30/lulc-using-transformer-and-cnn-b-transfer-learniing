"""
src/config.py - Centralized Configuration Module
================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024)

Config-driven parameters for model backbones, seeds, directory paths,
hyperparameters, and the 70/15/15 stratified data split.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class ProjectConfig:
    # --------------------------------------------------------------------------
    # 1. Budget Mode & Model Selection
    # --------------------------------------------------------------------------
    BUDGET_MODE: bool = True

    # Pretrained model backbones verified via timm.list_models(pretrained=True)
    BUDGET_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "vit": "vit_small_patch16_224.augreg_in21k",
        "deit": "deit_small_patch16_224.fb_in1k",
        "swin": "swin_tiny_patch4_window7_224.ms_in22k",
    })

    FULL_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "vit": "vit_base_patch16_224.augreg_in21k",
        "deit": "deit_base_patch16_224.fb_in1k",
        "swin": "swin_small_patch4_window7_224.ms_in22k",
    })

    # --------------------------------------------------------------------------
    # 2. Reproducibility & Seeds
    # --------------------------------------------------------------------------
    SEEDS: List[int] = field(default_factory=lambda: [42, 43, 44])
    PRIMARY_SEED: int = 42

    # --------------------------------------------------------------------------
    # 3. Directory Paths
    # --------------------------------------------------------------------------
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    FIGURES_DIR: Path = PROJECT_ROOT / "figures"
    CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
    CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
    NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
    SRC_DIR: Path = PROJECT_ROOT / "src"
    SPLITS_PATH: Path = PROJECT_ROOT / "data" / "splits.json"

    # --------------------------------------------------------------------------
    # 4. EuroSAT Dataset Settings
    # --------------------------------------------------------------------------
    DATASET_NAME: str = "EuroSAT_RGB"
    NUM_CLASSES: int = 10
    IMAGE_SIZE: int = 224  # Standard ViT / DeiT / Swin input resolution
    INPUT_CHANNELS: int = 3
    CLASS_NAMES: List[str] = field(default_factory=lambda: [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ])

    # Stratified split: 70% Train, 15% Val, 15% Test
    TRAIN_SPLIT: float = 0.70
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15

    # Fallback normalization (overridden dynamically via timm's pretrained config)
    DEFAULT_MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    DEFAULT_STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # --------------------------------------------------------------------------
    # 5. Training Hyperparameters (matching Khan et al. 2024)
    # --------------------------------------------------------------------------
    BATCH_SIZE: int = 16
    MAX_EPOCHS: int = 10
    LEARNING_RATE: float = 0.001
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 5
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True

    # --------------------------------------------------------------------------
    # 6. Fine-Tuning Strategies
    # --------------------------------------------------------------------------
    STRATEGIES: List[str] = field(default_factory=lambda: [
        "scratch",
        "full",
        "unfreeze_last_3",
        "unfreeze_last_2",
        "linear_probe",
    ])

    # --------------------------------------------------------------------------
    # 7. Hardware & Precision
    # --------------------------------------------------------------------------
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = torch.cuda.is_available()

    # --------------------------------------------------------------------------
    # 8. Visualization
    # --------------------------------------------------------------------------
    DPI: int = 200

    def get_model_name(self, model_family: str) -> str:
        family = model_family.lower()
        mapping = self.BUDGET_MODELS if self.BUDGET_MODE else self.FULL_MODELS
        if family not in mapping:
            raise ValueError(f"Unknown model family '{model_family}'. Choose from {list(mapping.keys())}")
        return mapping[family]

    def ensure_directories(self) -> None:
        for p in [
            self.DATA_DIR,
            self.RESULTS_DIR,
            self.FIGURES_DIR,
            self.CHECKPOINTS_DIR,
            self.CONFIGS_DIR,
            self.NOTEBOOKS_DIR,
            self.SRC_DIR,
        ]:
            p.mkdir(parents=True, exist_ok=True)


# Global configuration instance
cfg = ProjectConfig()
cfg.ensure_directories()


if __name__ == "__main__":
    print("=" * 70)
    print("LULC ViT Project Configuration Initialized")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {cfg.DEVICE} (AMP enabled: {cfg.USE_AMP})")
    print(f"Budget Mode: {cfg.BUDGET_MODE}")
    print(f"Data Splits: Train={cfg.TRAIN_SPLIT*100:.0f}%, Val={cfg.VAL_SPLIT*100:.0f}%, Test={cfg.TEST_SPLIT*100:.0f}%")
    print("=" * 70)
