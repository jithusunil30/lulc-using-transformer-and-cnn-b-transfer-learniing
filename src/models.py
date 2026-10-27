"""
src/models.py - Transformer Model Architecture & Freezing Strategies
=====================================================================
Research Project: "Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers
for LULC, with Quantitative Evaluation of Explanations"
Reference: Khan et al. (Scientific Reports 2024)

Strategies:
1. 'scratch': Random weight initialization (no pretraining).
2. 'linear_probe': Pretrained backbone, all frozen except classification head.
3. 'full_ft': Pretrained backbone, all parameters unfrozen and trainable.
4. 'unfreeze_last_k': Pretrained backbone, freeze initial blocks, unfreeze last k blocks,
   plus final norm and classification head (k in {1, 2, 3}).

Block definitions:
- ViT / DeiT: model.blocks[i]
- Swin: Flatten blocks across all stages ([b for layer in model.layers for b in layer.blocks])
- Always unfreeze final normalization layer and classification head.
"""

from typing import Tuple, Dict, Any, List, Optional
import torch
import torch.nn as nn
import timm

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg


def get_flattened_blocks(model: nn.Module) -> List[nn.Module]:
    """
    Extracts a flat list of blocks for any supported architecture:
    - ViT & DeiT: model.blocks
    - Swin: Flatten blocks across all hierarchical stages (model.layers)
    - ConvNeXt: Flatten blocks across all stages (model.stages)
    - ResNet: Flatten residual blocks across layer1, layer2, layer3, layer4
    """
    if hasattr(model, "blocks"):
        # ViT, DeiT
        return list(model.blocks)
    elif hasattr(model, "layers"):
        # Swin Transformer
        flat_blocks = []
        for stage in model.layers:
            if hasattr(stage, "blocks"):
                for b in stage.blocks:
                    flat_blocks.append(b)
        return flat_blocks
    elif hasattr(model, "stages"):
        # ConvNeXt
        flat_blocks = []
        for stage in model.stages:
            if hasattr(stage, "blocks"):
                for b in stage.blocks:
                    flat_blocks.append(b)
        return flat_blocks
    elif hasattr(model, "layer1"):
        # ResNet
        flat_blocks = []
        for i in range(1, 5):
            layer = getattr(model, f"layer{i}", None)
            if layer is not None:
                for b in layer:
                    flat_blocks.append(b)
        return flat_blocks
    else:
        raise ValueError(f"Unrecognized architecture type: {type(model)}")


def unfreeze_head_and_norm(model: nn.Module) -> None:
    """
    Ensures final normalization layers and classification heads are trainable.
    """
    # Classification head
    if hasattr(model, "head") and model.head is not None:
        for p in model.head.parameters():
            p.requires_grad = True
    elif hasattr(model, "fc") and model.fc is not None:
        for p in model.fc.parameters():
            p.requires_grad = True
    elif hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            for p in classifier.parameters():
                p.requires_grad = True

    # Final normalization layer
    if hasattr(model, "norm") and model.norm is not None:
        for p in model.norm.parameters():
            p.requires_grad = True
    if hasattr(model, "fc_norm") and model.fc_norm is not None:
        for p in model.fc_norm.parameters():
            p.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Returns (trainable_params, total_params, trainable_pct).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return trainable, total, pct


def build_transformer_model(
    model_name: str,
    strategy: str = "full_ft",
    num_classes: int = cfg.NUM_CLASSES,
    k: int = 1,
    pretrained: Optional[bool] = None,
    lora_r: int = 16,
    lora_targets: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Builds and configures a Vision Transformer, DeiT, Swin, or CNN baseline (ConvNeXt / ResNet)
    according to the specified fine-tuning, LoRA, or scratch strategy.

    Parameters:
    -----------
    model_name : str
        timm model identifier (e.g. 'vit_small_patch16_224.augreg_in21k', 'convnext_tiny', 'resnet50')
    strategy : str
        One of {'scratch', 'linear_probe', 'full_ft', 'unfreeze_last_1', 'unfreeze_last_2', 'unfreeze_last_3', 'lora'}
    num_classes : int
        Number of output classes (10 for EuroSAT)
    k : int
        Number of blocks to unfreeze (used if strategy is 'unfreeze_last_k')
    pretrained : Optional[bool]
        Explicit override for loading timm pretrained weights (False avoids network calls)
    lora_r : int
        LoRA rank dimension (used when strategy is 'lora')
    lora_targets : Optional[List[str]]
        Target projection modules for LoRA
    """
    # Normalize strategy name
    strat = strategy.lower().strip()
    if strat.startswith("unfreeze_last_"):
        try:
            k = int(strat.split("_")[-1])
            strat_type = "unfreeze_last_k"
        except ValueError:
            strat_type = strat
    else:
        strat_type = strat

    if pretrained is not None:
        is_pretrained = bool(pretrained)
    else:
        is_pretrained = (strat != "scratch")

    # 1. Instantiate model from timm
    model = timm.create_model(
        model_name,
        pretrained=is_pretrained,
        num_classes=num_classes,
    )

    all_blocks = get_flattened_blocks(model)
    total_blocks = len(all_blocks)

    # 2. Apply strategy
    if strat == "scratch" or strat == "full_ft":
        # All parameters trainable
        for p in model.parameters():
            p.requires_grad = True

    elif strat == "linear_probe":
        # Freeze entire backbone
        for p in model.parameters():
            p.requires_grad = False
        # Train head only
        unfreeze_head_and_norm(model)

    elif strat_type == "unfreeze_last_k":
        if k < 1 or k > total_blocks:
            raise ValueError(f"k must be between 1 and {total_blocks}, got {k}")

        # Freeze all parameters first
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze the last k blocks
        selected_blocks = all_blocks[-k:]
        for blk in selected_blocks:
            for p in blk.parameters():
                p.requires_grad = True

        # Always also train final norm + classification head
        unfreeze_head_and_norm(model)

    elif strat == "lora":
        from peft import LoraConfig, get_peft_model
        # Freeze all parameters first
        for p in model.parameters():
            p.requires_grad = False

        if lora_targets is None:
            if "vit" in model_name.lower() or "deit" in model_name.lower():
                lora_targets = ["qkv"]
            elif "swin" in model_name.lower():
                lora_targets = ["qkv", "proj"]
            elif "convnext" in model_name.lower() or "resnet" in model_name.lower():
                lora_targets = ["fc1", "fc2", "conv"]
            else:
                lora_targets = ["qkv"]

        modules_to_save = ["head"]
        if hasattr(model, "fc"):
            modules_to_save.append("fc")
        if hasattr(model, "norm"):
            modules_to_save.append("norm")

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=lora_targets,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_cfg)

    else:
        raise ValueError(f"Unknown fine-tuning strategy: '{strategy}'")

    trainable_params, total_params, pct_trainable = count_parameters(model)

    metadata = {
        "model_name": model_name,
        "strategy": strategy,
        "strategy_type": strat_type,
        "k_blocks_unfrozen": k if strat_type == "unfreeze_last_k" else (total_blocks if strat in ["scratch", "full_ft"] else 0),
        "total_blocks": total_blocks,
        "is_pretrained": is_pretrained,
        "num_classes": num_classes,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(pct_trainable, 2),
        "lora_r": lora_r if strat == "lora" else None,
    }

    return model, metadata


# Convenient alias
build_model = build_transformer_model


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Model Builder & Freezing Strategies")
    print("=" * 70)

    test_models = [
        "vit_small_patch16_224.augreg_in21k",
        "deit_small_patch16_224.fb_in1k",
        "swin_tiny_patch4_window7_224.ms_in22k",
    ]

    strategies = ["scratch", "linear_probe", "full_ft", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"]

    for m_name in test_models:
        print(f"\n--- Model: {m_name} ---")
        for st in strategies:
            m, meta = build_transformer_model(m_name, strategy=st)
            print(f"  Strategy: {st:<18} | Trainable: {meta['trainable_params']:>10,} / {meta['total_params']:>10,} ({meta['trainable_pct']:>5.2f}%) | Blocks: {meta['total_blocks']}")
    print("=" * 70)
