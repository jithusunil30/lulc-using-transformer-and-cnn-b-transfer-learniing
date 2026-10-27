"""
tests/test_demo_app.py - Pytest Test Suite for Demo Application
===============================================================
Tests:
1. test_model_loading: Validates checkpoint loading, eval mode, and (1, 10) logits shape.
2. test_tiling_shapes: Validates 256x256 image tiling into 16 patches (4x4) and area summation.
3. test_explanation_output_shape: Validates Grad-CAM, IG, and Attention Rollout output shapes (224, 224).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from app.registry import (
    load_model,
    predict_patch,
    tile_and_predict_scene,
    get_registered_models,
    COARSE_CLASSES
)
from app.xai import (
    generate_attribution_map,
    render_attribution_overlay
)


@pytest.fixture(scope="module")
def default_model_name():
    models = get_registered_models()
    # Prioritize ViT unfreeze_last_3
    for m in models:
        if "unfreeze last 3" in m.lower():
            return m
    return models[0]


def test_model_loading(default_model_name):
    """
    Test 1: Model Loading & Inference Mode Verification
    Ensures checkpoint loads correctly onto CPU, model is in eval mode,
    and output tensor has shape (1, 10).
    """
    model, entry = load_model(default_model_name)

    assert isinstance(model, nn.Module), "Loaded model must be a torch.nn.Module"
    assert not model.training, "Loaded model must be in evaluation mode (model.eval())"
    assert "timm_model_id" in entry, "Metadata entry must contain timm_model_id"

    # Forward pass with dummy tensor
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        logits = model(dummy_input)

    assert logits.shape == (1, 10), f"Expected logits shape (1, 10), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits must not contain NaN values"

    # Test high-level predict_patch API
    dummy_img = Image.new("RGB", (64, 64), color=(60, 120, 80))
    pred_res = predict_patch(default_model_name, dummy_img, apply_calibration=True)

    assert pred_res["top_class"] in cfg.CLASS_NAMES
    assert pred_res["coarse_class"] in COARSE_CLASSES
    assert 0.0 <= pred_res["max_conf"] <= 1.0
    assert len(pred_res["top_classes"]) == 3


def test_tiling_shapes(default_model_name):
    """
    Test 2: Spatial Tiling & Dimension Consistency Verification
    Ensures a 256x256 scene is tiled into 16 patches (4x4) with stride 64,
    and output maps and area coverage sum to 100%.
    """
    scene_w, scene_h = 256, 256
    dummy_scene = Image.new("RGB", (scene_w, scene_h), color=(30, 80, 140))

    stride = 64
    tile_size = 64
    res = tile_and_predict_scene(
        default_model_name,
        dummy_scene,
        tile_size=tile_size,
        stride=stride,
        apply_calibration=True
    )

    expected_cols = scene_w // stride  # 4
    expected_rows = scene_h // stride  # 4
    expected_total = expected_cols * expected_rows  # 16

    assert res["n_cols"] == expected_cols, f"Expected {expected_cols} columns, got {res['n_cols']}"
    assert res["n_rows"] == expected_rows, f"Expected {expected_rows} rows, got {res['n_rows']}"
    assert res["total_tiles"] == expected_total, f"Expected {expected_total} tiles, got {res['total_tiles']}"
    assert len(res["tile_data"]) == expected_total

    # Verify classification and confidence maps dimensions
    assert res["classification_map"].size == (scene_w, scene_h)
    assert res["confidence_heatmap"].size == (scene_w, scene_h)
    assert res["blended_map"].size == (scene_w, scene_h)

    # Verify area percentages sum to 100%
    area_sum = sum(item["Area (%)"] for item in res["area_percentages"])
    assert 99.0 <= area_sum <= 101.0, f"Area percentages must sum to ~100%, got {area_sum}%"


def test_explanation_output_shape(default_model_name):
    """
    Test 3: Explainability Engine Attribution Output Shape & Value Bounds
    Verifies that Grad-CAM, Integrated Gradients, and Attention Rollout
    return 2D attribution heatmaps of shape (224, 224) normalized to [0, 1],
    and that 95th-percentile overlays render successfully.
    """
    dummy_patch = Image.new("RGB", (64, 64), color=(100, 150, 70))

    methods_to_test = [
        "Grad-CAM (pytorch-grad-cam)",
        "Integrated Gradients (Captum, n=30)",
        "Attention Rollout (ViT/DeiT)"
    ]

    for method in methods_to_test:
        attr_map = generate_attribution_map(default_model_name, dummy_patch, method=method)

        assert isinstance(attr_map, np.ndarray), f"{method} must return a numpy.ndarray"
        assert attr_map.shape == (224, 224), f"{method} output shape must be (224, 224), got {attr_map.shape}"
        assert not np.isnan(attr_map).any(), f"{method} attribution map must not contain NaN"
        assert attr_map.min() >= -1e-6, f"{method} values must be >= 0.0, got min {attr_map.min()}"
        assert attr_map.max() <= 1.0 + 1e-6, f"{method} values must be <= 1.0, got max {attr_map.max()}"

    # Verify standard and 95th-percentile overlays
    gradcam_attr = generate_attribution_map(default_model_name, dummy_patch, method="Grad-CAM (pytorch-grad-cam)")

    overlay_standard = render_attribution_overlay(dummy_patch, gradcam_attr, use_95th_percentile=False)
    assert isinstance(overlay_standard, Image.Image)
    assert overlay_standard.size == (224, 224)
    assert overlay_standard.mode == "RGB"

    overlay_95th = render_attribution_overlay(dummy_patch, gradcam_attr, use_95th_percentile=True)
    assert isinstance(overlay_95th, Image.Image)
    assert overlay_95th.size == (224, 224)
    assert overlay_95th.mode == "RGB"


def test_app_callbacks(default_model_name):
    """
    Test 4: Full App Callbacks End-to-End Verification
    Directly tests classify_patch_fn, run_scene_mapping, and run_comparison_top4
    to ensure zero runtime exceptions during user interaction.
    """
    from app.main import classify_patch_fn, run_scene_mapping, run_comparison_top4

    dummy_img = Image.new("RGB", (64, 64), color=(80, 140, 90))

    # Test Tab 1 callback
    warn_html, chart, coarse_html, overlay = classify_patch_fn(
        dummy_img,
        default_model_name,
        "Grad-CAM (pytorch-grad-cam)",
        False
    )
    assert warn_html is not None and len(warn_html) > 0
    assert chart is not None
    assert coarse_html is not None and "Coarse Land-Cover Class" in coarse_html
    assert isinstance(overlay, Image.Image)

    # Test Tab 2 callback
    dummy_scene = Image.new("RGB", (128, 128), color=(40, 100, 150))
    blended, conf_map, area_pcts, legend = run_scene_mapping(
        dummy_scene,
        default_model_name,
        "Grad-CAM (pytorch-grad-cam)"
    )
    assert isinstance(blended, Image.Image)
    assert isinstance(conf_map, Image.Image)
    assert len(area_pcts) > 0
    assert "EuroSAT" in legend

    # Test Tab 4 multi-model callback
    comp_outs = run_comparison_top4(dummy_img, "Grad-CAM (pytorch-grad-cam)")
    assert len(comp_outs) == 8  # 4 models * 2 outputs (overlay, info)
    for i in range(0, 8, 2):
        assert isinstance(comp_outs[i], Image.Image), f"Model {i//2} output image must be a PIL Image"
        assert isinstance(comp_outs[i+1], str), f"Model {i//2} output info must be a markdown string"
