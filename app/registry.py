"""
app/registry.py - Demo Model Registry & CPU-Friendly Inference Engine
====================================================================
Loads models lazily on demand, caches active checkpoints in memory,
applies temperature calibration, and provides batched patch/scene inference.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.cm as cm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.models import build_transformer_model

# In-memory model cache for lazy CPU inference
_MODEL_CACHE: Dict[str, Tuple[nn.Module, Dict[str, Any]]] = {}

# Load demo models configuration
DEMO_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo_models.json"
CLASS_MAP_PATH = PROJECT_ROOT / "configs" / "class_map.json"

with open(DEMO_CONFIG_PATH, "r", encoding="utf-8") as f:
    DEMO_MODELS_META = json.load(f)["models"]

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    CLASS_MAP_DATA = json.load(f)

EUROSAT_TO_COARSE: Dict[str, str] = CLASS_MAP_DATA.get("eurosat_to_coarse", {})
COARSE_CLASSES: List[str] = CLASS_MAP_DATA.get("coarse_classes", ["Forest", "Crop", "Water", "Built", "Grass/Shrub"])

# Color palettes for maps
COARSE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Forest": (34, 139, 34),       # Forest Green
    "Crop": (229, 168, 35),        # Goldenrod Amber
    "Water": (30, 144, 255),       # Dodger Blue
    "Built": (220, 20, 60),        # Crimson Red
    "Grass/Shrub": (144, 238, 144) # Soft Spring Green
}

EUROSAT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "AnnualCrop": (244, 208, 63),
    "Forest": (30, 132, 73),
    "HerbaceousVegetation": (82, 190, 128),
    "Highway": (231, 76, 60),
    "Industrial": (136, 78, 160),
    "Pasture": (169, 223, 191),
    "PermanentCrop": (212, 172, 13),
    "Residential": (230, 126, 34),
    "River": (52, 152, 219),
    "SeaLake": (41, 128, 185),
}

# Image transforms
INFERENCE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=cfg.DEFAULT_MEAN, std=cfg.DEFAULT_STD)
])


def get_registered_models() -> List[str]:
    """Returns human-readable names of all registered demo models."""
    return [m["name"] for m in DEMO_MODELS_META]


def get_model_entry(model_name: str) -> Dict[str, Any]:
    """Retrieves metadata dict for a specific registered model name."""
    for m in DEMO_MODELS_META:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Model '{model_name}' not registered in configs/demo_models.json")


def load_model(model_name: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Lazily loads and caches the specified model onto CPU.
    Guarantees thread-safe inference mode and lightweight memory footprint.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    entry = get_model_entry(model_name)
    ckpt_path = PROJECT_ROOT / entry["checkpoint_path"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Build model backbone (pretrained=False avoids network downloads)
    model, _ = build_transformer_model(
        model_name=entry["timm_model_id"],
        strategy=entry["strategy"],
        num_classes=cfg.NUM_CLASSES,
        pretrained=False
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL_CACHE[model_name] = (model, entry)
    return model, entry


def predict_patch(
    model_name: str,
    pil_img: Image.Image,
    apply_calibration: bool = True,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Performs calibrated inference on a single image patch.
    Capped and CPU-friendly.
    """
    model, entry = load_model(model_name)

    # Cap oversized images before feeding to model
    if max(pil_img.size) > 1024:
        pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    rgb_img = pil_img.convert("RGB")
    tensor = INFERENCE_TRANSFORM(rgb_img).unsqueeze(0)

    temperature = float(entry.get("temperature", 1.0))

    with torch.inference_mode():
        logits = model(tensor)
        if apply_calibration and temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    # Sort descending
    sorted_indices = np.argsort(-probs)
    top_indices = sorted_indices[:top_k]

    top_classes = [
        {"class": cfg.CLASS_NAMES[idx], "prob": float(probs[idx]), "pct": float(probs[idx] * 100.0)}
        for idx in top_indices
    ]

    all_probs = {cfg.CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
    top_class = cfg.CLASS_NAMES[top_indices[0]]
    max_conf = float(probs[top_indices[0]])
    coarse_class = EUROSAT_TO_COARSE.get(top_class, "Unknown")

    return {
        "model_name": model_name,
        "top_class": top_class,
        "max_conf": max_conf,
        "coarse_class": coarse_class,
        "top_classes": top_classes,
        "all_probs": all_probs,
        "temperature": temperature,
        "is_low_confidence": bool(max_conf < 0.60)
    }


def tile_and_predict_scene(
    model_name: str,
    pil_scene: Image.Image,
    tile_size: int = 64,
    stride: int = 64,
    apply_calibration: bool = True
) -> Dict[str, Any]:
    """
    Tiles an arbitrary large RGB scene into 64x64 patches with given stride,
    batches the inference on CPU, and synthesizes color-coded LULC map,
    confidence heatmap, and class area statistics.
    """
    model, entry = load_model(model_name)
    temperature = float(entry.get("temperature", 1.0))

    # Cap scene dimension if excessively huge for responsive web app
    orig_w, orig_h = pil_scene.size
    if max(orig_w, orig_h) > 1024:
        pil_scene = pil_scene.copy()
        pil_scene.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        orig_w, orig_h = pil_scene.size

    rgb_scene = pil_scene.convert("RGB")

    n_cols = max(1, orig_w // stride)
    n_rows = max(1, orig_h // stride)

    crop_w = n_cols * stride
    crop_h = n_rows * stride
    cropped_scene = rgb_scene.crop((0, 0, crop_w, crop_h))

    tiles_tensors = []
    tile_coords = []
    tile_crops = []

    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * stride
            y0 = r * stride
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            tile_img = cropped_scene.crop((x0, y0, x1, y1))
            tile_crops.append(tile_img)
            tiles_tensors.append(INFERENCE_TRANSFORM(tile_img))
            tile_coords.append((r, c, x0, y0, x1, y1))

    batch_tensor = torch.stack(tiles_tensors)

    with torch.inference_mode():
        logits = model(batch_tensor)
        if apply_calibration and temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1).numpy()

    pred_indices = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Build color-coded classification map & confidence heatmap
    class_map_arr = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    conf_map_arr = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    import matplotlib as mpl
    plasma = mpl.colormaps["plasma"]

    tile_data = []
    class_counts: Dict[str, int] = {cls: 0 for cls in cfg.CLASS_NAMES}
    coarse_counts: Dict[str, int] = {c: 0 for c in COARSE_CLASSES}

    for idx, (r, c, x0, y0, x1, y1) in enumerate(tile_coords):
        cls_idx = pred_indices[idx]
        conf = float(confidences[idx])
        cls_name = cfg.CLASS_NAMES[cls_idx]
        coarse_name = EUROSAT_TO_COARSE.get(cls_name, "Unknown")

        class_counts[cls_name] += 1
        if coarse_name in coarse_counts:
            coarse_counts[coarse_name] += 1

        color = EUROSAT_COLORS.get(cls_name, (128, 128, 128))
        class_map_arr[y0:y1, x0:x1] = color

        conf_rgb = (np.array(plasma(conf)[:3]) * 255.0).astype(np.uint8)
        conf_map_arr[y0:y1, x0:x1] = conf_rgb

        tile_data.append({
            "tile_idx": idx,
            "row": r,
            "col": c,
            "coords": (x0, y0, x1, y1),
            "class_name": cls_name,
            "coarse_class": coarse_name,
            "confidence": conf,
            "crop": tile_crops[idx]
        })

    total_tiles = len(tile_coords)
    area_percentages = [
        {"Class": cls, "Coarse Group": EUROSAT_TO_COARSE.get(cls, "-"), "Tiles": count, "Area (%)": round(count / total_tiles * 100.0, 1)}
        for cls, count in class_counts.items() if count > 0
    ]
    area_percentages.sort(key=lambda x: -x["Area (%)"])

    # Blend classification map with original scene (50% overlay) for spatial grounding
    orig_np = np.array(cropped_scene)
    blended_map_arr = (0.55 * class_map_arr + 0.45 * orig_np).astype(np.uint8)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "stride": stride,
        "crop_size": (crop_w, crop_h),
        "total_tiles": total_tiles,
        "classification_map": Image.fromarray(class_map_arr),
        "blended_map": Image.fromarray(blended_map_arr),
        "confidence_heatmap": Image.fromarray(conf_map_arr),
        "area_percentages": area_percentages,
        "tile_data": tile_data,
        "cropped_scene": cropped_scene
    }
