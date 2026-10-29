"""
app/xai.py - Interactive Explainability Engine for Gradio
=========================================================
Implements CPU-friendly attribution methods:
1. Integrated Gradients via Captum (n_steps=30, zero baseline)
2. Grad-CAM via pytorch-grad-cam with architecture-aware 2D reshape transforms
3. Attention Rollout for Vision Transformers (Abnar & Zuidema, 2020)
4. Paper-style 95th-percentile attribution thresholding (Khan et al., 2024)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torchvision.transforms as T
from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from app.registry import load_model, get_model_entry, INFERENCE_TRANSFORM


def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Strips CLS token and reshapes 1D patch tokens into 2D spatial feature map."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def swin_reshape_transform(tensor: torch.Tensor, height: int = 7, width: int = 7) -> torch.Tensor:
    """Reshapes Swin stage feature tokens into 2D spatial feature map."""
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def get_gradcam_target_layers(model: nn.Module, model_family: str):
    """Identifies the final normalization layer for gradient-weighted CAM."""
    fam = model_family.lower()
    if fam in ["vit", "deit"]:
        return [model.blocks[-1].norm1]
    elif fam == "swin":
        return [model.layers[-1].blocks[-1].norm1]
    elif hasattr(model, "stages"):
        return [model.stages[-1].blocks[-1].norm]
    else:
        # Fallback to the last submodule before head
        return [list(model.children())[-2]]


def get_gradcam_reshape_tf(model_family: str):
    fam = model_family.lower()
    if fam in ["vit", "deit"]:
        return vit_reshape_transform
    elif fam == "swin":
        return swin_reshape_transform
    return None


def compute_attention_rollout(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Computes Attention Rollout across all transformer self-attention blocks.
    Returns 2D attention map of shape (224, 224) normalized to [0, 1].
    """
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

    num_tokens = attentions[0].size(-1)
    eye = torch.eye(num_tokens, device=input_tensor.device).unsqueeze(0)
    joint_attn = eye.clone()

    for attn_layer in attentions:
        head_avg = attn_layer.mean(dim=1)
        head_avg = 0.5 * head_avg + 0.5 * eye
        head_avg = head_avg / head_avg.sum(dim=-1, keepdim=True)
        joint_attn = torch.matmul(head_avg, joint_attn)

    cls_to_patches = joint_attn[0, 0, 1:]
    grid_size = int(np.sqrt(cls_to_patches.size(0)))
    rollout_map = cls_to_patches.reshape(1, 1, grid_size, grid_size)

    rollout_224 = nn.functional.interpolate(rollout_map, size=(224, 224), mode="bilinear", align_corners=False)
    rollout_arr = rollout_224.squeeze().cpu().numpy()

    rmin, rmax = rollout_arr.min(), rollout_arr.max()
    if rmax - rmin > 1e-8:
        return (rollout_arr - rmin) / (rmax - rmin)
    return np.zeros_like(rollout_arr)


def generate_attribution_map(
    model_name: str,
    pil_img: Image.Image,
    method: str = "Grad-CAM (pytorch-grad-cam)",
    target_class_idx: Optional[int] = None
) -> np.ndarray:
    """
    Computes a 2D normalized [0, 1] attribution map for the image.
    """
    model, entry = load_model(model_name)
    fam = entry.get("model_family", "vit").lower()

    rgb_img = pil_img.convert("RGB").resize((224, 224))
    input_tensor = INFERENCE_TRANSFORM(rgb_img).unsqueeze(0)

    # Determine target class
    if target_class_idx is None:
        with torch.inference_mode():
            logits = model(input_tensor)
            target_class_idx = int(torch.argmax(logits, dim=1).item())

    method_clean = method.lower()

    # Memory optimization: only enable gradients for final block and head for Grad-CAM
    cam_active_params = []
    if "grad-cam" in method_clean:
        if hasattr(model, "head") and model.head is not None:
            cam_active_params.extend(list(model.head.parameters()))
        if hasattr(model, "norm") and model.norm is not None:
            cam_active_params.extend(list(model.norm.parameters()))
        if fam in ["vit", "deit"] and hasattr(model, "blocks"):
            cam_active_params.extend(list(model.blocks[-1].parameters()))
        elif fam == "swin" and hasattr(model, "layers"):
            cam_active_params.extend(list(model.layers[-1].blocks[-1].parameters()))
        elif hasattr(model, "stages"):
            cam_active_params.extend(list(model.stages[-1].blocks[-1].parameters()))

    orig_grad_states = [(p, p.requires_grad) for p in cam_active_params]
    for p in cam_active_params:
        p.requires_grad = True

    try:
        if "grad-cam" in method_clean:
            target_layers = get_gradcam_target_layers(model, fam)
            reshape_tf = get_gradcam_reshape_tf(fam)
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_tf)
            targets = [ClassifierOutputTarget(target_class_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            attr_2d = grayscale_cam[0]

        elif "integrated gradients" in method_clean:
            # CPU-optimized with n_steps=30 and internal_batch_size=10
            ig = IntegratedGradients(model)
            baseline = torch.zeros_like(input_tensor)
            attr = ig.attribute(
                input_tensor,
                baselines=baseline,
                target=target_class_idx,
                n_steps=30,
                internal_batch_size=10
            )
            attr_2d = torch.sum(torch.abs(attr), dim=1).squeeze().detach().cpu().numpy()

        elif "attention rollout" in method_clean or "rollout" in method_clean:
            if fam in ["vit", "deit"]:
                attr_2d = compute_attention_rollout(model, input_tensor)
            else:
                target_layers = get_gradcam_target_layers(model, "swin")
                reshape_tf = get_gradcam_reshape_tf("swin")
                cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_tf)
                attr_2d = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class_idx)])[0]

        else:
            raise ValueError(f"Unknown explanation method: {method}")
    finally:
        for p, state in orig_grad_states:
            p.requires_grad = state
        import gc
        gc.collect()

    # Min-max normalize to [0, 1]
    amin, amax = attr_2d.min(), attr_2d.max()
    if amax - amin > 1e-8:
        attr_norm = (attr_2d - amin) / (amax - amin)
    else:
        attr_norm = np.zeros_like(attr_2d)

    return attr_norm


def render_attribution_overlay(
    pil_img: Image.Image,
    attr_map: np.ndarray,
    use_95th_percentile: bool = False
) -> Image.Image:
    """
    Renders an attribution map over the input image.
    If use_95th_percentile is True, applies Khan et al. (2024) thresholding where
    only the top 5% salient pixels are highlighted using the Turbo colormap.
    """
    orig_rgb = pil_img.convert("RGB").resize((224, 224))
    orig_np = np.array(orig_rgb).astype(np.float32)

    import matplotlib as mpl
    turbo = mpl.colormaps["turbo"]

    if use_95th_percentile:
        p95 = float(np.percentile(attr_map, 95))
        mask = (attr_map >= p95)
        # Rescale [p95, 1.0] -> [0.0, 1.0] for vivid colormap rendering
        range_val = 1.0 - p95
        scaled_salient = np.clip((attr_map - p95) / (range_val + 1e-8), 0.0, 1.0)
        heatmap_rgb = (np.array(turbo(scaled_salient)[:, :, :3]) * 255.0).astype(np.float32)

        # Alpha blend only top 5% pixels (75% heatmap, 25% satellite RGB)
        blended = orig_np.copy()
        alpha = 0.75
        for c in range(3):
            blended[:, :, c] = np.where(mask, (1.0 - alpha) * orig_np[:, :, c] + alpha * heatmap_rgb[:, :, c], orig_np[:, :, c])

        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    else:
        # Standard full-image heatmap overlay
        heatmap_rgb = (np.array(turbo(attr_map)[:, :, :3]) * 255.0).astype(np.float32)
        alpha = 0.55
        blended = (1.0 - alpha) * orig_np + alpha * heatmap_rgb
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
