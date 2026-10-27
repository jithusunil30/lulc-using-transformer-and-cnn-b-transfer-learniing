"""
app/main.py - Vision Transformers for LULC: Interactive Research Demo
====================================================================
Gradio Blocks application for:
- Tab 1: Classify Patch (predictions, calibrated confidence, XAI overlays, low-confidence alert)
- Tab 2: Land-Cover Map (scene tiling, color-coded classification map, confidence heatmap, tile drilldown)
- Tab 3: Domain-Shift Explorer (interactive Plotly charts, sensor vs geographic drop, misclassification gallery)
- Tab 4: Compare Models (side-by-side multi-model inference and attributions)
- Tab 5: About (research questions, methodology, empirical findings, and scientific limitations)
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import numpy as np

import gradio as gr
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from app.registry import (
    get_registered_models,
    get_model_entry,
    load_model,
    predict_patch,
    tile_and_predict_scene,
    EUROSAT_TO_COARSE,
    COARSE_CLASSES,
    EUROSAT_COLORS,
    COARSE_COLORS
)
from app.xai import generate_attribution_map, render_attribution_overlay
from app.charts import (
    plot_region_season_accuracy,
    plot_sensor_vs_geographic_drop,
    plot_calibration_ece_transfer
)

# Global in-memory cache for the most recently tiled scene to support tile click inspection
_CURRENT_SCENE_CACHE: Dict[str, Any] = {}

# Example patches and cached scenes
EXAMPLES_DIR = PROJECT_ROOT / "assets" / "examples"
SCENES_DIR = PROJECT_ROOT / "assets" / "scenes"
MISCLASS_DIR = PROJECT_ROOT / "assets" / "misclassifications"

EXAMPLE_IMAGES = sorted([str(p) for p in EXAMPLES_DIR.glob("*.jpg")]) if EXAMPLES_DIR.exists() else []
CACHED_SCENES = sorted([str(p) for p in SCENES_DIR.glob("*.png")]) if SCENES_DIR.exists() else []

MISCLASS_METADATA = []
if (MISCLASS_DIR / "metadata.json").exists():
    with open(MISCLASS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        MISCLASS_METADATA = json.load(f)


# ==============================================================================
# Helper Functions for Tab Interactions
# ==============================================================================

def create_top3_prob_chart(top_classes: List[Dict[str, Any]], temperature: float) -> go.Figure:
    """Builds horizontal bar chart for top-3 class probabilities."""
    classes = [item["class"] for item in reversed(top_classes)]
    probs = [item["prob"] * 100.0 for item in reversed(top_classes)]
    colors = ["#2563EB" if i == len(classes)-1 else "#94A3B8" for i in range(len(classes))]

    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation="h",
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        marker=dict(color=colors)
    ))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=40, t=25, b=25),
        height=180,
        xaxis=dict(range=[0, 115], title="Calibrated Probability (%)"),
        yaxis=dict(title=""),
        title=dict(text=f"Top Predictions (Temp T={temperature:.2f})", font=dict(size=13))
    )
    return fig


def classify_patch_fn(
    img: Optional[Image.Image],
    model_name: str,
    method_name: str,
    use_95th: bool
) -> Tuple[str, go.Figure, str, Optional[Image.Image]]:
    """Tab 1: Single patch prediction and explainability workflow."""
    if img is None:
        return (
            "<div style='color: #64748B;'>Please upload or select an example image.</div>",
            go.Figure(),
            "<div style='color: #64748B;'>No prediction yet.</div>",
            None
        )

    # 1. Calibrated prediction
    pred = predict_patch(model_name, img, apply_calibration=True)
    top_class = pred["top_class"]
    coarse_class = pred["coarse_class"]
    max_conf = pred["max_conf"]
    is_low_conf = pred["is_low_confidence"]
    temp = pred["temperature"]

    # 2. Warning banner
    if is_low_conf:
        warning_html = (
            f"<div style='background-color: #FEF3C7; border: 1.5px solid #F59E0B; color: #92400E; "
            f"padding: 12px 16px; border-radius: 8px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;'>"
            f"<span>⚠️</span> <span><strong>Low Confidence Warning:</strong> Maximum predicted probability is "
            f"<strong>{max_conf*100:.1f}%</strong> (threshold: 60.0%). The model exhibits uncertainty, suggesting "
            f"potential out-of-domain sensor calibration or agro-ecological distribution shift.</span></div>"
        )
    else:
        warning_html = (
            f"<div style='background-color: #ECFDF5; border: 1.5px solid #10B981; color: #065F46; "
            f"padding: 10px 14px; border-radius: 8px; font-weight: 500; margin-bottom: 12px;'>"
            f"✅ Confident Prediction: {max_conf*100:.1f}% probability (Calibrated with T={temp:.2f})</div>"
        )

    # 3. Coarse mapping card
    coarse_badge_color = {
        "Forest": "#15803D",
        "Crop": "#B45309",
        "Water": "#0369A1",
        "Built": "#B91C1C",
        "Grass/Shrub": "#15803D"
    }.get(coarse_class, "#334155")

    coarse_html = (
        f"<div style='background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-top: 8px;'>"
        f"<div style='font-size: 0.85em; text-transform: uppercase; color: #64748B; font-weight: 600;'>Harmonized Coarse Land-Cover Class</div>"
        f"<div style='font-size: 1.35em; font-weight: 700; color: {coarse_badge_color}; margin: 4px 0;'>"
        f"{coarse_class}</div>"
        f"<div style='font-size: 0.9em; color: #475569;'>Mapped from EuroSAT fine-grained class: <strong>{top_class}</strong></div>"
        f"</div>"
    )

    # 4. Probabilities chart
    chart_fig = create_top3_prob_chart(pred["top_classes"], temp)

    # 5. Explainability attribution overlay
    entry = get_model_entry(model_name)
    attr_map = generate_attribution_map(model_name, img, method=method_name)
    overlay_img = render_attribution_overlay(img, attr_map, use_95th_percentile=use_95th)

    return warning_html, chart_fig, coarse_html, overlay_img


def run_scene_mapping(
    scene_img: Optional[Image.Image],
    model_name: str,
    method_name: str
) -> Tuple[Optional[Image.Image], Optional[Image.Image], List[Dict[str, Any]], str]:
    """Tab 2: Tiling large RGB scenes and generating spatial classification maps."""
    if scene_img is None:
        return None, None, [], "<div style='color: #64748B;'>Please upload a scene or select a cached scene.</div>"

    res = tile_and_predict_scene(model_name, scene_img, tile_size=64, stride=64, apply_calibration=True)
    _CURRENT_SCENE_CACHE["last_result"] = res
    _CURRENT_SCENE_CACHE["model_name"] = model_name
    _CURRENT_SCENE_CACHE["method_name"] = method_name

    legend_items = "".join([
        f"<div style='display: flex; align-items: center; gap: 6px; margin: 2px 8px;'>"
        f"<span style='width: 14px; height: 14px; background-color: rgb{color}; border-radius: 3px; display: inline-block;'></span>"
        f"<span style='font-size: 0.85em; font-weight: 500;'>{cls}</span></div>"
        for cls, color in EUROSAT_COLORS.items()
    ])
    legend_html = (
        f"<div style='background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px; margin-top: 10px;'>"
        f"<div style='font-size: 0.8em; font-weight: 700; color: #475569; margin-bottom: 6px; text-transform: uppercase;'>EuroSAT 10-Class Palette:</div>"
        f"<div style='display: flex; flex-wrap: wrap;'>{legend_items}</div>"
        f"</div>"
    )

    return res["blended_map"], res["confidence_heatmap"], res["area_percentages"], legend_html


def inspect_tile_click(evt: gr.SelectData) -> Tuple[Optional[Image.Image], str, Optional[Image.Image]]:
    """Tab 2: Interactive tile inspection upon clicking on the classification map."""
    res = _CURRENT_SCENE_CACHE.get("last_result")
    if not res:
        return None, "No scene loaded. Click 'Generate Land-Cover Map' first.", None

    stride = res["stride"]
    n_cols = res["n_cols"]
    n_rows = res["n_rows"]

    # Extract clicked pixel coordinates
    if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 2:
        x, y = evt.index[0], evt.index[1]
    else:
        return None, "Invalid click selection.", None

    col = min(max(0, int(x // stride)), n_cols - 1)
    row = min(max(0, int(y // stride)), n_rows - 1)

    tile_data = res["tile_data"]
    target_tile = None
    for t in tile_data:
        if t["row"] == row and t["col"] == col:
            target_tile = t
            break

    if target_tile is None:
        return None, f"Tile ({row}, {col}) not found.", None

    tile_crop = target_tile["crop"]
    cls_name = target_tile["class_name"]
    coarse_name = target_tile["coarse_class"]
    conf = target_tile["confidence"]

    tile_info_html = (
        f"<div style='background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px;'>"
        f"<div style='font-size: 0.85em; color: #64748B;'>Selected Tile: <strong>Row {row}, Col {col}</strong> (x={x}, y={y})</div>"
        f"<div style='font-size: 1.2em; font-weight: 700; color: #1E293B; margin: 4px 0;'>Prediction: {cls_name}</div>"
        f"<div>Coarse Group: <strong>{coarse_name}</strong></div>"
        f"<div>Confidence: <strong>{conf*100:.1f}%</strong></div>"
        f"</div>"
    )

    # Generate on-demand explanation for selected tile
    model_name = _CURRENT_SCENE_CACHE.get("model_name", get_registered_models()[1])
    method_name = _CURRENT_SCENE_CACHE.get("method_name", "Grad-CAM (pytorch-grad-cam)")
    attr_map = generate_attribution_map(model_name, tile_crop, method=method_name)
    overlay = render_attribution_overlay(tile_crop, attr_map, use_95th_percentile=False)

    return tile_crop, tile_info_html, overlay


def compare_all_models_fn(
    img: Optional[Image.Image],
    method_name: str
) -> List[Tuple[Optional[Image.Image], str]]:
    """Tab 4: Multi-model side-by-side inference and attribution."""
    if img is None:
        return []

    results = []
    registered = get_registered_models()

    for m_name in registered:
        pred = predict_patch(m_name, img, apply_calibration=True)
        attr_map = generate_attribution_map(m_name, img, method=method_name)
        overlay = render_attribution_overlay(img, attr_map, use_95th_percentile=False)

        info = (
            f"**{m_name}**\n\n"
            f"- **Pred**: {pred['top_class']} ({pred['coarse_class']})\n"
            f"- **Confidence**: {pred['max_conf']*100:.1f}%\n"
            f"- **Trainable Params**: {pred.get('trainable_pct', 24.6):.1f}%\n"
            f"- **Calibrated T**: {pred['temperature']:.2f}"
        )
        results.append((overlay, info))

    return results


# ==============================================================================
# Gradio Blocks Application Layout
# ==============================================================================

MODELS_LIST = get_registered_models()
DEFAULT_MODEL = "ViT-Small (Unfreeze Last 3)" if "ViT-Small (Unfreeze Last 3)" in MODELS_LIST else MODELS_LIST[0]
METHODS_LIST = [
    "Grad-CAM (pytorch-grad-cam)",
    "Integrated Gradients (Captum, n=30)",
    "Attention Rollout (ViT/DeiT)"
]

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)

with gr.Blocks(title="Vision Transformers for LULC — Research Suite") as demo:

    gr.Markdown(
        """
        # 🛰️ Efficiency-Accuracy Trade-offs in Vision Transformers for LULC
        ### Comprehensive Empirical Re-evaluation & Explainability Suite for Satellite Remote Sensing
        *Benchmarking Vision Transformers (ViT, DeiT, Swin) and ConvNeXt across 5 fine-tuning strategies under distribution shift.*
        """
    )

    with gr.Tabs():

        # ======================================================================
        # TAB 1: CLASSIFY PATCH
        # ======================================================================
        with gr.TabItem("🔍 Classify Patch", id="tab_classify"):
            gr.Markdown("### Single Satellite Patch Classification & Faithful Explainability")
            gr.Markdown("Upload a Sentinel-2 RGB patch or select an example from the curated test set below.")

            with gr.Row():
                with gr.Column(scale=4):
                    input_patch = gr.Image(type="pil", label="Input Satellite Patch (64x64 RGB)", height=260)

                    if EXAMPLE_IMAGES:
                        gr.Examples(
                            examples=EXAMPLE_IMAGES[:10],
                            inputs=input_patch,
                            label="Curated EuroSAT Test Examples (1 per class)"
                        )

                    with gr.Row():
                        patch_model_dropdown = gr.Dropdown(
                            choices=MODELS_LIST,
                            value=DEFAULT_MODEL,
                            label="Target Model & Strategy"
                        )
                        patch_method_dropdown = gr.Dropdown(
                            choices=METHODS_LIST,
                            value="Grad-CAM (pytorch-grad-cam)",
                            label="Explanation Method"
                        )

                    use_95th_chk = gr.Checkbox(
                        label="Paper-style 95th-Percentile Salience Overlay (Khan et al. 2024)",
                        value=False,
                        info="When checked, masks out sub-95th percentile attributions to reproduce the paper's visualization style."
                    )
                    classify_btn = gr.Button("🚀 Run Inference & Explain", variant="primary", size="lg")

                with gr.Column(scale=5):
                    warning_box = gr.HTML(visible=True)
                    coarse_box = gr.HTML(visible=True)
                    prob_chart = gr.Plot(label="Top Class Probabilities")
                    overlay_display = gr.Image(type="pil", label="Attribution Overlay (Spatial Grounding)", height=280)

            classify_btn.click(
                fn=classify_patch_fn,
                inputs=[input_patch, patch_model_dropdown, patch_method_dropdown, use_95th_chk],
                outputs=[warning_box, prob_chart, coarse_box, overlay_display]
            )

        # ======================================================================
        # TAB 2: LAND-COVER MAP
        # ======================================================================
        with gr.TabItem("🗺️ Land-Cover Map", id="tab_map"):
            gr.Markdown("### Automated High-Resolution Scene Tiling & Spatial LULC Mapping")
            gr.Markdown("Upload an arbitrary large Sentinel-2 RGB scene or select a pre-cached composite scene to generate a full land-cover classification and confidence heatmap.")

            with gr.Row():
                with gr.Column(scale=4):
                    scene_input = gr.Image(type="pil", label="Upload Large RGB Scene (e.g. 256x256 or 512x512)", height=260)

                    if CACHED_SCENES:
                        gr.Examples(
                            examples=CACHED_SCENES,
                            inputs=scene_input,
                            label="Pre-cached Regional Satellite Scenes (256x256)"
                        )

                    scene_model_dropdown = gr.Dropdown(
                        choices=MODELS_LIST,
                        value=DEFAULT_MODEL,
                        label="Inference Backbone"
                    )
                    scene_method_dropdown = gr.Dropdown(
                        choices=METHODS_LIST,
                        value="Grad-CAM (pytorch-grad-cam)",
                        label="Explanation Method (for Tile Drilldown)"
                    )
                    map_btn = gr.Button("🧭 Generate Land-Cover Map", variant="primary", size="lg")
                    legend_display = gr.HTML()

                with gr.Column(scale=6):
                    with gr.Row():
                        classified_map_display = gr.Image(
                            type="pil",
                            label="Color-Coded Classification Map (Click any tile to inspect!)",
                            interactive=True
                        )
                        conf_map_display = gr.Image(
                            type="pil",
                            label="Calibrated Confidence Heatmap (Plasma Colormap)"
                        )

                    area_table = gr.Dataframe(
                        headers=["Class", "Coarse Group", "Tiles", "Area (%)"],
                        label="LULC Class Area Coverage (%)"
                    )

            gr.Markdown("#### 🔎 Interactive Tile Drilldown (Click above on the Classification Map)")
            with gr.Row():
                with gr.Column(scale=1):
                    selected_tile_crop = gr.Image(type="pil", label="Selected 64x64 Tile", height=160)
                with gr.Column(scale=2):
                    selected_tile_info = gr.HTML()
                with gr.Column(scale=2):
                    selected_tile_overlay = gr.Image(type="pil", label="Tile Attribution Map", height=160)

            map_btn.click(
                fn=run_scene_mapping,
                inputs=[scene_input, scene_model_dropdown, scene_method_dropdown],
                outputs=[classified_map_display, conf_map_display, area_table, legend_display]
            )

            classified_map_display.select(
                fn=inspect_tile_click,
                inputs=None,
                outputs=[selected_tile_crop, selected_tile_info, selected_tile_overlay]
            )

        # ======================================================================
        # TAB 3: DOMAIN-SHIFT EXPLORER
        # ======================================================================
        with gr.TabItem("🌐 Domain-Shift Explorer", id="tab_shift"):
            gr.Markdown("### Quantitative Evaluation Across Geographic Regions & Distribution Shift")
            gr.Markdown(
                "Exploring performance degradation across Google Earth Engine Sentinel-2 tiles collected from "
                "7 distinct agro-ecological regions (Europe Control vs. India Monsoon/Dry seasons)."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    shift_fam_dropdown = gr.Dropdown(
                        choices=["ViT", "DeiT", "Swin"],
                        value="ViT",
                        label="Model Family"
                    )
                    shift_strat_dropdown = gr.Dropdown(
                        choices=["full_ft", "unfreeze_last_3", "unfreeze_last_2", "linear_probe", "scratch"],
                        value="unfreeze_last_3",
                        label="Fine-Tuning Strategy"
                    )
                    refresh_chart_btn = gr.Button("🔄 Update Breakdown Chart")

            region_chart = gr.Plot(value=plot_region_season_accuracy("ViT", "unfreeze_last_3"))

            refresh_chart_btn.click(
                fn=plot_region_season_accuracy,
                inputs=[shift_fam_dropdown, shift_strat_dropdown],
                outputs=region_chart
            )

            gr.Markdown("---")
            gr.Markdown("### Core Scientific Finding: Sensor/Label Format Shift vs. Geographic Shift")
            sensor_drop_chart = gr.Plot(value=plot_sensor_vs_geographic_drop())

            gr.Markdown("---")
            gr.Markdown("### Calibration Failure Under Shift (In-Domain vs. Shifted ECE)")
            ece_chart = gr.Plot(value=plot_calibration_ece_transfer())

            if MISCLASS_METADATA:
                gr.Markdown("---")
                gr.Markdown("### Gallery of High-Confidence Errors Under Geographic Shift")
                gr.Markdown("Representative patches where models failed under domain shift due to sensor contrast and label semantics.")

                gallery_items = [
                    (
                        item["image_path"],
                        f"True: {item['true_coarse']} | Pred: {item['pred_coarse']} ({item['confidence']}% conf) [{item['region']} {item['season']}]"
                    )
                    for item in MISCLASS_METADATA[:12]
                ]
                gr.Gallery(
                    value=gallery_items,
                    columns=4,
                    height="auto",
                    label="Shift Errors Gallery"
                )

        # ======================================================================
        # TAB 4: COMPARE MODELS
        # ======================================================================
        with gr.TabItem("⚖️ Compare Models", id="tab_compare"):
            gr.Markdown("### Multi-Model Head-to-Head Comparison")
            gr.Markdown("Compare predictions, calibrated confidences, and explanation heatmaps across all fine-tuning strategies on the exact same patch.")

            with gr.Row():
                compare_img_input = gr.Image(type="pil", label="Input Patch to Compare", height=240)
                with gr.Column():
                    compare_method_dropdown = gr.Dropdown(
                        choices=METHODS_LIST,
                        value="Grad-CAM (pytorch-grad-cam)",
                        label="Explanation Method"
                    )
                    compare_btn = gr.Button("⚡ Compare All Registered Models", variant="primary", size="lg")

            if EXAMPLE_IMAGES:
                gr.Examples(
                    examples=EXAMPLE_IMAGES[:6],
                    inputs=compare_img_input,
                    label="Quick Test Examples"
                )

            gr.Markdown("#### Registered Backbones & Strategies")

            # Dynamically display top 4 models side-by-side
            top_4_models = MODELS_LIST[:4]
            with gr.Row():
                cards = []
                for idx, m_name in enumerate(top_4_models):
                    with gr.Column():
                        card_img = gr.Image(type="pil", label=m_name, height=220)
                        card_txt = gr.Markdown(f"**{m_name}**\n\n*Click Compare to run*")
                        cards.append((card_img, card_txt))

            def run_comparison_top4(img: Optional[Image.Image], method: str):
                if img is None:
                    return [None, "No image"] * len(top_4_models)
                outs = []
                for m_name in top_4_models:
                    pred = predict_patch(m_name, img, apply_calibration=True)
                    attr_map = generate_attribution_map(m_name, img, method=method)
                    overlay = render_attribution_overlay(img, attr_map, use_95th_percentile=False)
                    info = (
                        f"**{m_name}**\n\n"
                        f"- **Prediction**: `{pred['top_class']}` ({pred['coarse_class']})\n"
                        f"- **Calibrated Conf**: `{pred['max_conf']*100:.1f}%`\n"
                        f"- **Strategy**: `{pred.get('strategy', 'fine-tuned')}`\n"
                        f"- **Temperature T**: `{pred['temperature']:.2f}`"
                    )
                    outs.extend([overlay, info])
                return outs

            flat_outputs = [item for pair in cards for item in pair]
            compare_btn.click(
                fn=run_comparison_top4,
                inputs=[compare_img_input, compare_method_dropdown],
                outputs=flat_outputs
            )

        # ======================================================================
        # TAB 5: ABOUT & METHODOLOGY
        # ======================================================================
        with gr.TabItem("📖 About & Findings", id="tab_about"):
            gr.Markdown(
                """
                ## Project Overview & Research Questions
                This interactive suite is part of the research project:
                **"Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers for LULC, with Quantitative Evaluation of Explanations"**,
                re-evaluating and extending the findings of *Khan et al. (Scientific Reports 2024)*.

                ### Core Research Questions & Empirical Answers:

                1. **Is full fine-tuning necessary for Vision Transformers in remote sensing?**
                   - **Finding**: **No.** Unfreezing only the final 3 transformer blocks (`unfreeze_last_3`) achieves **97.48% - 98.62%** accuracy, matching complete fine-tuning within 1.0% across ViT, DeiT, and Swin while reducing trainable parameters by **42% to 75%** and training memory by up to 72%.

                2. **Does transfer learning reduce computational cost per epoch?**
                   - **Finding**: **No.** As shown by our paired t-test ($t = 0.42, p = 0.681$), per-epoch FLOPs and latency are identical between scratch and pretrained models. Pretraining reduces *sample complexity* (epochs to converge), but computational savings require architectural freezing.

                3. **Can Grad-CAM be applied to Vision Transformers?**
                   - **Finding**: **Yes.** The paper's claim that Grad-CAM cannot be applied to ViTs is refuted. Utilizing 2D spatial reshape transforms restoring the $14 \times 14$ grid yields highly faithful explanations (Deletion AUC $0.28 - 0.34$, $p < 0.0001$).

                4. **Are qualitative 95th-percentile attention maps valid?**
                   - **Finding**: **No.** Applying the Adebayo parameter-randomization sanity check demonstrates that blurred-baseline Integrated Gradients acts as an unlearned edge detector ($r = 0.468, p < 0.0001$), failing model sensitivity.

                5. **Does in-domain calibration transfer under geographic shift?**
                   - **Finding**: **No.** Temperature scaling fitted on validation sets reduces in-domain ECE to $<0.005$, but fails out-of-domain (ECE remains $0.40 - 0.70$). Selective prediction with confidence thresholds is required.

                ---

                ### Scientific Limitations & Dataset Nuances:
                - **Dynamic World Label Noise**: Dynamic World labels used for out-of-domain evaluation are generated via an automated deep learning model with estimated 10-15% pixel boundary noise.
                - **RGB Spectral Restriction**: While Sentinel-2 provides 13 multispectral bands (including Red Edge, NIR, and SWIR), models were evaluated strictly on optical RGB (B4, B3, B2) to match standard computer vision backbones.
                - **Coarse-Class Harmonization**: Harmonizing EuroSAT's 10 classes to Dynamic World's 5 coarse classes introduces semantic abstraction (e.g. Highways and Residential mapped to Built).
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=custom_theme, share=False)
