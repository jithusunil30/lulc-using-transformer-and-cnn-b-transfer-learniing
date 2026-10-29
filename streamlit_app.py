"""
streamlit_app.py - Vision Transformers for LULC: Interactive Research Dashboard
==============================================================================
A modern, comprehensive Streamlit web application providing:
1. 🔍 Classify Patch: Single patch inference, calibrated confidence, coarse-class mapping,
   XAI heatmaps (Grad-CAM, IG n=30, Attention Rollout) with the 95th-percentile salience toggle,
   and dynamic low-confidence warnings (<0.60).
2. 🗺️ Land-Cover Map: High-resolution satellite scene tiling (64x64, stride 64), color-coded
   classification map, Plasma confidence heatmap, class coverage area statistics, and interactive
   tile inspection with on-demand explanation.
3. 🌐 Domain-Shift Explorer: Interactive Plotly analytics across 7 geographic regions and seasons,
   sensor/label shift vs. geographic shift breakdown, calibration transfer (ECE), and a gallery
   of real misclassified patches under shift.
4. ⚖️ Compare Models: Multi-model head-to-head comparison on the exact same patch with side-by-side
   predictions, calibrated confidences, and attribution overlays.
5. 📊 Claims & Rigor Benchmark: Interactive tables from claims_table.csv, modern CNN baseline
   (ConvNeXt-Tiny) benchmarking, LoRA parameter-budget efficiency, and hyperparameter sensitivity surfaces.
6. 📖 About & Methodology: Research questions, empirical answers, and scientific limitations.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent
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

# Page configuration
st.set_page_config(
    page_title="Vision Transformers for LULC",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .coarse-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 4px 0;
    }
    .warn-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        color: #92400E;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        color: #065F46;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Data paths
EXAMPLES_DIR = PROJECT_ROOT / "assets" / "examples"
SCENES_DIR = PROJECT_ROOT / "assets" / "scenes"
MISCLASS_DIR = PROJECT_ROOT / "assets" / "misclassifications"
RESULTS_DIR = PROJECT_ROOT / "results"

@st.cache_data
def load_misclass_metadata():
    meta_path = MISCLASS_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.cache_data
def load_csv_data(filename: str):
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None

EXAMPLE_IMAGES = sorted(list(EXAMPLES_DIR.glob("*.jpg"))) if EXAMPLES_DIR.exists() else []
CACHED_SCENES = sorted(list(SCENES_DIR.glob("*.png"))) if SCENES_DIR.exists() else []
MISCLASS_ITEMS = load_misclass_metadata()
MODELS_LIST = get_registered_models()
DEFAULT_MODEL = "ViT-Small (Unfreeze Last 3)" if "ViT-Small (Unfreeze Last 3)" in MODELS_LIST else MODELS_LIST[0]
METHODS_LIST = [
    "Grad-CAM (pytorch-grad-cam)",
    "Integrated Gradients (Captum, n=30)",
    "Attention Rollout (ViT/DeiT)"
]

# Sidebar navigation
st.sidebar.markdown("# 🛰️ LULC Research Suite")
st.sidebar.markdown("**Vision Transformers & Modern CNN Benchmarking**")
st.sidebar.caption("Re-evaluating Khan et al. (Scientific Reports 2024)")

nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Classify Patch",
        "🗺️ Land-Cover Map",
        "🌐 Domain-Shift Explorer",
        "⚖️ Model Comparison",
        "📊 Paper Claims & Rigor Benchmark",
        "📖 About & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Global Model Settings")
sidebar_model = st.sidebar.selectbox("Default Model Backbone", MODELS_LIST, index=MODELS_LIST.index(DEFAULT_MODEL))
sidebar_method = st.sidebar.selectbox("Default Explanation Method", METHODS_LIST, index=0)

model_meta = get_model_entry(sidebar_model)
st.sidebar.caption(f"**Strategy**: `{model_meta['strategy']}` | **Params**: `{model_meta['trainable_pct']}%` | **Optimal T**: `{model_meta['temperature']:.2f}`")


# ==============================================================================
# SECTION 1: CLASSIFY PATCH
# ==============================================================================
if nav_selection == "🔍 Classify Patch":
    st.markdown('<div class="main-header">🔍 Single Patch Classification & Faithful Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify Sentinel-2 RGB patches (64x64) with temperature calibration and evaluate spatial attribution faithfulness.</div>', unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("1. Select or Upload Image")
        input_source = st.radio("Choose input method:", ["Curated EuroSAT Test Examples", "Upload Custom RGB Image"], horizontal=True)

        input_img = None
        if input_source == "Curated EuroSAT Test Examples":
            if EXAMPLE_IMAGES:
                example_names = [p.stem.replace("_example", "") for p in EXAMPLE_IMAGES]
                selected_idx = st.selectbox("Select EuroSAT Class Example:", range(len(EXAMPLE_IMAGES)), format_func=lambda i: example_names[i])
                input_img = Image.open(EXAMPLE_IMAGES[selected_idx]).convert("RGB")
                st.image(input_img, caption=f"Selected: {example_names[selected_idx]} (64x64 RGB)", width=240)
        else:
            uploaded_file = st.file_uploader("Upload Sentinel-2 RGB Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                input_img = Image.open(uploaded_file).convert("RGB")
                st.image(input_img, caption="Uploaded Image", width=240)

        st.subheader("2. Inference & Attribution Settings")
        c1, c2 = st.columns(2)
        with c1:
            active_model = st.selectbox("Model & Strategy:", MODELS_LIST, index=MODELS_LIST.index(sidebar_model), key="patch_m")
        with c2:
            active_method = st.selectbox("XAI Method:", METHODS_LIST, index=METHODS_LIST.index(sidebar_method), key="patch_x")

        use_95th = st.checkbox(
            "Paper-Style 95th-Percentile Salience Overlay (Khan et al. 2024)",
            value=False,
            help="When checked, masks out sub-95th percentile attributions to reproduce the paper's visualization style."
        )

        run_classify = st.button("🚀 Run Inference & Explain", type="primary", use_container_width=True)

    with col_output:
        st.subheader("3. Model Prediction & Attribution")
        if input_img is not None:
            if run_classify:
                with st.spinner("Executing calibrated inference and computing attribution gradients..."):
                    pred = predict_patch(active_model, input_img, apply_calibration=True)
                    attr_map = generate_attribution_map(active_model, input_img, method=active_method)
                    overlay_img = render_attribution_overlay(input_img, attr_map, use_95th_percentile=use_95th)

                    st.session_state["patch_pred_cache"] = pred
                    st.session_state["patch_overlay_cache"] = overlay_img
                    st.session_state["patch_attr_cache"] = attr_map

            pred = st.session_state.get("patch_pred_cache")
            overlay_img = st.session_state.get("patch_overlay_cache")

            if pred is None:
                st.info("👆 Select a patch or upload an image, then click **'🚀 Run Inference & Explain'**.")
            else:
                # Warning banner or success banner
                if pred["is_low_confidence"]:
                    st.markdown(
                        f"""
                        <div class="warn-box">
                            ⚠️ <strong>Low Confidence Alert:</strong> Maximum predicted probability is 
                            <strong>{pred['max_conf']*100:.1f}%</strong> (threshold: 60.0%). 
                            The model exhibits high uncertainty, likely encountering out-of-domain distribution shift.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="success-box">
                            ✅ <strong>High Confidence Prediction:</strong> {pred['max_conf']*100:.1f}% probability (Calibrated with T={pred['temperature']:.2f})
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Coarse Class Card
                coarse_color = {
                    "Forest": "#15803D",
                    "Crop": "#B45309",
                    "Water": "#0369A1",
                    "Built": "#B91C1C",
                    "Grass/Shrub": "#15803D"
                }.get(pred["coarse_class"], "#334155")

                st.markdown(
                    f"""
                    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 14px;">
                        <span style="font-size: 0.85rem; text-transform: uppercase; color: #64748B; font-weight: 600;">Harmonized Coarse Land-Cover Class</span><br>
                        <span style="font-size: 1.4rem; font-weight: 800; color: {coarse_color};">{pred['coarse_class']}</span><br>
                        <span style="font-size: 0.95rem; color: #475569;">EuroSAT 10-Class Prediction: <strong>{pred['top_class']}</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Top-3 Probabilities horizontal bar chart
                top_items = pred["top_classes"]
                classes = [item["class"] for item in reversed(top_items)]
                probs = [item["prob"] * 100 for item in reversed(top_items)]
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
                    title=dict(text=f"Top Predictions (T={pred['temperature']:.2f})", font=dict(size=13))
                )
                st.plotly_chart(fig, use_container_width=True)

                # Attribution Overlay
                st.image(overlay_img, caption=f"Explanation: {active_method} ({'95th-Percentile Salient' if use_95th else 'Continuous Heatmap'})", width=360)


# ==============================================================================
# SECTION 2: LAND-COVER MAP
# ==============================================================================
elif nav_selection == "🗺️ Land-Cover Map":
    st.markdown('<div class="main-header">🗺️ Automated High-Resolution Scene Tiling & Spatial LULC Mapping</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tiling large RGB scenes into 64x64 patches with stride 64, executing batched CPU inference, and generating spatial classification and confidence maps.</div>', unsafe_allow_html=True)

    col_scene_in, col_scene_out = st.columns([1, 1.4], gap="large")

    with col_scene_in:
        st.subheader("1. Choose or Upload Satellite Scene")
        scene_mode = st.radio("Scene Source:", ["Pre-cached Regional Scenes (256x256)", "Upload Custom Large Scene"], horizontal=True)

        scene_img = None
        if scene_mode == "Pre-cached Regional Scenes (256x256)":
            if CACHED_SCENES:
                scene_names = [p.stem.replace("scene_", "").replace("_", " ").title() for p in CACHED_SCENES]
                chosen_scene_idx = st.selectbox("Select Cached Scene:", range(len(CACHED_SCENES)), format_func=lambda i: scene_names[i])
                scene_img = Image.open(CACHED_SCENES[chosen_scene_idx]).convert("RGB")
                st.image(scene_img, caption=f"Scene: {scene_names[chosen_scene_idx]} (256x256)", width=280)
        else:
            scene_file = st.file_uploader("Upload Large Scene (PNG/JPG)", type=["png", "jpg", "jpeg"], key="large_scene_upload")
            if scene_file is not None:
                scene_img = Image.open(scene_file).convert("RGB")
                st.image(scene_img, caption="Custom Scene", width=280)

        st.subheader("2. Model & Tiling Controls")
        map_model = st.selectbox("Backbone Model:", MODELS_LIST, index=MODELS_LIST.index(sidebar_model), key="map_m")
        map_stride = st.slider("Tiling Stride (pixels):", min_value=32, max_value=64, value=64, step=32, help="Patch size is fixed at 64x64.")

        run_map_btn = st.button("🧭 Generate Land-Cover Map", type="primary", use_container_width=True)

        st.markdown("#### 🎨 EuroSAT 10-Class Legend")
        legend_cols = st.columns(2)
        for i, (c_name, color) in enumerate(EUROSAT_COLORS.items()):
            col_target = legend_cols[i % 2]
            col_target.markdown(
                f"<div style='display: flex; align-items: center; gap: 8px; margin: 2px 0;'>"
                f"<span style='width: 14px; height: 14px; background-color: rgb{color}; border-radius: 3px; display: inline-block;'></span>"
                f"<span style='font-size: 0.85rem;'>{c_name}</span></div>",
                unsafe_allow_html=True
            )

    with col_scene_out:
        st.subheader("3. Classification Map & Confidence Heatmap")
        if scene_img is not None:
            if run_map_btn:
                with st.spinner("Tiling scene, running batched CPU inference, and synthesizing maps..."):
                    res = tile_and_predict_scene(map_model, scene_img, tile_size=64, stride=map_stride, apply_calibration=True)
                    st.session_state["scene_res_cache"] = res
                    st.session_state["scene_img_cache"] = scene_img

            res = st.session_state.get("scene_res_cache")

            if res is None:
                st.info("👈 Select a scene, then click **'🧭 Generate Land-Cover Map'**.")
            else:
                m_tab1, m_tab2 = st.tabs(["Color-Coded Classification Map", "Calibrated Confidence Heatmap (Plasma)"])
                with m_tab1:
                    st.image(res["blended_map"], caption=f"Spatial LULC Map ({res['total_tiles']} Tiles: {res['n_rows']}x{res['n_cols']})", use_container_width=True)
                with m_tab2:
                    st.image(res["confidence_heatmap"], caption="Confidence Heatmap (Plasma: 0.0 to 1.0)", use_container_width=True)

                st.markdown("#### 📊 Land-Cover Class Coverage Area (%)")
                df_area = pd.DataFrame(res["area_percentages"])
                st.dataframe(df_area, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("#### 🔎 Interactive Tile Drilldown")
                st.caption("Inspect any individual 64x64 patch from the tiled scene.")
                selected_tile_idx = st.slider("Select Tile Index to Inspect:", min_value=0, max_value=res["total_tiles"]-1, value=0)

                target_tile = res["tile_data"][selected_tile_idx]
                tile_c1, tile_c2, tile_c3 = st.columns([1, 1.5, 1.2])

                with tile_c1:
                    st.image(target_tile["crop"], caption=f"Tile #{selected_tile_idx} (Row {target_tile['row']}, Col {target_tile['col']})", width=140)
                with tile_c2:
                    st.markdown(
                        f"""
                        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px;">
                            <span style="font-size: 0.85rem; color: #64748B;">Prediction</span><br>
                            <span style="font-size: 1.25rem; font-weight: 700; color: #1E293B;">{target_tile['class_name']}</span><br>
                            <span>Coarse: <strong>{target_tile['coarse_class']}</strong></span><br>
                            <span>Confidence: <strong>{target_tile['confidence']*100:.1f}%</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with tile_c3:
                    tile_attr = generate_attribution_map(map_model, target_tile["crop"], method="Grad-CAM (pytorch-grad-cam)")
                    tile_overlay = render_attribution_overlay(target_tile["crop"], tile_attr, use_95th_percentile=False)
                    st.image(tile_overlay, caption="Tile Grad-CAM", width=140)


# ==============================================================================
# SECTION 3: DOMAIN-SHIFT EXPLORER
# ==============================================================================
elif nav_selection == "🌐 Domain-Shift Explorer":
    st.markdown('<div class="main-header">🌐 Geographic Distribution Shift Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Vision Transformers and CNN baselines on Google Earth Engine Sentinel-2 imagery across 7 agro-ecological regions (Europe vs. India).</div>', unsafe_allow_html=True)

    tab_geo1, tab_geo2, tab_geo3, tab_geo4 = st.tabs([
        "📊 Accuracy by Region & Season",
        "📉 Sensor vs. Geographic Shift",
        "🎯 Calibration Transfer (ECE)",
        "🖼️ Shift Errors Gallery"
    ])

    with tab_geo1:
        st.subheader("Accuracy Across 7 Geographic Regions & Seasons")
        c1, c2 = st.columns(2)
        with c1:
            sfam = st.selectbox("Model Family:", ["ViT", "DeiT", "Swin"], index=0)
        with c2:
            sstrat = st.selectbox("Fine-Tuning Strategy:", ["unfreeze_last_3", "full_ft", "unfreeze_last_2", "linear_probe", "scratch"], index=0)

        fig_reg = plot_region_season_accuracy(sfam, sstrat)
        st.plotly_chart(fig_reg, use_container_width=True)

    with tab_geo2:
        st.subheader("Core Scientific Finding: Sensor/Label Format Shift vs. Pure Geographic Shift")
        st.info("Transfer degradation is overwhelmingly driven by sensor calibration and label annotation differences between EuroSAT and Dynamic World (39%-79% drop), whereas pure geographic shift between Europe and India is statistically negligible (p > 0.50).")
        fig_drop = plot_sensor_vs_geographic_drop()
        st.plotly_chart(fig_drop, use_container_width=True)

    with tab_geo3:
        st.subheader("Model Calibration Transfer Failure Under Distribution Shift")
        st.warning("In-domain temperature scaling fitted on validation data lowers in-domain ECE to <0.005, but completely fails under distribution shift (ECE remains 0.40 - 0.70). Selective prediction with confidence thresholds is required.")
        fig_ece = plot_calibration_ece_transfer()
        st.plotly_chart(fig_ece, use_container_width=True)

    with tab_geo4:
        st.subheader("Curated Gallery of High-Confidence Shift Errors")
        st.caption("Representative patches where fully fine-tuned Vision Transformers failed under distribution shift.")

        if MISCLASS_ITEMS:
            cols = st.columns(4)
            for idx, item in enumerate(MISCLASS_ITEMS[:12]):
                col = cols[idx % 4]
                img_p = Path(item["image_path"])
                if not img_p.exists():
                    img_p = PROJECT_ROOT / item["image_path"]

                if img_p.exists():
                    err_img = Image.open(img_p).convert("RGB")
                    col.image(
                        err_img,
                        caption=f"True: {item['true_coarse']}\nPred: {item['pred_coarse']} ({item['confidence']}% conf)\n[{item['region']} {item['season']}]",
                        use_container_width=True
                    )


# ==============================================================================
# SECTION 4: MODEL COMPARISON
# ==============================================================================
elif nav_selection == "⚖️ Model Comparison":
    st.markdown('<div class="main-header">⚖️ Multi-Model Head-to-Head Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate predictions, calibrated confidences, and explanation heatmaps across all fine-tuning strategies on the exact same patch.</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 3], gap="large")

    with c_left:
        st.subheader("Select Comparison Patch")
        cmp_source = st.radio("Patch Source:", ["Curated Example", "Upload Image"], horizontal=True, key="cmp_src")

        cmp_img = None
        if cmp_source == "Curated Example" and EXAMPLE_IMAGES:
            cmp_choice = st.selectbox("Class:", range(len(EXAMPLE_IMAGES)), format_func=lambda i: EXAMPLE_IMAGES[i].stem.replace("_example", ""))
            cmp_img = Image.open(EXAMPLE_IMAGES[cmp_choice]).convert("RGB")
            st.image(cmp_img, caption="Selected Patch", width=220)
        else:
            cmp_up = st.file_uploader("Upload RGB Patch", type=["png", "jpg", "jpeg"], key="cmp_up")
            if cmp_up:
                cmp_img = Image.open(cmp_up).convert("RGB")
                st.image(cmp_img, caption="Uploaded Patch", width=220)

        cmp_method = st.selectbox("Attribution Method:", METHODS_LIST, index=0, key="cmp_method")
        run_cmp = st.button("⚡ Compare All Models", type="primary", use_container_width=True)

    with c_right:
        st.subheader("Side-by-Side Architectural Evaluation")
        if cmp_img is not None:
            if run_cmp:
                models_to_compare = MODELS_LIST[:4]
                cmp_cols = st.columns(len(models_to_compare))

                with st.spinner("Computing predictions and attribution heatmaps across all backbones..."):
                    for idx, m_name in enumerate(models_to_compare):
                        with cmp_cols[idx]:
                            m_pred = predict_patch(m_name, cmp_img, apply_calibration=True)
                            m_attr = generate_attribution_map(m_name, cmp_img, method=cmp_method)
                            m_overlay = render_attribution_overlay(cmp_img, m_attr, use_95th_percentile=False)

                            st.image(m_overlay, use_container_width=True)
                            st.markdown(
                                f"""
                                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px; margin-top: 6px;">
                                    <strong style="color: #1E293B; font-size: 0.95rem;">{m_name}</strong><br>
                                    <span style="font-size: 0.85rem; color: #64748B;">Pred: <strong>{m_pred['top_class']}</strong></span><br>
                                    <span style="font-size: 0.85rem; color: #64748B;">Coarse: <strong>{m_pred['coarse_class']}</strong></span><br>
                                    <span style="font-size: 0.85rem; color: #2563EB;">Confidence: <strong>{m_pred['max_conf']*100:.1f}%</strong></span><br>
                                    <span style="font-size: 0.75rem; color: #94A3B8;">Trainable: {m_pred.get('trainable_pct', 24.6):.1f}% | T={m_pred['temperature']:.2f}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            else:
                st.info("👈 Select a patch and attribution method, then click **'⚡ Compare All Models'**.")


# ==============================================================================
# SECTION 5: PAPER CLAIMS & RIGOR BENCHMARK
# ==============================================================================
elif nav_selection == "📊 Paper Claims & Rigor Benchmark":
    st.markdown('<div class="main-header">📊 Statistical Claims Table & Rigor Benchmark (Phase 11)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Formal hypothesis tests, effect sizes, 95% confidence intervals, CNN baselines, and LoRA parameter-budget efficiency.</div>', unsafe_allow_html=True)

    r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs([
        "📋 Paper Claims & Statistical Verdicts",
        "🤖 CNN Baseline (ConvNeXt-Tiny)",
        "⚡ LoRA vs. Layer Unfreezing",
        "🎯 Hyperparameter Sensitivity Surface"
    ])

    with r_tab1:
        st.subheader("Formal Statistical Evaluation of Core Paper Claims")
        df_claims = load_csv_data("claims_table.csv")
        if df_claims is not None:
            st.dataframe(df_claims, use_container_width=True, hide_index=True)
        else:
            st.info("claims_table.csv not found in results/.")

    with r_tab2:
        st.subheader("Modern CNN Baseline Benchmark (ConvNeXt-Tiny)")
        st.markdown("Matched in parameters (27.8M) to Swin-Tiny (27.5M) across all 5 strategies to ensure findings are not transformer-specific.")
        df_cnn = load_csv_data("phase11_cnn_baseline.csv")
        if df_cnn is not None:
            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                st.dataframe(df_cnn, use_container_width=True, hide_index=True)
            with c_right:
                cnn_fig_path = PROJECT_ROOT / "figures" / "phase11_cnn_vs_transformer.png"
                if cnn_fig_path.exists():
                    st.image(str(cnn_fig_path), caption="ConvNeXt-Tiny vs Swin-Tiny Pareto Frontier", use_container_width=True)

    with r_tab3:
        st.subheader("Low-Rank Adaptation (LoRA) vs. Block Unfreezing")
        st.markdown("Evaluating Low-Rank Adaptation (LoRA $r \\in [8, 16, 32, 64]$) against layer unfreezing at matched parameter budgets.")
        df_lora = load_csv_data("phase11_lora_vs_unfreeze.csv")
        if df_lora is not None:
            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                st.dataframe(df_lora, use_container_width=True, hide_index=True)
            with c_right:
                lora_fig_path = PROJECT_ROOT / "figures" / "phase11_lora_vs_unfreeze.png"
                if lora_fig_path.exists():
                    st.image(str(lora_fig_path), caption="LoRA vs. Layer Unfreezing Pareto Frontier", use_container_width=True)

    with r_tab4:
        st.subheader("Hyperparameter Sensitivity (Validation Isolation)")
        st.markdown("Learning rate and batch size 15-cell grid evaluated strictly on the EuroSAT validation split (zero test set access).")
        df_hyp = load_csv_data("phase11_hyperparameter_sensitivity.csv")
        if df_hyp is not None:
            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                st.dataframe(df_hyp, use_container_width=True, hide_index=True)
            with c_right:
                hyp_fig_path = PROJECT_ROOT / "figures" / "phase11_hyperparameter_sensitivity.png"
                if hyp_fig_path.exists():
                    st.image(str(hyp_fig_path), caption="2D Validation Accuracy Heatmap Surface", use_container_width=True)


# ==============================================================================
# SECTION 6: ABOUT & METHODOLOGY
# ==============================================================================
elif nav_selection == "📖 About & Methodology":
    st.markdown('<div class="main-header">📖 About & Scientific Findings</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Project Overview
    This interactive dashboard represents the empirical re-evaluation and scientific extension of:
    **"Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers for LULC, with Quantitative Evaluation of Explanations"**,
    investigating the methodological claims made in *Khan et al., Scientific Reports (2024)*.

    ---

    ### Core Research Questions & Empirical Answers:

    1. **Is full fine-tuning necessary for Vision Transformers in remote sensing?**
       - **Finding**: **No.** Unfreezing only the final 3 transformer blocks (`unfreeze_last_3`) achieves **97.48% - 98.62%** accuracy, matching complete fine-tuning within 1.0% across ViT, DeiT, and Swin while reducing trainable parameters by **42% to 75%** and training memory by up to 72%.

    2. **Does transfer learning reduce computational cost per epoch?**
       - **Finding**: **No.** As shown by our paired t-test ($t = 0.42, p = 0.681$), per-epoch FLOPs and latency are identical between scratch and pretrained models. Pretraining reduces *sample complexity* (epochs to converge), but computational savings require architectural freezing.

    3. **Can Grad-CAM be applied to Vision Transformers?**
       - **Finding**: **Yes.** The paper's claim that Grad-CAM cannot be applied to ViTs is refuted. Utilizing 2D spatial reshape transforms restoring the $14 \\times 14$ grid yields highly faithful explanations (Deletion AUC $0.28 - 0.34$, $p < 0.0001$).

    4. **Are qualitative 95th-percentile attention maps valid?**
       - **Finding**: **No.** Applying the Adebayo parameter-randomization sanity check demonstrates that blurred-baseline Integrated Gradients acts as an unlearned edge detector ($r = 0.468, p < 0.0001$), failing model sensitivity.

    5. **Does in-domain calibration transfer under geographic shift?**
       - **Finding**: **No.** Temperature scaling fitted on validation sets reduces in-domain ECE to $<0.005$, but fails out-of-domain (ECE remains $0.40 - 0.70$). Selective prediction with confidence thresholds is required.

    ---

    ### Scientific Limitations & Dataset Nuances:
    - **Dynamic World Label Noise**: Dynamic World labels used for out-of-domain evaluation are generated via an automated deep learning model with estimated 10-15% pixel boundary noise.
    - **RGB Spectral Restriction**: While Sentinel-2 provides 13 multispectral bands (including Red Edge, NIR, and SWIR), models were evaluated strictly on optical RGB (B4, B3, B2) to match standard computer vision backbones.
    - **Coarse-Class Harmonization**: Harmonizing EuroSAT's 10 classes to Dynamic World's 5 coarse classes introduces semantic abstraction (e.g. Highways and Residential mapped to Built).
    """)
