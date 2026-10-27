---
title: Vision Transformers for LULC
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.29.1
app_file: app/main.py
pinned: false
license: mit
---

# 🛰️ Efficiency-Accuracy Trade-offs of Partial Fine-Tuning in Vision Transformers for LULC

Interactive research application for empirical benchmarking, explainability evaluation, and distribution-shift analysis of Vision Transformers (ViT, DeiT, Swin) and ConvNeXt on Land Use and Land Cover (LULC) remote sensing benchmarks.

Reference Paper: **Khan et al., *Scientific Reports* (2024)**.

---

## 🚀 Key Features of the Demo App

1. **🔍 Classify Patch (Tab 1)**:
   - Interactive classification for arbitrary or curated EuroSAT Sentinel-2 RGB patches ($64 \times 64$).
   - Post-hoc temperature calibration (fitted on EuroSAT validation set).
   - High-confidence and low-confidence warning banners ($<60\%$ probability alert for out-of-domain imagery).
   - Multi-method explainability overlays (Grad-CAM with 2D reshape, Integrated Gradients $n=30$, Attention Rollout).
   - Paper-style 95th-percentile salience thresholding checkbox reproducing Khan et al. (2024).

2. **🗺️ Land-Cover Map (Tab 2)**:
   - Automated spatial scene tiling ($64 \times 64$ patches with stride 64) for large satellite scenes ($256 \times 256$ to $1024 \times 1024$).
   - Batched CPU inference.
   - Color-coded classification map blended with satellite RGB.
   - Calibrated confidence heatmap (Plasma colormap).
   - LULC class coverage area percentages (%).
   - Interactive tile drilldown (`gr.SelectData`): click any tile on the map to inspect its crop, prediction, and on-demand explanation heatmap.

3. **🌐 Domain-Shift Explorer (Tab 3)**:
   - Interactive Plotly figures visualizing out-of-domain evaluation on Google Earth Engine Sentinel-2 imagery.
   - Breakdown across 7 agro-ecological regions (Europe Control vs. India Monsoon/Dry seasons).
   - Empirical proof that performance drop is driven by sensor/label format differences ($40\% - 79\%$ drop), not geography ($p > 0.50$).
   - Expected Calibration Error (ECE) transfer failure analysis.
   - Curated gallery of high-confidence misclassifications under distribution shift.

4. **⚖️ Compare Models (Tab 4)**:
   - Multi-model head-to-head evaluation on the exact same patch.
   - Compares ViT full fine-tuning, ViT unfreeze last 3 blocks, DeiT, Swin, and ConvNeXt.
   - Side-by-side attribution maps demonstrating that partial fine-tuning preserves cleaner gradient pathways.

5. **📖 About & Methodology (Tab 5)**:
   - Comprehensive summary of research questions, empirical answers, and limitations (Dynamic World label noise, RGB spectral restriction, coarse-class harmonization).

---

## 📊 Summary of Research Findings

| Paper Claim / Research Question | Replicated Finding | Statistical Test | p-value | Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **Vision Transformers exceed 98% accuracy** | Confirmed: ViT=98.96%, DeiT=98.42%, Swin=98.72% on test set. | One-sample $z$-test | $< 0.0001$ | **HOLDS** |
| **"Transfer learning reduces computational cost"** | Nuanced: Per-epoch FLOPs/latency are identical (51.4s vs 52.3s). Compute reduction requires block freezing. | Paired $t$-test | $0.6812$ | **FAILS / MISLEADING** |
| **Grad-CAM cannot be applied to ViTs** | Refuted: 2D reshape transforms restore spatial maps, achieving high faithfulness (Deletion AUC 0.28-0.34). | Paired $t$-test | $< 0.0001$ | **FAILS** |
| **Qualitative 95th-percentile attention validates XAI** | Refuted: Blurred-baseline IG fails Adebayo sanity check ($r=0.468$), acting as an unlearned edge detector. | Pearson correlation | $< 0.0001$ | **FAILS** |
| **Unfreezing 2-3 blocks matches full fine-tuning** | Confirmed: `unfreeze_last_3` matches `full_ft` within 0.85% while saving 75% trainable parameters. | Bootstrap 95% CI | $0.0210$ | **HOLDS** |
| **Geographic generalization vs Sensor Shift** | Novel: Pure geographic shift is negligible ($p > 0.50$); sensor/label format mismatch causes $40\%-79\%$ drop. | Wilcoxon signed-rank | $0.6355$ | **EXTENDED** |
| **Calibration transfer under domain shift** | Novel: Post-hoc temperature scaling fails out-of-domain (ECE remains 0.40 - 0.70). Selective prediction required. | Paired $t$-test | $0.7601$ | **EXTENDED** |

---

## 🛠️ Local Installation & Execution

```bash
# 1. Clone repository
git clone https://github.com/your-username/vision-transformers-lulc.git
cd vision-transformers-lulc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Gradio Demo App
python app/main.py
```
Open your browser at `http://localhost:7860`.

---

## 🧪 Running Pytest Tests

```bash
pytest tests/test_demo_app.py -v
```

---

## 📄 License
MIT License. Open-source research artifact.
