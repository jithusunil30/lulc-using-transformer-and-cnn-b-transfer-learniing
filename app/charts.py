"""
app/charts.py - Interactive Plotly Charts for Domain-Shift Explorer
===================================================================
Visualizes out-of-domain evaluation results from Phase 9 and Phase 10:
- Accuracy by region and season
- Sensor/Label format shift vs pure geographic shift
- In-domain vs shifted calibration failure (ECE)
"""

import sys
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BREAKDOWN_CSV = PROJECT_ROOT / "results" / "geographic_shift_region_season_breakdown.csv"
SHIFT_EVAL_CSV = PROJECT_ROOT / "results" / "geographic_shift_evaluation.csv"
CALIBRATION_CSV = PROJECT_ROOT / "results" / "phase10_calibration_summary.csv"


def plot_region_season_accuracy(selected_model_family: str = "ViT", selected_strategy: str = "unfreeze_last_3"):
    """
    Renders grouped bar chart of accuracy across 7 regions and seasons for chosen model & strategy.
    """
    if not BREAKDOWN_CSV.exists():
        fig = go.Figure()
        fig.add_annotation(text="Results file not found.", showarrow=False)
        return fig

    df = pd.read_csv(BREAKDOWN_CSV)
    sub = df[(df["model_family"].str.upper() == selected_model_family.upper()) &
             (df["strategy"].str.lower() == selected_strategy.lower())].copy()

    if sub.empty:
        sub = df[df["model_family"].str.upper() == selected_model_family.upper()].copy()

    sub["Region Display"] = sub["region"].str.replace("_", " ").str.title()
    sub["Season Display"] = sub["season"].str.capitalize()
    sub["Label"] = sub["Region Display"] + " (" + sub["Season Display"] + ")"

    fig = px.bar(
        sub,
        x="Region Display",
        y="accuracy",
        color="Season Display",
        barmode="group",
        text="accuracy",
        color_discrete_map={
            "Summer": "#F39C12",
            "Winter": "#3498DB",
            "Dry": "#E67E22",
            "Monsoon": "#2ECC71"
        },
        labels={"accuracy": "Coarse Accuracy (%)", "Region Display": "Geographic Region", "Season Display": "Season"},
        title=f"Accuracy by Geographic Region & Season ({selected_model_family} - {selected_strategy})"
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[0, 100], title="Coarse-Class Accuracy (%)"),
        xaxis=dict(title="Geographic Region"),
        legend=dict(title="Season", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, system-ui, sans-serif")
    )
    return fig


def plot_sensor_vs_geographic_drop():
    """
    Renders grouped chart comparing in-domain accuracy, Europe-DW control, and India-DW shifted accuracy.
    Empirically demonstrates that domain degradation is caused by sensor/label mismatch, NOT geography.
    """
    if not SHIFT_EVAL_CSV.exists():
        fig = go.Figure()
        fig.add_annotation(text="Shift evaluation CSV not found.", showarrow=False)
        return fig

    df = pd.read_csv(SHIFT_EVAL_CSV)
    df["Model & Strategy"] = df["model_family"] + " (" + df["strategy"] + ")"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="EuroSAT Test (In-Domain)",
        x=df["Model & Strategy"],
        y=df["eurosat_accuracy"],
        marker_color="#27AE60",
        text=df["eurosat_accuracy"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="Europe-DW (Control: Sensor/Label Shift)",
        x=df["Model & Strategy"],
        y=df["europe_dw_accuracy"],
        marker_color="#E67E22",
        text=df["europe_dw_accuracy"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="India-DW (Full Shift: Geography + Sensor)",
        x=df["Model & Strategy"],
        y=df["india_dw_accuracy"],
        marker_color="#C0392B",
        text=df["india_dw_accuracy"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title="Distribution Shift Breakdown: In-Domain vs. Europe Control vs. India Shift",
        yaxis=dict(range=[0, 115], title="Coarse Accuracy (%)"),
        xaxis=dict(tickangle=-45, title="Model Architecture & Fine-Tuning Strategy"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=100),
        font=dict(family="Inter, system-ui, sans-serif")
    )
    return fig


def plot_calibration_ece_transfer():
    """
    Compares Expected Calibration Error (ECE) across domains:
    EuroSAT In-Domain (Uncalibrated vs. Calibrated) vs India-DW Shifted.
    """
    if not CALIBRATION_CSV.exists():
        fig = go.Figure()
        fig.add_annotation(text="Calibration summary CSV not found.", showarrow=False)
        return fig

    df = pd.read_csv(CALIBRATION_CSV)
    df["Model Label"] = df["model_family"] + " (" + df["strategy"] + ")"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="In-Domain Uncalibrated ECE",
        x=df["Model Label"],
        y=df["eurosat_coarse_ece_uncal"],
        marker_color="#3498DB",
        text=df["eurosat_coarse_ece_uncal"].apply(lambda x: f"{x:.3f}"),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="In-Domain Calibrated ECE (Temp Scaling)",
        x=df["Model Label"],
        y=df["eurosat_coarse_ece_cal"],
        marker_color="#2ECC71",
        text=df["eurosat_coarse_ece_cal"].apply(lambda x: f"{x:.3f}"),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="Shifted ECE on India-DW (Zero Transfer)",
        x=df["Model Label"],
        y=df["india_dw_ece_cal"],
        marker_color="#E74C3C",
        text=df["india_dw_ece_cal"].apply(lambda x: f"{x:.3f}"),
        textposition="outside"
    ))

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title="Calibration Failure Under Domain Shift: Expected Calibration Error (ECE)",
        yaxis=dict(range=[0, 0.85], title="Expected Calibration Error (Lower is Better)"),
        xaxis=dict(tickangle=-45, title="Model & Strategy"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=100),
        font=dict(family="Inter, system-ui, sans-serif")
    )
    return fig
