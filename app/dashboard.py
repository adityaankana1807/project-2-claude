"""ST-SAGE dashboard: an executable, interactive view of the India crime
forecasting + linkage project. Run with:

    .venv/Scripts/streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA_SYNTHETIC, OUTPUTS_DIR

st.set_page_config(page_title="ST-SAGE: India Crime Forecasting", layout="wide")


@st.cache_data
def load_forecast_artifacts():
    with np.load(OUTPUTS_DIR / "test_predictions.npz", allow_pickle=True) as npz:
        arrays = {k: npz[k] for k in npz.files}  # materialize before the file handle closes
    with open(OUTPUTS_DIR / "eval_metrics.json") as f:
        eval_metrics = json.load(f)
    with open(OUTPUTS_DIR / "training_history.json") as f:
        history = json.load(f)
    return arrays, eval_metrics, history


@st.cache_data
def load_panel():
    return pd.read_csv(DATA_SYNTHETIC / "district_month_panel.csv")


@st.cache_data
def load_linkage():
    with open(OUTPUTS_DIR / "linkage_metrics.json") as f:
        metrics = json.load(f)
    incidents = pd.read_csv(DATA_SYNTHETIC / "linkage" / "incidents.csv")
    return metrics, incidents


st.title("ST-SAGE: Spatio-Temporal Small-Area Graph Estimator")
st.caption(
    "A novel hybrid algorithm for India crime-against-women forecasting and serial-offender "
    "linkage, built on real NCRB district data + Census 2011, with a synthetic monthly panel "
    "extending the real 2001-2012 trend forward. See the About tab for data provenance and "
    "algorithm design."
)

tab_map, tab_district, tab_linkage, tab_about = st.tabs(
    ["Hotspot Forecast Map", "District Drilldown", "Crime Linkage Demo", "About / Algorithm"]
)

with tab_map:
    npz, eval_metrics, history = load_forecast_artifacts()  # npz is now a plain dict of arrays
    categories = list(npz["categories"])
    month_labels = [f"test month {i+1}" for i in range(npz["pred"].shape[0])]
    month_idx = st.select_slider("Test month", options=list(range(len(month_labels))),
                                  format_func=lambda i: month_labels[i])
    cat_choice = st.selectbox("Category", ["All (total)"] + categories)

    pred = npz["pred"][month_idx]
    true = npz["true"][month_idx]
    if cat_choice == "All (total)":
        pred_vals, true_vals = pred.sum(axis=-1), true.sum(axis=-1)
    else:
        ci = categories.index(cat_choice)
        pred_vals, true_vals = pred[:, ci], true[:, ci]

    df_map = pd.DataFrame({
        "state": npz["state"], "district": npz["district"],
        "lat": npz["lat"].astype(float), "lon": npz["lon"].astype(float),
        "predicted": pred_vals, "actual": true_vals,
    })
    df_map["abs_error"] = (df_map["predicted"] - df_map["actual"]).abs()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter_map(
            df_map, lat="lat", lon="lon", size="predicted", color="predicted",
            hover_name="district", hover_data=["state", "actual"],
            color_continuous_scale="OrRd", size_max=22, zoom=3.2,
            center={"lat": 22.5, "lon": 80}, map_style="open-street-map",
            title="ST-SAGE predicted risk",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig2 = px.scatter_map(
            df_map, lat="lat", lon="lon", size="actual", color="actual",
            hover_name="district", hover_data=["state", "predicted"],
            color_continuous_scale="Blues", size_max=22, zoom=3.2,
            center={"lat": 22.5, "lon": 80}, map_style="open-street-map",
            title="Actual (synthetic ground truth)",
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Held-out test metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision@10 (hotspot districts)", f"{eval_metrics['precision_at_10']:.2f}")
    c2.metric("Precision@50", f"{eval_metrics['precision_at_50']:.2f}")
    c3.metric("National monthly total MAE", f"{eval_metrics['national_total_mae']:.0f}")

    st.plotly_chart(
        go.Figure(data=[
            go.Scatter(y=history["train_loss"], name="train NLL"),
            go.Scatter(y=history["val_loss"], name="val NLL"),
        ]).update_layout(title="Training curve (Negative-Binomial NLL)", height=300),
        width="stretch",
    )
    st.image(str(OUTPUTS_DIR / "forecast_vs_actual.png"))

with tab_district:
    panel = load_panel()
    districts = sorted(panel["district"].unique())
    d = st.selectbox("District", districts, index=districts.index("JAIPUR") if "JAIPUR" in districts else 0)
    sub = panel[panel["district"] == d].sort_values(["year", "month"])
    sub["date"] = pd.to_datetime(sub[["year", "month"]].assign(day=1))
    st.line_chart(sub.set_index("date")["total"], height=300)
    st.dataframe(sub[["date", "state", "total", "crime_rate_per_100k_women"] + list(
        c for c in sub.columns if c in ["Rape", "Kidnapping_&_Abduction", "Dowry_Deaths"]
    )].tail(24), width="stretch")

with tab_linkage:
    metrics, incidents = load_linkage()
    st.subheader("Layer 3: serial-offender linkage on synthetic FIR-style incidents")
    st.write(
        f"{metrics['n_incidents']} incidents, {metrics['n_true_linked_pairs']} true linked pairs "
        f"out of {metrics['n_total_pairs']:,} candidate pairs."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC (Jaccard)", f"{metrics['auc']['jaccard']:.3f}")
    c2.metric("AUC (Tonkin et al. 2025 metric)", f"{metrics['auc']['tonkin2025']:.3f}")
    c3.metric("AUC (combined MO+distance+time)", f"{metrics['auc']['combined_logreg']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Median First Rank (Jaccard)", f"{metrics['median_first_rank']['jaccard']:.0f}")
    c5.metric("Median First Rank (Tonkin)", f"{metrics['median_first_rank']['tonkin2025']:.0f}")
    c6.metric("Median First Rank (combined)", f"{metrics['median_first_rank']['combined_logreg']:.0f}")

    st.image(str(OUTPUTS_DIR / "linkage_roc.png"))

    st.caption(
        "Note: on this synthetic dataset, plain Jaccard slightly outperforms the Tonkin et al. "
        "(2025) metric used alone -- the opposite of that paper's UK ViCLAS finding. Our "
        "synthetic MO noise process doesn't fully reproduce the extreme low-base-rate sparsity "
        "pathology the new metric was designed to fix. Both are dominated by the combined "
        "distance+time+MO model, which is near-perfect here because our synthetic 'marauder' "
        "offenders cluster tightly in space relative to the national scale -- a limitation of "
        "the synthetic generator, documented in the README, not a general claim about MO metrics."
    )

    fig = px.scatter_map(
        incidents, lat="lat", lon="lon", color="series_id",
        hover_data=["district", "state", "date", "is_singleton"],
        zoom=3.2, center={"lat": 22.5, "lon": 80}, map_style="open-street-map",
        title="Synthetic incidents (color = true series id, -1 = singleton)",
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, width="stretch")

with tab_about:
    st.markdown((Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8"))
