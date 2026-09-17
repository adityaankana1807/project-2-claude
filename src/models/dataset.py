"""Turns the synthetic district-month panel + district graph into the
tensors ST-SAGE trains on: a (T, N, C) count cube, an (N, N) normalized
adjacency, SAE-smoothed rate features, and closeness/period/trend lag
windows for a next-month forecasting task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CRIME_CATEGORIES, DATA_PROCESSED, DATA_SYNTHETIC
from src.data.spatial_graph import adjacency_matrix, load_graph
from src.models.sae_smoothing import rolling_eb_spatial_smoothing

CLOSENESS_LAGS = [1, 2, 3]
PERIOD_LAGS = [12, 24, 36]      # same calendar month, 1/2/3 years back
TREND_LAGS = [6, 9, 12]         # medium-run baseline
ALL_LAGS = sorted(set(CLOSENESS_LAGS + PERIOD_LAGS + TREND_LAGS))
MAX_LAG = max(ALL_LAGS)
SAE_WINDOW = 12

# Rough India "wedding / festive season" months (Oct-Feb) used as an explicit
# calendar covariate rather than left for the model to infer from limited
# history -- see synthesize.py's SEASONAL_WEIGHTS for the matching assumption.
FESTIVE_MONTHS = {10, 11, 12, 1, 2}


@dataclass
class GraphData:
    n_districts: int
    adjacency_norm: np.ndarray       # (N, N) row-normalized, self-loop added (GCN-style)
    district_ids: np.ndarray
    state: np.ndarray
    district: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def _gcn_normalize(adj: np.ndarray) -> np.ndarray:
    a = adj + np.eye(adj.shape[0])
    deg = a.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    d_inv_sqrt[mask] = np.power(deg[mask], -0.5)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ a @ d_mat


def load_panel_cube():
    """Pivot the synthetic panel into (T, N, C) counts + (T, N) exposure,
    time-ordered, district-ordered to match the graph's node ids."""
    panel = pd.read_csv(DATA_SYNTHETIC / "district_month_panel.csv")
    panel["t"] = (panel["year"] - panel["year"].min()) * 12 + (panel["month"] - 1)
    n_districts = panel["district_id"].nunique()
    n_t = panel["t"].nunique()

    panel = panel.sort_values(["t", "district_id"])
    counts = np.zeros((n_t, n_districts, len(CRIME_CATEGORIES)), dtype=np.float32)
    exposure = np.zeros((n_t, n_districts), dtype=np.float32)

    for ci, cat in enumerate(CRIME_CATEGORIES):
        arr = panel.pivot(index="t", columns="district_id", values=cat).to_numpy()
        counts[:, :, ci] = arr
    exposure[:] = panel.pivot(index="t", columns="district_id", values="female_population").to_numpy()

    months = panel.drop_duplicates("t").sort_values("t")["month"].to_numpy()
    years = panel.drop_duplicates("t").sort_values("t")["year"].to_numpy()
    return counts, exposure, months, years


def build_graph_data() -> GraphData:
    districts = pd.read_csv(DATA_PROCESSED / "districts_harmonized.csv").sort_values("district_id")
    g = load_graph()
    adj = adjacency_matrix(g)
    adj_norm = _gcn_normalize(adj)
    return GraphData(
        n_districts=len(districts),
        adjacency_norm=adj_norm.astype(np.float32),
        district_ids=districts["district_id"].to_numpy(),
        state=districts["state"].to_numpy(),
        district=districts["district"].to_numpy(),
        lat=districts["lat"].to_numpy(),
        lon=districts["lon"].to_numpy(),
    )


class CrimeForecastDataset(Dataset):
    """One example = all N districts at a single target month t (a full
    graph snapshot), so a GCN layer can mix neighbour information within a
    batch item. `t_indices` controls which months belong to this split."""

    def __init__(self, counts, exposure, sae_rate, months, t_indices):
        self.counts = counts
        self.exposure = exposure
        self.sae_rate = sae_rate
        self.months = months
        self.t_indices = t_indices

    def __len__(self):
        return len(self.t_indices)

    def __getitem__(self, idx):
        t = self.t_indices[idx]
        closeness = np.stack([self.counts[t - lag] for lag in CLOSENESS_LAGS], axis=0)  # (L,N,C)
        period = np.stack([self.counts[t - lag] for lag in PERIOD_LAGS], axis=0)
        trend = np.stack([self.counts[t - lag] for lag in TREND_LAGS], axis=0)
        sae_feat = self.sae_rate[t]  # (N, C) causal smoothed rate as of t
        neighbor_prev = self.counts[t - 1]  # used again for the near-repeat gate

        month = self.months[t]
        month_onehot = np.zeros(12, dtype=np.float32)
        month_onehot[month - 1] = 1.0
        festive = np.float32(1.0 if month in FESTIVE_MONTHS else 0.0)

        target = self.counts[t]  # (N, C)
        log_exposure = np.log(np.clip(self.exposure[t], 1.0, None)).astype(np.float32)

        return {
            "closeness": torch.from_numpy(closeness).float(),
            "period": torch.from_numpy(period).float(),
            "trend": torch.from_numpy(trend).float(),
            "sae_feat": torch.from_numpy(sae_feat).float(),
            "prev_month": torch.from_numpy(neighbor_prev).float(),
            "month_onehot": torch.from_numpy(month_onehot).float(),
            "festive": torch.tensor(festive).float(),
            "log_exposure": torch.from_numpy(log_exposure).float(),
            "target": torch.from_numpy(target).float(),
            "t": t,
        }


def make_datasets(train_end_year: int = 2020, val_end_year: int = 2022):
    counts, exposure, months, years = load_panel_cube()
    graph = build_graph_data()
    sae_rate = rolling_eb_spatial_smoothing(counts, exposure, graph.adjacency_norm, window=SAE_WINDOW)

    valid_t = np.arange(MAX_LAG, len(months))
    valid_years = years[valid_t]

    train_t = valid_t[valid_years <= train_end_year]
    val_t = valid_t[(valid_years > train_end_year) & (valid_years <= val_end_year)]
    test_t = valid_t[valid_years > val_end_year]

    train_ds = CrimeForecastDataset(counts, exposure, sae_rate, months, train_t)
    val_ds = CrimeForecastDataset(counts, exposure, sae_rate, months, val_t)
    test_ds = CrimeForecastDataset(counts, exposure, sae_rate, months, test_t)

    # Empirical global log-rate per category on the training split only, used
    # to sensibly initialize the model's output bias (see STSage) so an
    # untrained/undertrained network starts near a realistic baseline instead
    # of an arbitrary one -- this is what keeps rare, high-exposure districts
    # from producing runaway extrapolated counts.
    train_counts = counts[train_t].sum(axis=(0, 1))    # (C,)
    train_exposure = exposure[train_t].sum()           # scalar, same exposure applies to every category
    global_log_rate = np.log(np.clip(train_counts / max(train_exposure, 1.0), 1e-8, None)).astype(np.float32)

    return train_ds, val_ds, test_ds, graph, global_log_rate
