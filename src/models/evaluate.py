"""Evaluates a trained ST-SAGE checkpoint on the held-out test months:
per-category MAE/RMSE, and a hotspot precision@k metric (of the top-k
districts by predicted risk, how many are actually in the top-k by realized
count) -- the decision-relevant metric the linkage literature (and disease
mapping / hotspot policing practice generally) favours over pure error
averages, since in practice only a ranked shortlist of districts gets acted
on.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CRIME_CATEGORIES, MODELS_DIR, OUTPUTS_DIR
from src.models.dataset import make_datasets
from src.models.st_sage import STSage
from src.models.train import _collate_with_adjacency


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    true_top = set(np.argsort(-y_true)[:k].tolist())
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    return len(true_top & pred_top) / k


def evaluate(k_list=(10, 25, 50)):
    _, _, test_ds, graph, _ = make_datasets()
    adj_norm = torch.from_numpy(graph.adjacency_norm).float()
    test_t = test_ds.t_indices

    ckpt = torch.load(MODELS_DIR / "st_sage.pt", map_location="cpu")
    model = STSage(n_categories=ckpt["n_categories"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=_collate_with_adjacency)

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            mu, _ = model(batch, adj_norm)
            all_pred.append(mu.numpy())
            all_true.append(batch["target"].numpy())
    pred = np.concatenate(all_pred, axis=0)  # (T_test, N, C)
    true = np.concatenate(all_true, axis=0)

    mae = np.abs(pred - true).mean(axis=(0, 1))
    rmse = np.sqrt(((pred - true) ** 2).mean(axis=(0, 1)))

    pred_total = pred.sum(axis=-1)  # (T_test, N)
    true_total = true.sum(axis=-1)
    precisions = {}
    for k in k_list:
        p = [precision_at_k(true_total[t], pred_total[t], k) for t in range(pred_total.shape[0])]
        precisions[f"precision_at_{k}"] = float(np.mean(p))

    metrics = {
        "mae_per_category": dict(zip(CRIME_CATEGORIES, mae.tolist())),
        "rmse_per_category": dict(zip(CRIME_CATEGORIES, rmse.tolist())),
        "national_total_mae": float(np.abs(pred_total.sum(axis=1) - true_total.sum(axis=1)).mean()),
        **precisions,
    }
    with open(OUTPUTS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    np.savez(
        OUTPUTS_DIR / "test_predictions.npz",
        pred=pred, true=true, t_indices=np.asarray(test_t),
        district_ids=graph.district_ids, state=graph.state, district=graph.district,
        lat=graph.lat, lon=graph.lon, categories=np.asarray(CRIME_CATEGORIES),
    )

    # Plot national monthly total: predicted vs actual over the test window.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(true_total.sum(axis=1), label="actual", marker="o")
    ax.plot(pred_total.sum(axis=1), label="ST-SAGE predicted", marker="x")
    ax.set_xlabel("test month index")
    ax.set_ylabel("national total crime-against-women incidents")
    ax.set_title("ST-SAGE: predicted vs actual (held-out test months)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "forecast_vs_actual.png", dpi=150)
    print(f"Saved plot to {OUTPUTS_DIR / 'forecast_vs_actual.png'}")

    return metrics, pred, true, graph


if __name__ == "__main__":
    evaluate()
