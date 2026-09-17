"""End-to-end crime-linkage demo (Layer 3): generates synthetic incidents,
scores all candidate pairs with Jaccard vs. the Tonkin et al. (2025)
low-base-rate similarity metric, combines similarity + geographic distance +
temporal proximity via logistic regression, and reports the
decision-relevant metrics the 2025 UK ViCLAS benchmarking study used
(AUC, % of true links found in the top-100/top-500 ranked candidates per
query, and Median First Rank) rather than AUC alone -- because true linked
pairs are a vanishing fraction of all possible pairs, so AUC alone can look
good while a ranked shortlist is still useless to an analyst.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.config import DATA_SYNTHETIC, OUTPUTS_DIR, RANDOM_SEED
from src.linkage import synthetic_incidents
from src.linkage.similarity import (
    day_gap_matrix,
    distance_matrix_km,
    jaccard_matrix,
    tonkin2025_matrix,
)


def _upper_triangle_pairs(n: int):
    i, j = np.triu_indices(n, k=1)
    return i, j


def median_first_rank(scores: np.ndarray, linked: np.ndarray, n: int) -> tuple[float, float, float]:
    """For each query incident with >=1 true series-mate, rank all other
    incidents by score and find the best (lowest) rank of a true linked
    incident. Returns (median first rank, mean top-100 hit rate, mean
    top-500 hit rate)."""
    score_mat = np.full((n, n), -np.inf)
    i, j = _upper_triangle_pairs(n)
    score_mat[i, j] = scores
    score_mat[j, i] = scores
    linked_mat = np.zeros((n, n), dtype=bool)
    linked_mat[i, j] = linked
    linked_mat[j, i] = linked

    first_ranks, top100_hits, top500_hits = [], [], []
    for q in range(n):
        true_mates = np.where(linked_mat[q])[0]
        if len(true_mates) == 0:
            continue
        order = np.argsort(-score_mat[q])
        order = order[order != q]
        rank_of = {idx: r + 1 for r, idx in enumerate(order)}
        ranks = [rank_of[m] for m in true_mates]
        first_ranks.append(min(ranks))
        top100_hits.append(np.mean([r <= 100 for r in ranks]))
        top500_hits.append(np.mean([r <= 500 for r in ranks]))

    return float(np.median(first_ranks)), float(np.mean(top100_hits)), float(np.mean(top500_hits))


def main():
    incidents = synthetic_incidents.generate()
    df, mo = incidents.incidents, incidents.mo_matrix
    n = len(df)

    jacc = jaccard_matrix(mo)
    tonkin = tonkin2025_matrix(mo)
    dist_km = distance_matrix_km(df["lat"].to_numpy(), df["lon"].to_numpy())
    day_gap = day_gap_matrix(df["date"].to_numpy())

    i, j = _upper_triangle_pairs(n)
    series = df["series_id"].to_numpy()
    linked = (series[i] == series[j]) & (series[i] != -1)

    jacc_pairs, tonkin_pairs = jacc[i, j], tonkin[i, j]
    dist_pairs, gap_pairs = dist_km[i, j], day_gap[i, j]

    auc_jaccard = roc_auc_score(linked, jacc_pairs)
    auc_tonkin = roc_auc_score(linked, tonkin_pairs)

    # Combined model: similarity + distance + time via logistic regression,
    # exactly the feature set used across the reviewed linkage literature.
    X = np.column_stack([tonkin_pairs, -dist_pairs, -gap_pairs])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, linked, np.arange(len(linked)), test_size=0.3, random_state=RANDOM_SEED, stratify=linked
    )
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    combined_scores_test = clf.predict_proba(X_test)[:, 1]
    auc_combined = roc_auc_score(y_test, combined_scores_test)

    # Decision-relevant ranking metrics computed over the FULL pair set
    # (Jaccard, Tonkin, and the fitted combined model), per Tonkin et al. (2025).
    combined_scores_all = clf.predict_proba(X)[:, 1]
    mfr_jacc, top100_jacc, top500_jacc = median_first_rank(jacc_pairs, linked, n)
    mfr_tonkin, top100_tonkin, top500_tonkin = median_first_rank(tonkin_pairs, linked, n)
    mfr_combined, top100_combined, top500_combined = median_first_rank(combined_scores_all, linked, n)

    metrics = {
        "n_incidents": n,
        "n_true_linked_pairs": int(linked.sum()),
        "n_total_pairs": int(len(linked)),
        "auc": {"jaccard": auc_jaccard, "tonkin2025": auc_tonkin, "combined_logreg": auc_combined},
        "median_first_rank": {"jaccard": mfr_jacc, "tonkin2025": mfr_tonkin, "combined_logreg": mfr_combined},
        "top100_hit_rate": {"jaccard": top100_jacc, "tonkin2025": top100_tonkin, "combined_logreg": top100_combined},
        "top500_hit_rate": {"jaccard": top500_jacc, "tonkin2025": top500_tonkin, "combined_logreg": top500_combined},
        "combined_model_coefficients": dict(zip(
            ["tonkin_similarity", "neg_distance_km", "neg_day_gap"], clf.coef_[0].tolist()
        )),
    }
    with open(OUTPUTS_DIR / "linkage_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, scores in [("Jaccard", jacc_pairs), ("Tonkin et al. 2025", tonkin_pairs)]:
        fpr, tpr, _ = roc_curve(linked, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(linked, scores):.3f})")
    fpr, tpr, _ = roc_curve(y_test, combined_scores_test)
    ax.plot(fpr, tpr, label=f"Combined (dist+time+MO) (AUC={auc_combined:.3f})", linestyle="--")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Crime-linkage ROC: MO similarity metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "linkage_roc.png", dpi=150)
    print(f"Saved plot to {OUTPUTS_DIR / 'linkage_roc.png'}")

    out_dir = DATA_SYNTHETIC / "linkage"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "incidents.csv", index=False)
    np.save(out_dir / "mo_matrix.npy", mo)

    return metrics


if __name__ == "__main__":
    main()
