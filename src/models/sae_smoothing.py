"""Layer 1 -- Small Area Estimation (SAE) smoothing.

Implements the classic Marshall (1991) empirical-Bayes shrinkage estimator
used throughout disease/crime mapping (and the exact technique the
"Crime Against Women in India: District-Level Risk Estimation Using Small
Area Estimation" paper is built on): raw district rates are noisy when the
population/exposure is small, so each district's rate is shrunk toward a
reference mean by an amount that depends on how much statistical
information that district actually has.

We extend the classic (global-mean) version with *spatial* borrowing: the
reference each district shrinks toward is a graph-weighted local average of
its k-NN neighbours' rates (per the research synthesis: "borrow strength
across neighbouring/similar areas"), not just the flat national mean.

Crucially this is computed **causally**: the smoothed value available to
predict month t only uses counts/exposure from a trailing window strictly
before t, so it can be used as a leakage-free model feature.
"""
from __future__ import annotations

import numpy as np


def _marshall_eb_shrinkage(y: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Global empirical-Bayes shrinkage weights for one time slice.

    y: (N,) trailing-window counts per district
    e: (N,) trailing-window exposure (population-months) per district
    returns w: (N,) shrinkage weight in [0, 1] (1 = trust the raw rate fully)
    """
    e = np.clip(e, 1e-6, None)
    m = y.sum() / e.sum()  # overall (national) rate
    r = y / e
    s2 = np.sum(e * (r - m) ** 2) / e.sum()  # exposure-weighted variance of raw rates
    phi = e.mean()
    a_hat = max(s2 - m / phi, 0.0)  # method-of-moments prior variance component
    w = a_hat / (a_hat + m / e)
    return np.clip(w, 0.0, 1.0), m


def rolling_eb_spatial_smoothing(
    counts: np.ndarray,       # (T, N, C)
    exposure: np.ndarray,     # (T, N)  e.g. population or female_population
    adjacency: np.ndarray,    # (N, N) nonnegative edge weights (no self-loop needed)
    window: int = 12,
) -> np.ndarray:
    """Returns smoothed_rate of shape (T, N, C).

    smoothed_rate[t] is computed from counts/exposure in [t-window, t-1] only
    (never t itself or later), so it is safe to use as a predictor feature
    for month t. The first `window` months fall back to the (unsmoothed)
    global mean rate since there isn't enough trailing history yet.
    """
    T, N, C = counts.shape
    smoothed = np.zeros((T, N, C), dtype=np.float64)

    # Self-inclusive, row-normalized "local mean" operator: each district's
    # local reference is its own rate plus its graph neighbours' rates,
    # weighted by edge strength. Self weight of 1.0 matches the maximum
    # possible neighbour edge weight (1/(1+distance_km) at distance 0).
    local_op = adjacency.copy()
    np.fill_diagonal(local_op, 1.0)
    row_sums = local_op.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    local_op = local_op / row_sums  # (N, N) rows sum to 1

    for t in range(T):
        lo = max(0, t - window)
        if lo == t:  # no history at all yet
            smoothed[t] = 0.0
            continue
        y_win = counts[lo:t].sum(axis=0)      # (N, C)
        e_win = exposure[lo:t].sum(axis=0)    # (N,)

        for c in range(C):
            y = y_win[:, c]
            w, m_global = _marshall_eb_shrinkage(y, e_win)
            r = y / np.clip(e_win, 1e-6, None)
            local_mean = local_op @ r  # graph-weighted local reference rate
            smoothed[t, :, c] = w * r + (1 - w) * local_mean

    return smoothed
