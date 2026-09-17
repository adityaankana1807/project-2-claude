"""Crime-linkage similarity metrics, per the research synthesis:

* Jaccard's coefficient -- the field-standard MO similarity measure
  (Bennell & Canter tradition; used in Woodhams & Labuschagne 2012, Halford
  2023, Tonkin et al. 2011).
* The Tonkin et al. (2025) low-base-rate-robust replacement:
      sim = log(1 + 3 + 3*a - (b + c))
  where a = behaviours present in BOTH crimes, b/c = behaviours present in
  only one. It up-weights joint presence 3x relative to Jaccard, which the
  2025 UK ViCLAS benchmark (10,918 solved sexual offences, 446 MO variables,
  186 of them present in <1% of crimes) showed measurably outperforms
  Jaccard on exactly this kind of sparse-behaviour data -- the situation
  India's inconsistently-recorded FIR/MO data is expected to share.
* Haversine inter-crime distance and day-gap -- the two most
  crime-type-agnostic linkage features across the whole literature reviewed
  (Tonkin et al. 2011 unpublished ms.; Halford 2023), used here as
  additional, always-computable ranking features alongside MO similarity.
"""
from __future__ import annotations

import numpy as np


def jaccard(mo_a: np.ndarray, mo_b: np.ndarray) -> float:
    a = int(np.sum((mo_a == 1) & (mo_b == 1)))
    union = int(np.sum((mo_a == 1) | (mo_b == 1)))
    return a / union if union > 0 else 0.0


_TONKIN_FLOOR = 1e-6  # the raw "3 + 3a - (b+c)" term goes negative for very
# dissimilar pairs (the log is undefined there); we floor it to a small
# positive epsilon rather than change the paper's formula, so heavily
# mismatched pairs simply bottom out at the lowest similarity instead of
# producing NaN.


def tonkin2025_similarity(mo_a: np.ndarray, mo_b: np.ndarray) -> float:
    both_present = int(np.sum((mo_a == 1) & (mo_b == 1)))
    only_a = int(np.sum((mo_a == 1) & (mo_b == 0)))
    only_b = int(np.sum((mo_a == 0) & (mo_b == 1)))
    inner = 1 + 3 + 3 * both_present - (only_a + only_b)
    return float(np.log(max(inner, _TONKIN_FLOOR)))


def jaccard_matrix(mo_matrix: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Jaccard over a (N, M) binary matrix -> (N, N)."""
    mo = mo_matrix.astype(np.float32)
    both = mo @ mo.T
    row_sums = mo.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - both
    union[union == 0] = 1.0
    return both / union


def tonkin2025_matrix(mo_matrix: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Tonkin et al. (2025) similarity -> (N, N)."""
    mo = mo_matrix.astype(np.float32)
    both = mo @ mo.T                                   # a
    row_sums = mo.sum(axis=1, keepdims=True)
    only_a_plus_only_b = (row_sums + row_sums.T) - 2 * both  # b + c
    inner = 1 + 3 + 3 * both - only_a_plus_only_b
    return np.log(np.clip(inner, _TONKIN_FLOOR, None))


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_matrix_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def day_gap_matrix(dates: np.ndarray) -> np.ndarray:
    days = (dates.astype("datetime64[D]").astype(np.int64))
    return np.abs(days[:, None] - days[None, :]).astype(np.float32)
