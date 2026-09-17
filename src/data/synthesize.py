"""Synthetic district-month crime panel for India, grounded in real data:

Real anchors used
------------------
* District x category baseline incident counts for 2001 (real NCRB data,
  681 districts) -- `districts_harmonized.csv`.
* National year-on-year growth per category, 2001-2012 (real NCRB totals)
  -- extrapolated forward to SIM_END_YEAR with a damped growth rate once the
  real series runs out.
* District population / female population / urban share (real Census 2011)
  -- projected backward/forward with a constant annual growth rate.
* District spatial adjacency (k-NN graph over real lat/lon centroids).

Simulated (explicitly synthetic) layers, documented as modelling assumptions
------------------------------------------------------------------------
* Category-specific monthly seasonality curves (wedding-season effect for
  dowry/cruelty categories, summer-mobility effect for stranger-perpetrated
  categories).
* A district-level log-normal "trend dispersion" random effect so districts
  don't all move in perfect lockstep with the national trend.
* A spatially-smoothed annual random field (via the k-NN graph) so
  neighbouring districts co-move.
* A near-repeat / self-excitation term (Hawkes-style): a district (and its
  graph neighbours) that saw an elevated month is modestly more likely to see
  an elevated following month, motivated by the modus-operandi / near-repeat
  literature. This only *reallocates* probability mass across months, so it
  does not change the annual total.

The output is a fully reproducible (fixed seed) monthly panel:
district_id, state, district, year, month, date, <7 category counts>, total,
population, female_population, crime_rate_per_100k_women.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CRIME_CATEGORIES,
    DATA_PROCESSED,
    DATA_SYNTHETIC,
    EXTRAPOLATION_GROWTH_CAP,
    NEAR_REPEAT_DECAY_MONTHS,
    NEAR_REPEAT_NEIGHBOR_WEIGHT,
    NEAR_REPEAT_SELF_WEIGHT,
    POPULATION_ANNUAL_GROWTH,
    RANDOM_SEED,
    SEASONAL_WEIGHTS,
    SIM_END_YEAR,
    SIM_START_YEAR,
)
from src.data.spatial_graph import adjacency_matrix, load_graph


def _national_yearly_ratios() -> pd.DataFrame:
    """Real 2001-2012 national ratios per category (relative to 2001),
    extrapolated to SIM_END_YEAR with a damped average growth rate."""
    nat = pd.read_csv(DATA_PROCESSED.parent / "raw" / "ncrb_yearly_national_totals.csv")
    nat = nat.rename(
        columns={
            "Kidnapping and Abduction": "Kidnapping_&_Abduction",
            "Dowry Deaths": "Dowry_Deaths",
            "Assault on women with intent to outrage her modesty": "Assault_on_women_with_intent_to_outrage_her_modesty",
            "Insult to modesty of Women": "Insult_to_modesty_of_Women",
            "Cruelty by Husband or his Relatives": "Cruelty_by_Husband_or_his_Relatives",
            "Importation of Girls": "Importation_of_Girls",
        }
    ).set_index("Year")

    years = list(range(SIM_START_YEAR, SIM_END_YEAR + 1))
    ratios = pd.DataFrame(index=years, columns=CRIME_CATEGORIES, dtype=float)

    for cat in CRIME_CATEGORIES:
        base = nat.loc[SIM_START_YEAR, cat]
        last_real_year = nat.index.max()
        for y in years:
            if y <= last_real_year:
                ratios.loc[y, cat] = nat.loc[y, cat] / base
            else:
                # damped extrapolation: average YoY growth over the last 3
                # real years, capped so long horizons don't blow up.
                recent = nat.loc[last_real_year - 3 : last_real_year, cat]
                yoy = recent.pct_change().dropna()
                growth = float(np.clip(yoy.mean(), -EXTRAPOLATION_GROWTH_CAP, EXTRAPOLATION_GROWTH_CAP))
                prev_ratio = ratios.loc[y - 1, cat]
                ratios.loc[y, cat] = prev_ratio * (1 + growth)
    return ratios


def _spatial_smoothed_field(a_norm: np.ndarray, n_districts: int, n_years: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Independent log-normal noise per district-year, smoothed over the
    spatial graph (own value + neighbour average) so nearby districts
    co-move, as in the spatio-temporal crime hotspot literature."""
    raw = rng.normal(0, sigma, size=(n_years, n_districts))
    smoothed = 0.6 * raw + 0.4 * (raw @ a_norm.T)
    return np.exp(smoothed)


def generate(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    districts = pd.read_csv(DATA_PROCESSED / "districts_harmonized.csv")
    n = len(districts)
    graph = load_graph()
    adj = adjacency_matrix(graph)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    a_norm = adj / row_sums  # row-normalized: neighbour-average operator

    years = list(range(SIM_START_YEAR, SIM_END_YEAR + 1))
    n_years = len(years)
    ratios = _national_yearly_ratios()

    baseline = districts[CRIME_CATEGORIES].to_numpy(dtype=float)  # (n, 7), 2001 real counts
    baseline = np.clip(baseline, 0.1, None)  # avoid degenerate all-zero districts

    # District-level trend dispersion: fixed multiplicative heterogeneity
    # around the national trend, drawn once per district x category.
    trend_dispersion = rng.lognormal(mean=0.0, sigma=0.15, size=(n, len(CRIME_CATEGORIES)))
    trend_dispersion /= trend_dispersion.mean(axis=0, keepdims=True)  # keep national trend unbiased

    # Spatially-smoothed annual random field, shared across categories per year.
    spatial_field = _spatial_smoothed_field(a_norm, n, n_years, sigma=0.12, rng=rng)  # (n_years, n)
    spatial_field /= spatial_field.mean(axis=1, keepdims=True)  # keep national trend unbiased

    pop_2011 = districts["population"].to_numpy(dtype=float)
    female_share = (districts["female_population"] / districts["population"]).to_numpy(dtype=float)

    records = []
    prev_month_counts = np.zeros((n, len(CRIME_CATEGORIES)))
    decay = np.exp(-1.0 / NEAR_REPEAT_DECAY_MONTHS)

    for yi, year in enumerate(years):
        pop_factor = (1 + POPULATION_ANNUAL_GROWTH) ** (year - 2011)
        population_year = pop_2011 * pop_factor

        # Note: `ratio_row` already reflects the real national trend (which
        # itself embeds whatever population growth actually occurred), so
        # population growth is NOT reapplied here -- doing so would double
        # count it. `population_year` is tracked separately purely as a
        # covariate/denominator (e.g. crime rate per 100k women).
        ratio_row = ratios.loc[year].to_numpy(dtype=float)  # (7,)
        annual_intensity = (
            baseline
            * ratio_row[None, :]
            * trend_dispersion
            * spatial_field[yi][:, None]
        )
        annual_count = rng.poisson(annual_intensity).astype(float)  # (n, 7)

        for m in range(12):
            seasonal = np.array([SEASONAL_WEIGHTS[c][m] for c in CRIME_CATEGORIES])  # (7,)
            neighbor_prev = a_norm @ prev_month_counts  # (n, 7)
            excitation = 1.0 + NEAR_REPEAT_SELF_WEIGHT * decay * _row_normalize(prev_month_counts) \
                + NEAR_REPEAT_NEIGHBOR_WEIGHT * decay * _row_normalize(neighbor_prev)
            month_weight = seasonal[None, :] * excitation  # (n, 7)

            if m < 11:
                # Reallocate probability mass across the remaining months of
                # the year using a running multinomial split, so the annual
                # total is preserved exactly.
                remaining_weight_sum = _remaining_seasonal_sum(m)
                month_share = np.clip(month_weight / remaining_weight_sum, 1e-6, 1.0)
                month_count = rng.binomial(annual_count.astype(int).clip(min=0), np.clip(month_share, 0, 1))
                month_count = np.minimum(month_count, annual_count)
                annual_count = annual_count - month_count
            else:
                month_count = np.clip(annual_count, 0, None)

            prev_month_counts = month_count.astype(float)

            date = pd.Timestamp(year=year, month=m + 1, day=1)
            for di in range(n):
                row = {
                    "district_id": int(districts.iloc[di]["district_id"]),
                    "state": districts.iloc[di]["state"],
                    "district": districts.iloc[di]["district"],
                    "lat": districts.iloc[di]["lat"],
                    "lon": districts.iloc[di]["lon"],
                    "year": year,
                    "month": m + 1,
                    "date": date,
                    "population": population_year[di],
                    "female_population": population_year[di] * female_share[di],
                }
                for ci, cat in enumerate(CRIME_CATEGORIES):
                    row[cat] = int(month_count[di, ci])
                row["total"] = int(month_count[di].sum())
                records.append(row)

    panel = pd.DataFrame.from_records(records)
    panel["crime_rate_per_100k_women"] = (
        panel["total"] / panel["female_population"].clip(lower=1) * 100_000
    )
    return panel


def _row_normalize(x: np.ndarray) -> np.ndarray:
    col_max = x.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    return x / col_max


def _remaining_seasonal_sum(current_month_idx: int) -> np.ndarray:
    sums = []
    for cat in CRIME_CATEGORIES:
        weights = SEASONAL_WEIGHTS[cat][current_month_idx:]
        sums.append(sum(weights))
    return np.array(sums)


def main():
    panel = generate()
    out_csv = DATA_SYNTHETIC / "district_month_panel.csv"
    out_parquet = DATA_SYNTHETIC / "district_month_panel.parquet"
    panel.to_csv(out_csv, index=False)
    try:
        panel.to_parquet(out_parquet, index=False)
    except Exception as e:
        print(f"(parquet export skipped: {e})")
    print(f"Wrote {len(panel):,} rows ({panel['district_id'].nunique()} districts x "
          f"{panel['year'].nunique()} years x 12 months) to:\n  {out_csv}")
    print(panel.groupby("year")["total"].sum())


if __name__ == "__main__":
    main()
