# ST-SAGE: Spatio-Temporal Small-Area Graph Estimator for India Crime Forecasting & Linkage

A novel hybrid algorithm for **crime-against-women forecasting and serial-offender linkage
in India**, developed by synthesizing 18 research papers on crime prediction, small-area
estimation, and case linkage, then grounded in **real Indian government data**:

- Real district-level NCRB "Crime Against Women" counts for 2001 (681 districts, 35
  states/UTs, 7 legal categories, with coordinates) — sourced from the Open Government
  Data Platform India (data.gov.in).
- Real national year-on-year totals per category, 2001–2012 (NCRB).
- Real Census 2011 district demographics (population, sex ratio, literacy, SC/ST,
  urban/rural households) — 640 districts.

Because NCRB does not publish granular monthly/incident-level open data, this project
**synthesizes** a full monthly district-level panel (2001–2024) and a labelled
incident-level linkage dataset, both explicitly and transparently derived from the real
anchors above plus documented modelling assumptions (seasonality, near-repeat effects,
spatial autocorrelation). Every synthetic assumption is called out in code comments and
below — nothing is presented as real data that isn't.

## What's actually novel here

No single reviewed paper combines all of this; ST-SAGE is a new synthesis:

1. **Small-Area Estimation (Layer 1)** — a Marshall (1991) empirical-Bayes shrinkage
   estimator, extended to borrow strength from a *spatial* k-NN graph of real district
   centroids rather than just the national mean (`src/models/sae_smoothing.py`), computed
   **causally** (only past months) so it's safe to use as a forecasting feature. This
   directly operationalizes the "Crime Against Women in India: District-Level Risk
   Estimation Using Small Area Estimation" paper's approach.
2. **Graph-based ST-ResNet/LSTM hybrid (Layer 2)** — replaces the raster-CNN spatial
   backbone used in most crime-forecasting deep learning papers with a **graph
   convolution** over real district adjacency (since India's forecasting unit is an
   irregular administrative polygon, not a uniform pixel grid), keeps the
   closeness/period/trend multi-branch temporal decomposition from
   "Hybrid ST-ResNet and LSTM approach for precise crime hotspot prediction", adds a GRU
   for short-range dynamics, and — new — a **learned near-repeat/self-excitation gate**
   over the graph (own + neighbour last-month counts), inspired by the modus-operandi /
   near-repeat burglary literature. The output head is an **exposure-offset
   Negative-Binomial regression** (population as the offset, per-category learned
   dispersion), so Layer 1's population-normalization idea is built into the loss itself,
   not just a preprocessing step.
3. **MO-similarity linkage engine (Layer 3)** — implements and head-to-head compares
   Jaccard's coefficient against the Tonkin et al. (2025, UK ViCLAS benchmark)
   low-base-rate-robust similarity metric `log(1 + 3 + 3a - (b+c))`, combined with
   inter-crime distance and temporal proximity (the two most crime-type-agnostic linkage
   features across the whole literature) via logistic regression, evaluated with the
   decision-relevant metrics that benchmark actually uses (top-100/top-500 hit rate,
   Median First Rank) rather than AUC alone.

See `src/linkage/run_linkage_demo.py` output / the dashboard's Linkage tab for an honest
discussion of where our synthetic data does **not** reproduce the real paper's finding
(Jaccard slightly beats the new metric here) — documented rather than hidden.

## Real data sources

| File | Source | What it is |
|---|---|---|
| `data/raw/ncrb_district_crime_2001.csv` | github.com/namratapoojary/Analysis-and-Prediction-of-Crime-Against-Women (mirrors data.gov.in / NCRB) | Real district-level crime-against-women counts, 2001 |
| `data/raw/ncrb_yearly_national_totals.csv` | same source | Real national yearly totals per category, 2001–2012 |
| `data/raw/census2011_districts.csv` | github.com/nishusharma1608/India-Census-2011-Analysis | Real Census 2011 district demographics |

`src/data/harmonize.py` joins these two on normalized (state, district) with a state-name
alias table + fuzzy fallback; ~81% match exactly/fuzzily, the rest are imputed with the
state mean and flagged via `census_match_quality`.

## Synthetic data (explicitly documented assumptions)

`src/data/synthesize.py` builds `data/synthetic/district_month_panel.csv`
(196,128 rows = 681 districts × 24 years × 12 months) by:

- Anchoring every district-category series to its **real 2001 count**.
- Scaling forward by the **real 2001–2012 national trend** per category (extrapolated
  beyond 2012 with a damped, capped growth rate).
- Adding a **district-level log-normal heterogeneity** term and a **spatially-smoothed
  annual random field** (via the k-NN graph) so neighbouring districts co-move — both
  mean-centered so they don't bias the calibrated national trend.
- Disaggregating annual counts into months via **documented seasonal weight curves**
  (a wedding/festive-season effect for dowry/cruelty categories, a summer-mobility effect
  for stranger-perpetrated categories) and a **near-repeat self-excitation** term that
  reallocates (not adds) probability mass across months based on the district's and its
  graph-neighbours' previous month.

`src/linkage/synthetic_incidents.py` builds a labelled incident-level dataset (real
district locations, synthetic dates/MO) for the linkage demo, since real FIR text is
restricted/sensitive: serial-offender "series" share a noisy MO template with low base
rates, plus singleton distractor crimes, so linkage accuracy can be objectively measured
against ground truth.

## Project layout

```
src/
  config.py                 constants (categories, seasonal weights, lags, seeds)
  data/
    harmonize.py             real NCRB + Census join
    spatial_graph.py         k-NN district adjacency graph
    synthesize.py            synthetic district-month panel generator
  models/
    sae_smoothing.py          Layer 1: causal empirical-Bayes spatial smoothing
    dataset.py                 tensor/window construction for training
    st_sage.py                 the ST-SAGE PyTorch model
    train.py / evaluate.py     training loop / held-out evaluation
  linkage/
    synthetic_incidents.py     Layer 3 synthetic incident generator
    similarity.py               Jaccard, Tonkin et al. (2025) metric, distance/time
    run_linkage_demo.py         end-to-end linkage evaluation
app/
  dashboard.py                 Streamlit dashboard (forecast map, drilldown, linkage, about)
scripts/
  run_pipeline.py               one-command end-to-end run
```

## Running it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_pipeline.py
.\.venv\Scripts\streamlit.exe run app\dashboard.py
```

or just `./run.ps1` to do both.

## Results (current run)

- Forecasting (held-out 2023–2024 months): precision@10 hotspot districts ≈ 0.53,
  precision@50 ≈ 0.73 — see `outputs/eval_metrics.json` for full per-category MAE/RMSE.
- Linkage: combined distance+time+MO model AUC ≈ 0.9998, Median First Rank = 1, on the
  synthetic incident set — see `outputs/linkage_metrics.json`.

## Known limitations

- NCRB only publishes *annual* district totals, so the monthly panel is synthetic —
  useful for developing/validating the algorithm architecture, not as a substitute for
  real monthly police data.
- ~19% of districts fall back to state-mean imputation for census covariates due to
  district-name spelling mismatches between the two source datasets.
- The linkage demo's "marauder-only" jitter model makes geography an unrealistically
  strong signal; a real deployment would need actual offender typology diversity
  (forager/marauder/commuter, per Halford 2023) to stress-test the MO-similarity metrics
  properly.
- Extrapolated years (2013–2024) rely on a capped average growth rate, not further real
  anchors — treat long-horizon forecasts as illustrative.
