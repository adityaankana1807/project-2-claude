"""Synthetic incident-level records with modus-operandi (MO) flags, for
demonstrating the crime-LINKAGE half of the project (Layer 3) -- distinct
from the district-month aggregate panel used for forecasting, since linking
individual crimes to a common serial offender requires incident-level
records, which NCRB does not publish openly (real FIR free text is
restricted/sensitive). This generator creates a labelled ground truth
(series membership) so linkage algorithms can be objectively evaluated,
following the design of the real studies reviewed (Woodhams & Labuschagne
2012; Tonkin et al. 2025): serial-offender series share a noisy MO
"template", behaviours have realistically low base rates, and a pool of
singleton (unlinked) crimes acts as distractors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DATA_SYNTHETIC, RANDOM_SEED

N_MO_BEHAVIOURS = 30
N_SERIES = 40
SERIES_LEN_MEAN = 4.5
N_SINGLETONS = 400
WITHIN_SERIES_NOISE = 0.12   # prob. a template bit is flipped for a given crime
MARAUDER_JITTER_DEG = 0.05   # ~5km jitter around the series' home district


@dataclass
class IncidentSet:
    incidents: pd.DataFrame          # one row per incident
    mo_matrix: np.ndarray            # (n_incidents, N_MO_BEHAVIOURS) binary


def generate(seed: int = RANDOM_SEED) -> IncidentSet:
    rng = np.random.default_rng(seed)
    districts = pd.read_csv(DATA_PROCESSED / "districts_harmonized.csv")

    # Behaviour base rates: mostly rare (mirrors the reviewed literature's
    # finding that most MO variables are present in well under 10% of crimes).
    base_rates = rng.beta(0.6, 5.0, size=N_MO_BEHAVIOURS)

    rows = []
    mo_rows = []
    incident_id = 0

    for series_id in range(N_SERIES):
        series_len = max(2, int(rng.poisson(SERIES_LEN_MEAN)))
        home = districts.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        template = (rng.random(N_MO_BEHAVIOURS) < base_rates).astype(int)
        start_date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))
        cur_date = start_date

        for _ in range(series_len):
            flips = rng.random(N_MO_BEHAVIOURS) < WITHIN_SERIES_NOISE
            mo = np.where(flips, 1 - template, template)
            lat = home["lat"] + rng.normal(0, MARAUDER_JITTER_DEG)
            lon = home["lon"] + rng.normal(0, MARAUDER_JITTER_DEG)
            rows.append(dict(
                incident_id=incident_id, series_id=series_id, is_singleton=False,
                district=home["district"], state=home["state"], lat=lat, lon=lon,
                date=cur_date,
            ))
            mo_rows.append(mo)
            incident_id += 1
            cur_date = cur_date + pd.Timedelta(days=int(rng.exponential(25)) + 1)

    for _ in range(N_SINGLETONS):
        d = districts.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        mo = (rng.random(N_MO_BEHAVIOURS) < base_rates).astype(int)
        date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 3400)))
        rows.append(dict(
            incident_id=incident_id, series_id=-1, is_singleton=True,
            district=d["district"], state=d["state"], lat=d["lat"], lon=d["lon"],
            date=date,
        ))
        mo_rows.append(mo)
        incident_id += 1

    incidents = pd.DataFrame(rows)
    mo_matrix = np.stack(mo_rows, axis=0)
    return IncidentSet(incidents=incidents, mo_matrix=mo_matrix)


def main():
    data = generate()
    out_dir = DATA_SYNTHETIC / "linkage"
    out_dir.mkdir(exist_ok=True)
    data.incidents.to_csv(out_dir / "incidents.csv", index=False)
    np.save(out_dir / "mo_matrix.npy", data.mo_matrix)
    n_linked_pairs = sum(
        len(g) * (len(g) - 1) // 2 for _, g in data.incidents[~data.incidents.is_singleton].groupby("series_id")
    )
    print(f"{len(data.incidents)} incidents ({data.incidents.series_id.nunique() - 1} series, "
          f"{data.incidents.is_singleton.sum()} singletons), {n_linked_pairs} true linked pairs")


if __name__ == "__main__":
    main()
