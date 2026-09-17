"""Load and harmonize the real India datasets:

1. NCRB district-wise Crime-Against-Women counts for 2001 (681 districts, 35
   states/UTs, 7 legal categories, with lat/lon) -- sourced from the Open
   Government Data Platform India (data.gov.in) mirror at
   github.com/namratapoojary/Analysis-and-Prediction-of-Crime-Against-Women.
2. NCRB national yearly totals per category, 2001-2012 -- same source.
3. Census 2011 district-level demographics (population, sex split, literacy,
   SC/ST, workforce, urban/rural households) -- 640 districts, sourced from
   github.com/nishusharma1608/India-Census-2011-Analysis.

District names are spelled differently across the two sources (different
transliteration/abbreviation conventions), so we join on a normalized
(state, district) key with a fallback fuzzy match.
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd

from src.config import CRIME_CATEGORIES, DATA_PROCESSED, DATA_RAW


_STATE_ALIASES = {
    "A & N ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
    "D & N HAVELI": "DADRA AND NAGAR HAVELI",
    "DAMAN & DIU": "DAMAN AND DIU",
    "DELHI": "NCT OF DELHI",
    "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
    "ODISHA": "ORISSA",
    "PUDUCHERRY": "PONDICHERRY",
}


def _normalize(name: str) -> str:
    name = str(name).upper().strip()
    name = re.sub(r"[.\-']", "", name)
    name = re.sub(r"\s+", " ", name)
    # common NCRB/Census abbreviation differences
    name = name.replace(" DIST", "").replace("DISTRICT", "").strip()
    name = _STATE_ALIASES.get(name, name)
    return name


def load_crime_2001() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "ncrb_district_crime_2001.csv")
    df = df.rename(columns={"STATE/UT": "state", "DISTRICT": "district"})
    for col in ["Latitude", "Longitude"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
        )
    state_norm = df["state"].map(_normalize)
    district_norm = df["district"].map(_normalize)
    df = pd.concat([df, state_norm.rename("state_norm"), district_norm.rename("district_norm")], axis=1)
    return df


def load_census_2011() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "census2011_districts.csv")
    df = df.rename(columns={"State name": "state", "District name": "district"})
    state_norm = df["state"].map(_normalize)
    district_norm = df["district"].map(_normalize)
    df = pd.concat([df, state_norm.rename("state_norm"), district_norm.rename("district_norm")], axis=1)
    return df


def load_national_yearly_totals() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "ncrb_yearly_national_totals.csv")
    df = df.rename(
        columns={
            "Kidnapping and Abduction": "Kidnapping_&_Abduction",
            "Dowry Deaths": "Dowry_Deaths",
            "Assault on women with intent to outrage her modesty": "Assault_on_women_with_intent_to_outrage_her_modesty",
            "Insult to modesty of Women": "Insult_to_modesty_of_Women",
            "Cruelty by Husband or his Relatives": "Cruelty_by_Husband_or_his_Relatives",
            "Importation of Girls": "Importation_of_Girls",
        }
    )
    return df


def _best_fuzzy_match(target: str, candidates: list[str]) -> str | None:
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=0.72)
    return matches[0] if matches else None


def harmonize() -> pd.DataFrame:
    crime = load_crime_2001()
    census = census_df = load_census_2011()

    census_by_state = {
        s: sub for s, sub in census_df.groupby("state_norm")
    }

    matched_rows = []
    for _, row in crime.iterrows():
        state_key = row["state_norm"]
        district_key = row["district_norm"]
        census_state = census_by_state.get(state_key)

        pop = female_pop = literate = sc = st = urban_hh = rural_hh = np.nan
        match_quality = "no_state"

        if census_state is not None:
            exact = census_state[census_state["district_norm"] == district_key]
            if len(exact) == 1:
                c = exact.iloc[0]
                match_quality = "exact"
            else:
                fuzzy_name = _best_fuzzy_match(district_key, census_state["district_norm"].tolist())
                if fuzzy_name is not None:
                    c = census_state[census_state["district_norm"] == fuzzy_name].iloc[0]
                    match_quality = "fuzzy"
                else:
                    c = None
                    match_quality = "unmatched"

            if match_quality in ("exact", "fuzzy"):
                pop = c["Population"]
                female_pop = c["Female"]
                literate = c["Literate"]
                sc = c["SC"]
                st = c["ST"]
                urban_hh = c["Urban_Households"]
                rural_hh = c["Rural_Households"]

        matched_rows.append(
            dict(
                population=pop,
                female_population=female_pop,
                literate=literate,
                sc_population=sc,
                st_population=st,
                urban_households=urban_hh,
                rural_households=rural_hh,
                census_match_quality=match_quality,
            )
        )

    census_cols = pd.DataFrame(matched_rows)
    merged = pd.concat([crime.reset_index(drop=True), census_cols], axis=1)

    # Impute missing population/demographics with the state-level mean so every
    # district has usable covariates (flagged via census_match_quality).
    for col in [
        "population",
        "female_population",
        "literate",
        "sc_population",
        "st_population",
        "urban_households",
        "rural_households",
    ]:
        state_mean = merged.groupby("state_norm")[col].transform("mean")
        overall_mean = merged[col].mean()
        merged[col] = merged[col].fillna(state_mean).fillna(overall_mean)

    # Impute missing lat/lon with the state centroid.
    for col in ["Latitude", "Longitude"]:
        state_mean = merged.groupby("state_norm")[col].transform("mean")
        merged[col] = merged[col].fillna(state_mean)

    merged["urban_share"] = merged["urban_households"] / (
        merged["urban_households"] + merged["rural_households"]
    ).replace(0, np.nan)
    merged["urban_share"] = merged["urban_share"].fillna(merged["urban_share"].mean())
    merged["literacy_rate"] = (merged["literate"] / merged["population"]).clip(0, 1)

    merged = merged.rename(columns={"Latitude": "lat", "Longitude": "lon"})
    merged["district_id"] = np.arange(len(merged))

    keep_cols = (
        ["district_id", "state", "district", "lat", "lon", "population", "female_population",
         "literacy_rate", "urban_share", "census_match_quality"]
        + CRIME_CATEGORIES
        + ["Total"]
    )
    merged = merged[keep_cols].rename(columns={"Total": "total_2001"})
    return merged


def main():
    merged = harmonize()
    out_path = DATA_PROCESSED / "districts_harmonized.csv"
    merged.to_csv(out_path, index=False)
    match_counts = merged["census_match_quality"].value_counts()
    print(f"Wrote {len(merged)} districts to {out_path}")
    print("Census match quality:\n", match_counts)


if __name__ == "__main__":
    main()
