"""
Phase 1 — Validation Set (2022)
2026 Quantum Sustainability Challenge: Wildfire Risk Modeling

Builds the 2022 validation matrix with the same 17-feature schema as the
training set (2018-2021).

IMPORTANT — ESTIMATED WEATHER DATA:
  The source dataset contains no weather rows for 2022 (coverage ends 2021-12).
  The 13 weather features for 2022 are therefore ESTIMATED using the per-zip
  climatological mean across 2018-2021. Temperature means are highly stable
  (year-to-year std ~0.5-0.9 C) so these estimates are reliable. Precipitation
  is noisier (~119mm std) but still far better than leaving the columns empty.

  This imputation is used ONLY to test model output and validate that the
  pipeline produces sensible predictions before applying it to 2023.
  Do not treat 2022 validation metrics as ground truth — they reflect a model
  evaluated on estimated weather inputs, not measured ones.

Depends on:
  wildfire_weather.csv       (raw data)
  data/fires_geocoded.csv    (output of geocode_zip_recovery.py)
  data/train_features.csv    (output of phase1_train.py)
  data/train_labels.csv      (output of phase1_train.py)

Outputs (saved to ./data/):
  val_features.csv   — 17-feature matrix, one row per zip (2022)
  val_labels.csv     — binary wildfire label per zip (2022)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Config 

RAW_PATH        = "data/wildfire_weather.csv"
FIRES_PATH      = Path("data/fires_geocoded.csv")
TRAIN_FEAT_PATH = Path("data/train_features.csv")
TRAIN_LAB_PATH  = Path("data/train_labels.csv")
OUTPUT_DIR      = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

VAL_YEAR = 2022

WEATHER_COLS = [
    "tmax_annual", "tmin_annual", "prcp_annual", "tmax_peak", "prcp_min",
    "tmax_dry",    "tmin_dry",    "prcp_dry",    "dry_months_dry",
    "tmax_wet",    "tmin_wet",    "prcp_wet",    "dry_months_wet",
]
FIRE_COLS = ["fires_last1", "fires_last3", "acres_last3", "years_since_fire"]


# Load sources 

print("Loading data...")
df          = pd.read_csv(RAW_PATH, low_memory=False)
wildfires   = pd.read_csv(FIRES_PATH)
train_feats = pd.read_csv(TRAIN_FEAT_PATH)
train_labs  = pd.read_csv(TRAIN_LAB_PATH)

wildfires["zip"]  = wildfires["zip"].astype(int)
wildfires["year"] = wildfires["year"].astype(int)

weather_raw = df[df["OBJECTID"].isna()].copy()
all_zips    = sorted(weather_raw["zip"].dropna().unique().astype(int))

print(f"  Wildfire records : {len(wildfires):,}")
print(f"  Unique zips      : {len(all_zips):,}")


# Step 2: Estimate 2022 weather via per-zip climatological means 
#
# ESTIMATED DATA — see module docstring for full explanation.
# Each zip's 13 weather features are set to the mean value of that feature
# across 2018-2021. This is a reasonable approximation because:
#   - Temperature features are stable year-to-year (std ~0.5-0.9 C)
#   - Precipitation is noisier but captures the baseline dryness of each zip
#   - All 2,593 zips have complete coverage in the training years (no NaNs)

print("\nEstimating 2022 weather from 2018-2021 climatological means...")
train_full = pd.concat([train_labs[["zip", "year"]], train_feats], axis=1)

clim_means = (
    train_full
    .groupby("zip")[WEATHER_COLS]
    .mean()
    .reset_index()
)
clim_means["year"] = VAL_YEAR  # tag as 2022 for clarity

print(f"  Zips with estimated weather : {len(clim_means):,}  (expected {len(all_zips):,})")
print(f"  NaNs in estimates           : {clim_means[WEATHER_COLS].isnull().sum().sum()}  (expected 0)")

# Show how stable the estimates are (lower std = more trustworthy)
per_zip_std = train_full.groupby("zip")[WEATHER_COLS].std().mean()
print("\n  Year-to-year std per feature (lower = more reliable estimate):")
for col, val in per_zip_std.items():
    flag = "  <-- noisy, interpret with caution" if val > 50 else ""
    print(f"    {col:<22} {val:.3f}{flag}")


# Build binary labels for 2022 

labels_2022 = pd.DataFrame({"zip": all_zips, "year": VAL_YEAR, "wildfire": 0})

fire_flags = (
    wildfires[wildfires["year"] == VAL_YEAR]
    .groupby("zip")
    .size()
    .reset_index(name="fire_count")
    .assign(wildfire=1)
)

labels_2022 = labels_2022.merge(
    fire_flags[["zip", "wildfire"]],
    on="zip",
    how="left",
    suffixes=("", "_flag"),
)
labels_2022["wildfire"] = labels_2022["wildfire_flag"].fillna(0).astype(int)
labels_2022 = labels_2022.drop(columns="wildfire_flag")

print(f"\n2022 label matrix: {len(labels_2022):,} rows")
print(f"  Positive (fire) zips : {labels_2022['wildfire'].sum()}")
print(f"  Positive rate        : {labels_2022['wildfire'].mean():.3%}")


# Build fire history features (lag-safe, prior years only) 

fire_by_zip_year = (
    wildfires
    .groupby(["zip", "year"])
    .agg(n_fires=("GIS_ACRES", "count"), total_acres=("GIS_ACRES", "sum"))
    .reset_index()
)

rows = []
for z in all_zips:
    zip_hist = (
        fire_by_zip_year[fire_by_zip_year["zip"] == z]
        .set_index("year")
    )
    y = VAL_YEAR

    fires_last1 = int(zip_hist.loc[y - 1, "n_fires"]) \
                  if (y - 1) in zip_hist.index else 0

    prior3      = [y - 3, y - 2, y - 1]
    fires_last3 = sum(
        int(zip_hist.loc[yr, "n_fires"])       if yr in zip_hist.index else 0
        for yr in prior3
    )
    acres_last3 = sum(
        float(zip_hist.loc[yr, "total_acres"]) if yr in zip_hist.index else 0.0
        for yr in prior3
    )

    years_since = 10
    for lookback in range(1, 11):
        if (y - lookback) in zip_hist.index:
            years_since = lookback
            break

    rows.append({
        "zip": z, "year": y,
        "fires_last1":      fires_last1,
        "fires_last3":      fires_last3,
        "acres_last3":      acres_last3,
        "years_since_fire": years_since,
    })

fire_hist = pd.DataFrame(rows)
print(f"\nFire history features (actual lagged data): {fire_hist.shape}")


# Join labels + estimated weather + fire history 

val = (
    labels_2022
    .merge(clim_means[["zip"] + WEATHER_COLS], on="zip", how="left")
    .merge(fire_hist[["zip"] + FIRE_COLS],     on="zip", how="left")
)

# Final schema check — must match training feature order exactly
feature_cols = WEATHER_COLS + FIRE_COLS
assert list(val[feature_cols].columns) == feature_cols, \
    "Feature column order does not match training schema — check joins"

print(f"\nFinal validation matrix : {val.shape}")
print(f"  Total NaNs in features: {val[feature_cols].isnull().sum().sum()}  (expected 0)")
print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")


# Save 

val[feature_cols].to_csv(OUTPUT_DIR / "val_features.csv",  index=False)
val[["zip", "year", "wildfire"]].to_csv(OUTPUT_DIR / "val_labels.csv", index=False)

print(f"\nSaved to {OUTPUT_DIR}/")
print(f"  val_features.csv : {val.shape[0]:,} rows x {len(feature_cols)} cols")
print(f"  val_labels.csv   : {val.shape[0]:,} rows")
print(f"  Positive rate    : {val['wildfire'].mean():.3%}")
# 2022 weather features are estimated from 2018-2021 means. Validation metrics should be interpreted as approximate benchmarks only.