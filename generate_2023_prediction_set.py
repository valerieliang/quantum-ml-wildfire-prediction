"""
Builds the feature matrix the trained model will score to produce the final
2023 wildfire risk predictions required by the challenge submission.

Unlike the training (2018–2021) and validation (2022) sets, there are NO
labels here — 2023 is the target year the model is trying to predict.

Feature availability for 2023
-------------------------------
  Weather features      : NO  — source dataset ends at December 2021.
                                Estimated from 2018–2021 per-zip means,
                                same imputation strategy used for 2022.
  Fire history features : YES — computed from prior years only (lag-safe).
                                fires_last1  uses 2022 incidents
                                fires_last3  uses 2020, 2021, 2022 incidents
                                acres_last3  uses 2020, 2021, 2022 acreage
                                years_since  searches back to 2013

Schema
------
  17 columns — identical order to train_features.csv and val_features.csv.
  The model trained on train_features.csv can be applied directly.

Depends on:
  wildfire_weather.csv       (raw source — defines the zip universe)
  fires_geocoded.csv         (wildfire table with recovered zip codes)
  train_features.csv         (used to compute per-zip weather means)
  train_labels.csv           (used to join zip metadata for clim. means)

Output:
  predict_2023_features.csv  — 2,593 rows x 17 feature cols (one row per CA zip)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Config

RAW_PATH        = "data/wildfire_weather.csv"
FIRES_PATH      = "data/fires_geocoded.csv"
TRAIN_FEAT_PATH = "data/train_features.csv"
TRAIN_LAB_PATH  = "data/train_labels.csv"
OUTPUT_DIR      = Path("data/")
OUTPUT_DIR.mkdir(exist_ok=True)

PRED_YEAR = 2023

# Must match train_features.csv and val_features.csv exactly
WEATHER_COLS = [
    "tmax_annual", "tmin_annual", "prcp_annual", "tmax_peak", "prcp_min",
    "tmax_dry",    "tmin_dry",    "prcp_dry",    "dry_months_dry",
    "tmax_wet",    "tmin_wet",    "prcp_wet",    "dry_months_wet",
]
FIRE_COLS = ["fires_last1", "fires_last3", "acres_last3", "years_since_fire"]
FEATURE_COLS = WEATHER_COLS + FIRE_COLS   # 17 columns total


# Load sources

print("Loading data...")
ww          = pd.read_csv(RAW_PATH, low_memory=False)
wildfires   = pd.read_csv(FIRES_PATH)
train_feats = pd.read_csv(TRAIN_FEAT_PATH)
train_labs  = pd.read_csv(TRAIN_LAB_PATH)

wildfires["zip"]  = wildfires["zip"].astype(int)
wildfires["year"] = wildfires["year"].astype(int)

# Zip universe — all CA zip codes present in the weather table
weather_raw = ww[ww["OBJECTID"].isna()].copy()
all_zips    = sorted(weather_raw["zip"].dropna().unique().astype(int))

print(f"  Wildfire records  : {len(wildfires):,}")
print(f"  CA zip codes      : {len(all_zips):,}")
print(f"  2023 fire records : {(wildfires['year'] == PRED_YEAR).sum()}")


# Estimate 2023 weather from 2018–2021 climatological means 
#
# No weather observations exist for 2023 in the source dataset.
# Each zip code's weather features are set to its own 4-year average (2018–2021).
# This is the same imputation strategy applied to the 2022 validation set,
# keeping the feature space consistent across validation and prediction.
#
# Reliability note:
#   Temperature features (tmax_*, tmin_*) : stable year-to-year, reliable
#   Precipitation features (prcp_*)        : noisier, ~75–210mm std
#   Drought proxies (dry_months_*)         : moderate, ~1–2 months std

print("\nEstimating 2023 weather from 2018–2021 per-zip means...")
train_full = pd.concat([train_labs[["zip", "year"]], train_feats], axis=1)

clim_means = (
    train_full
    .groupby("zip")[WEATHER_COLS]
    .mean()
    .reset_index()
)

assert len(clim_means) == len(all_zips), \
    f"Climatological means cover {len(clim_means)} zips, expected {len(all_zips)}"
assert clim_means[WEATHER_COLS].isnull().sum().sum() == 0, \
    "Unexpected nulls in climatological means"

print(f"  Zips with weather estimates : {len(clim_means):,}")
print(f"  Null values                 : 0 (confirmed)")


# Build fire history features for 2023 
#
# Computed from prior years only — no leakage.
#
#   fires_last1       : wildfire count in 2022
#   fires_last3       : wildfire count over 2020, 2021, 2022
#   acres_last3       : total acres burned over 2020, 2021, 2022
#   years_since_fire  : years since most recent fire before 2023 (capped at 10)
#
# These are actual observed values (not estimates), making them the highest-
# quality signal available for 2023 prediction.

print("\nBuilding 2023 fire history features...")
fire_by_zip_year = (
    wildfires
    .groupby(["zip", "year"])
    .agg(n_fires=("GIS_ACRES", "count"), total_acres=("GIS_ACRES", "sum"))
    .reset_index()
)

y = PRED_YEAR
rows = []
for z in all_zips:
    zf = (
        fire_by_zip_year[fire_by_zip_year["zip"] == z]
        .set_index("year")
    )

    fires_last1 = int(zf.loc[y - 1, "n_fires"])      if (y - 1) in zf.index else 0
    fires_last3 = sum(
        int(zf.loc[yr, "n_fires"]) if yr in zf.index else 0
        for yr in [y - 3, y - 2, y - 1]
    )
    acres_last3 = sum(
        float(zf.loc[yr, "total_acres"]) if yr in zf.index else 0.0
        for yr in [y - 3, y - 2, y - 1]
    )
    years_since = next(
        (lag for lag in range(1, 11) if (y - lag) in zf.index),
        10
    )

    rows.append({
        "zip":              z,
        "fires_last1":      fires_last1,
        "fires_last3":      fires_last3,
        "acres_last3":      round(acres_last3, 2),
        "years_since_fire": years_since,
    })

fire_hist = pd.DataFrame(rows)
print(f"  Zips with fire history : {len(fire_hist):,}")
print(f"  Zips with fires_last1 > 0 : {(fire_hist['fires_last1'] > 0).sum()}")
print(f"  Zips with fires_last3 > 0 : {(fire_hist['fires_last3'] > 0).sum()}")


# Join weather estimates and fire history 

# Start from the full zip universe — every CA zip gets a prediction row
pred = (
    pd.DataFrame({"zip": all_zips, "year": PRED_YEAR})
    .merge(clim_means[["zip"] + WEATHER_COLS], on="zip", how="left")
    .merge(fire_hist[["zip"]  + FIRE_COLS],    on="zip", how="left")
)

assert len(pred) == len(all_zips), "Row count changed after join"
assert list(pred[FEATURE_COLS].columns) == FEATURE_COLS, \
    "Feature column order does not match expected schema"

total_nulls = pred[FEATURE_COLS].isnull().sum().sum()
assert total_nulls == 0, f"Unexpected nulls after join: {total_nulls}"

print(f"\nFinal prediction matrix : {pred.shape[0]:,} rows x {len(FEATURE_COLS)} features")
print(f"  Null values           : 0 (confirmed)")
print(f"  Feature columns       : {FEATURE_COLS}")


# Save 

out_path = OUTPUT_DIR / "predict_2023_features.csv"
pred[["zip", "year"] + FEATURE_COLS].to_csv(out_path, index=False)

print(f"\nSaved -> {out_path}")
print(f"  {pred.shape[0]:,} rows  (one per California zip code)")
print(f"  {len(FEATURE_COLS)} feature columns (13 weather estimated + 4 fire history)")
print(f"  zip and year columns included for traceability")