"""
Training Set (2018–2021)

Builds the model training matrix using only years where BOTH fire labels
AND weather features are fully available (2018–2021).

Depends on: data/fires_geocoded.csv  (output of geocode_zip_recovery.py)

Outputs (saved to ./data/):
  train_features.csv  — feature matrix, one row per zip x year (2018–2021)
  train_labels.csv    — binary wildfire label per zip x year
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Config

RAW_PATH      = "data/wildfire_weather.csv"
FIRES_PATH    = Path("data/fires_geocoded.csv")
OUTPUT_DIR    = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_YEARS   = [2018, 2019, 2020, 2021]

DRY_MONTHS    = [6, 7, 8, 9, 10]        # Jun–Oct  (fire season)
WET_MONTHS    = [11, 12, 1, 2, 3, 4, 5] # Nov–May  (wet/fuel-growth season)


# Load sources

print("Loading data...")
df        = pd.read_csv(RAW_PATH, low_memory=False)
wildfires = pd.read_csv(FIRES_PATH)

wildfires["zip"]  = wildfires["zip"].astype(int)
wildfires["year"] = wildfires["year"].astype(int)

weather_raw = df[df["OBJECTID"].isna()].copy()
all_zips    = sorted(weather_raw["zip"].dropna().unique().astype(int))

print(f"  Wildfire records     : {len(wildfires):,}")
print(f"  Unique zips (weather): {len(all_zips):,}")


# Build binary labels (zip x year, training years only) 

zip_year_index = pd.MultiIndex.from_product(
    [all_zips, TRAIN_YEARS], names=["zip", "year"]
)
labels = pd.DataFrame(index=zip_year_index).reset_index()

fire_flags = (
    wildfires[wildfires["year"].isin(TRAIN_YEARS)]
    .groupby(["zip", "year"])
    .size()
    .reset_index(name="fire_count")
    .assign(wildfire=1)
)

labels = labels.merge(
    fire_flags[["zip", "year", "wildfire"]],
    on=["zip", "year"],
    how="left",
)
labels["wildfire"] = labels["wildfire"].fillna(0).astype(int)

print(f"\nLabel matrix: {labels.shape[0]:,} rows")
print("  Positive labels per year:")
print(labels.groupby("year")["wildfire"].sum().to_string())
print(f"  Overall positive rate: {labels['wildfire'].mean():.3%}")


# Build weather features (zip x year) 
#
# All training years (2018–2021) have complete monthly weather rows,
# so there will be no NaN gaps in this split.

weather = weather_raw[
    ["zip", "year_month", "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"]
].copy()
weather = weather[weather["zip"].notna()].copy()
weather["zip"]   = weather["zip"].astype(int)
weather["year"]  = weather["year_month"].str[:4].astype(int)
weather["month"] = weather["year_month"].str[5:].astype(int)

# Restrict to training years
weather = weather[weather["year"].isin(TRAIN_YEARS)]

def agg_season(df, months, suffix):
    """Aggregate monthly rows into seasonal summaries per zip x year."""
    sub = df[df["month"].isin(months)]
    return (
        sub.groupby(["zip", "year"])
        .agg(
            **{
                f"tmax_{suffix}":       ("avg_tmax_c",  "mean"),
                f"tmin_{suffix}":       ("avg_tmin_c",  "mean"),
                f"prcp_{suffix}":       ("tot_prcp_mm", "sum"),
                # drought proxy: count of months with < 5 mm precipitation
                f"dry_months_{suffix}": ("tot_prcp_mm", lambda x: (x < 5).sum()),
            }
        )
        .reset_index()
    )

dry_feats    = agg_season(weather, DRY_MONTHS, "dry")
wet_feats    = agg_season(weather, WET_MONTHS, "wet")

annual_feats = (
    weather.groupby(["zip", "year"])
    .agg(
        tmax_annual=("avg_tmax_c",  "mean"),
        tmin_annual=("avg_tmin_c",  "mean"),
        prcp_annual=("tot_prcp_mm", "sum"),
        tmax_peak  =("avg_tmax_c",  "max"),  # hottest single month
        prcp_min   =("tot_prcp_mm", "min"),  # driest single month
    )
    .reset_index()
)

weather_feats = (
    annual_feats
    .merge(dry_feats, on=["zip", "year"], how="inner")
    .merge(wet_feats, on=["zip", "year"], how="inner")
)

print(f"\nWeather feature matrix: {weather_feats.shape}")

# Sanity check — no NaNs expected for 2018–2021
weather_nan = weather_feats.isnull().sum().sum()
print(f"  NaN values in weather features: {weather_nan}  (expected 0)")


# Build fire history features (zip x year, lag-safe)
#
# Only uses data from PRIOR years — no leakage into the label year.
# Uses the full geocoded fire table (all years) for accurate lookback
# windows at the edges of the training range.

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
    for y in TRAIN_YEARS:
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

        years_since = 10  # cap: no known fire in the lookback window
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
print(f"\nFire history features: {fire_hist.shape}")


# Join into final training matrix

train = (
    labels
    .merge(weather_feats, on=["zip", "year"], how="inner")  # inner: weather is complete for 2018–2021
    .merge(fire_hist,     on=["zip", "year"], how="left")
)

print(f"\nFinal training matrix: {train.shape}")

total_nan = train.drop(columns=["zip", "year", "wildfire"]).isnull().sum().sum()
print(f"  Total NaN values in features: {total_nan}  (expected 0)")

feature_cols = [c for c in train.columns if c not in ["zip", "year", "wildfire"]]
print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")


# Save

train[feature_cols].to_csv(OUTPUT_DIR / "train_features.csv", index=False)
train[["zip", "year", "wildfire"]].to_csv(OUTPUT_DIR / "train_labels.csv", index=False)

print(f"\nSaved to {OUTPUT_DIR}/")
print(f"  train_features.csv : {train.shape[0]:,} rows x {len(feature_cols)} cols")
print(f"  train_labels.csv   : {train.shape[0]:,} rows")
print(f"  Positive rate      : {train['wildfire'].mean():.3%}")
