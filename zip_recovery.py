"""
Zip Code Recovery via KD-Tree Spatial Lookup
2026 Quantum Sustainability Challenge: Wildfire Risk Modeling

337 fire incident rows are missing zip codes but all have latitude/longitude.
These account for ~36% of total burned acreage so recovery is worth doing.

Approach:
  - Build a reference table of (zip, lat, lon) centroids from fire rows that
    already have all three fields populated (1,881 rows across 559 unique zips)
  - Use a KD-tree to find the nearest known zip centroid for each missing row
  - Accept the match if the nearest centroid is within MAX_DIST_DEG (~33 km)
  - Drop the remaining rows that fall outside the threshold (remote federal land
    with no nearby zip reference; better to discard than mislabel)

Output:
  data/fires_geocoded.csv  — full fire table with zip gaps filled where possible
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

# Config

RAW_PATH      = "wildfire_weather.csv"
OUTPUT_DIR    = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Accept nearest-neighbour match only if within this distance (degrees)
# 0.3 deg ≈ 33 km
MAX_DIST_DEG  = 0.3

# Same wildfire filter as Phase 1
EXCLUDE_CAUSE     = 18
EXCLUDE_OBJECTIVE = 2


# 1: Load and split

print("Loading raw data...")
df = pd.read_csv(RAW_PATH, low_memory=False)

fire_raw = df[df["OBJECTID"].notna()].copy()
print(f"  Total fire rows: {len(fire_raw):,}")


# 2: Build zip centroids from rows that already have zip + coords

reference = fire_raw[
    fire_raw["zip"].notna() &
    fire_raw["latitude"].notna() &
    fire_raw["longitude"].notna()
].copy()
reference["zip"] = reference["zip"].astype(int)

# Use median lat/lon per zip to reduce sensitivity to edge-case outlier fires
centroids = (
    reference
    .groupby("zip")[["latitude", "longitude"]]
    .median()
    .reset_index()
)

print(f"\nCentroid reference table: {len(centroids)} unique zips")

tree = cKDTree(centroids[["latitude", "longitude"]].values)


# 3: Recover missing zip codes

missing_mask = fire_raw["zip"].isna() & fire_raw["latitude"].notna()
missing      = fire_raw[missing_mask].copy()

print(f"\nFire rows missing zip (with coords): {len(missing)}")

dists, idxs = tree.query(missing[["latitude", "longitude"]].values, k=1)

# Assign recovered zip and distance for traceability
missing["zip_recovered"]   = centroids.iloc[idxs]["zip"].values
missing["recovery_dist_deg"] = dists

# Accept only matches within threshold
within      = missing[dists <= MAX_DIST_DEG].copy()
beyond      = missing[dists >  MAX_DIST_DEG].copy()

within["zip"] = within["zip_recovered"]

print(f"  Recovered (within {MAX_DIST_DEG} deg): {len(within)}")
print(f"  Dropped   (beyond threshold)       : {len(beyond)}")
if len(beyond):
    print("  Dropped rows:")
    print(
        beyond[["FIRE_NAME", "Year", "latitude", "longitude",
                "recovery_dist_deg", "GIS_ACRES"]]
        .to_string(index=False)
    )


# 4: Assemble clean fire table

# Rows that already had a zip (unchanged)
had_zip = fire_raw[fire_raw["zip"].notna()].copy()

# Stack: original + recovered
fires_all = pd.concat([had_zip, within], ignore_index=True)
fires_all["zip"] = fires_all["zip"].astype(int)

print(f"\nFire rows before recovery : {len(fire_raw):,}")
print(f"Fire rows after  recovery : {len(fires_all):,}  (+{len(within)} recovered, -{len(beyond)} dropped)")


# 5: Apply wildfire filter 

wildfires = fires_all[
    (fires_all["OBJECTIVE"] != EXCLUDE_OBJECTIVE) &
    (fires_all["CAUSE"]     != EXCLUDE_CAUSE)
].copy()

wildfires["year"] = wildfires["Year"].astype(int)
wildfires["zip"]  = wildfires["zip"].astype(int)

print(f"\nAfter wildfire filter (excl. prescribed burns): {len(wildfires):,}")
print("  Fires per year:")
print(wildfires.groupby("year").size().to_string())

# Acreage recovered
orig_acres  = fire_raw["GIS_ACRES"].sum()
new_acres   = wildfires["GIS_ACRES"].sum()
print(f"\nTotal acres before recovery filter : {orig_acres:,.0f}")
print(f"Total acres after  recovery filter : {new_acres:,.0f}")
print(f"Coverage: {new_acres / orig_acres * 100:.1f}%")


# 6: Save

out_path = OUTPUT_DIR / "fires_geocoded.csv"
wildfires.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(wildfires):,} rows)")