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
  - Added California zip code filter (90001-96162) after assembly.
    The raw data contains 12 fire rows with out-of-state zip codes
    (AZ 85xxx, NV 89xxx, OR 97xxx) for fires near the CA border that were
    already tagged with their correct non-CA zip. These zips are absent from
    the CA weather dataset so including them creates orphaned fire rows that
    can never join to a feature vector. They are now dropped cleanly.

Output:
  data/fires_geocoded.csv  — full CA fire table with zip gaps filled where possible
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

# Config

RAW_PATH      = "data/wildfire_weather.csv"
OUTPUT_DIR    = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_DIST_DEG  = 0.3        # Accept nearest-neighbour match within ~33 km

EXCLUDE_CAUSE     = 18     # Prescribed / controlled burns
EXCLUDE_OBJECTIVE = 2      # Non-wildland objective

# All valid California USPS zip codes fall within this range
CA_ZIP_MIN = 90001
CA_ZIP_MAX = 96162


# Load and split 

print("Loading raw data...")
df = pd.read_csv(RAW_PATH, low_memory=False)

fire_raw = df[df["OBJECTID"].notna()].copy()
print(f"  Total fire rows: {len(fire_raw):,}")


# Build zip centroids from rows that already have zip + coords 

reference = fire_raw[
    fire_raw["zip"].notna() &
    fire_raw["latitude"].notna() &
    fire_raw["longitude"].notna()
].copy()
reference["zip"] = reference["zip"].astype(int)

# Restrict centroid table to CA zips so the KD-tree never assigns a recovered
# fire row to an out-of-state zip.
reference = reference[reference["zip"].between(CA_ZIP_MIN, CA_ZIP_MAX)].copy()

centroids = (
    reference
    .groupby("zip")[["latitude", "longitude"]]
    .median()
    .reset_index()
)

print(f"\nCentroid reference table: {len(centroids)} unique CA zips")

tree = cKDTree(centroids[["latitude", "longitude"]].values)


# Recover missing zip codes

missing_mask = fire_raw["zip"].isna() & fire_raw["latitude"].notna()
missing      = fire_raw[missing_mask].copy()

print(f"\nFire rows missing zip (with coords): {len(missing)}")

dists, idxs = tree.query(missing[["latitude", "longitude"]].values, k=1)

missing["zip_recovered"]     = centroids.iloc[idxs]["zip"].values
missing["recovery_dist_deg"] = dists

within = missing[dists <= MAX_DIST_DEG].copy()
beyond = missing[dists >  MAX_DIST_DEG].copy()

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


# Assemble fire table 

had_zip   = fire_raw[fire_raw["zip"].notna()].copy()
fires_all = pd.concat([had_zip, within], ignore_index=True)
fires_all["zip"] = fires_all["zip"].astype(int)

print(f"\nFire rows before recovery : {len(fire_raw):,}")
print(f"Fire rows after  recovery : {len(fires_all):,}  (+{len(within)} recovered, -{len(beyond)} dropped)")


# Filter to California zips only 
#
# 12 rows carry out-of-state zip codes (AZ 85xxx, NV 89xxx, OR 97xxx) for
# fires that occurred near the CA border and were already tagged with their
# real non-CA zip in the source data. Since these zips have no matching rows
# in the CA weather dataset they would create orphaned records that silently
# inflate fire-history counts for neighboring CA zip codes. Drop them here.

before_ca = len(fires_all)
fires_all = fires_all[fires_all["zip"].between(CA_ZIP_MIN, CA_ZIP_MAX)].copy()
n_oos     = before_ca - len(fires_all)

print(f"\nOut-of-state zip rows dropped (AZ/NV/OR): {n_oos}")
print(f"Fire rows after CA filter : {len(fires_all):,}")


# Apply wildfire filter 

wildfires = fires_all[
    (fires_all["OBJECTIVE"] != EXCLUDE_OBJECTIVE) &
    (fires_all["CAUSE"]     != EXCLUDE_CAUSE)
].copy()

wildfires["year"] = wildfires["Year"].astype(int)
wildfires["zip"]  = wildfires["zip"].astype(int)

print(f"\nAfter wildfire filter (excl. prescribed burns): {len(wildfires):,}")
print("  Fires per year:")
print(wildfires.groupby("year").size().to_string())

orig_acres = fire_raw["GIS_ACRES"].sum()
new_acres  = wildfires["GIS_ACRES"].sum()
print(f"\nTotal acres in raw fire rows  : {orig_acres:,.0f}")
print(f"Total acres after all filters : {new_acres:,.0f}")
print(f"Coverage: {new_acres / orig_acres * 100:.1f}%")

oos_remaining = wildfires[~wildfires["zip"].between(CA_ZIP_MIN, CA_ZIP_MAX)]
assert len(oos_remaining) == 0, \
    f"Out-of-state zips still present: {oos_remaining['zip'].unique()}"
print("\nCA-only assertion passed.")


# Save 

out_path = OUTPUT_DIR / "fires_geocoded.csv"
wildfires.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(wildfires):,} rows)")