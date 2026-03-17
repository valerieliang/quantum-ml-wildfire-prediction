"""
geocode_missing_zips.py
-----------------------
Uses the Census Bureau's free public geocoder to recover ZCTA5 zip codes
for wildfire records that have lat/lon but no zip code.

Scans the wildfire CSV directly on each run — no cache file. Any fire row
where OBJECTID is populated, zip is blank, and lat/lon are present will be
sent to the geocoder.

API: https://geocoding.geo.census.gov/geocoder/geographies/coordinates
  - Free, no API key, no account required
  - Returns ZCTA5 (Census Zip Code Tabulation Areas) — the official
    Census-defined boundaries, matching the zip codes in the weather rows
  - Layer 86 = ZIP Code Tabulation Areas

Output:
  wildfire_weather_geocoded.csv    — full fire table with zip column patched
"""

import pandas as pd
import numpy as np
import urllib.request
import json
import time

# ── Config ─────────────────────────────────────────────────────────────────

RAW_PATH = "wildfire_weather.csv"

# Census geocoder endpoint — coordinates -> ZCTA5
# x = longitude, y = latitude (Census convention)
# benchmark: Public_AR_Current  (current TIGER/Line)
# vintage:   Current_Current    (current geographies)
# layers:    86                 (ZIP Code Tabulation Areas)
CENSUS_URL = (
    "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    "?x={lon}&y={lat}"
    "&benchmark=Public_AR_Current"
    "&vintage=Current_Current"
    "&layers=86"
    "&format=json"
)

# Pause between requests - Census has no published rate limit but be polite
REQUEST_DELAY_S = 0.3


# Helper to query census data

def lookup_zcta(lat: float, lon: float) -> str | None:
    """
    Query the Census geocoder for a single coordinate pair.
    Returns the ZCTA5 string (e.g. '96094') or None if not found.

    Census response structure (abbreviated):
    {
      "result": {
        "geographies": {
          "ZIP Code Tabulation Areas": [
            { "GEOID": "96094", "ZCTA5CE10": "96094", ... }
          ]
        }
      }
    }
    """
    url = CENSUS_URL.format(lat=lat, lon=lon)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())

        zctas = (
            data.get("result", {})
                .get("geographies", {})
                .get("ZIP Code Tabulation Areas", [])
        )
        if zctas:
            # GEOID is the 5-digit ZCTA5 code
            return zctas[0].get("GEOID") or zctas[0].get("ZCTA5CE10")
    except Exception as e:
        print(f"    Warning: request failed for ({lat}, {lon}): {e}")
    return None


# Load and scan the wildfire file

print("Loading raw data...")
df = pd.read_csv(RAW_PATH, low_memory=False)

# Fire rows only (OBJECTID populated)
fire_raw = df[df["OBJECTID"].notna()].copy()
fire_raw["OBJECTID"] = fire_raw["OBJECTID"].astype(float)

# Identify rows that need geocoding: missing zip but have both coordinates (longitude, latitude)
to_geocode = fire_raw[
    fire_raw["zip"].isna() &
    fire_raw["latitude"].notna() &
    fire_raw["longitude"].notna()
].copy()

print(f"  Total fire rows            : {len(fire_raw):,}")
print(f"  Already have zip           : {fire_raw['zip'].notna().sum():,}")
print(f"  Missing zip (will geocode) : {len(to_geocode):,}")


# Geocode each missing row

results = []   # list of dicts: OBJECTID -> recovered zip or NaN

for i, (_, row) in enumerate(to_geocode.iterrows()):
    lat = row["latitude"]
    lon = row["longitude"]

    zcta = lookup_zcta(lat, lon)

    if zcta:
        print(f"  [{i+1}/{len(to_geocode)}] ({lat:.4f}, {lon:.4f}) -> {zcta}")
        results.append({"OBJECTID": row["OBJECTID"], "zip_recovered": int(zcta)})
    else:
        print(f"  [{i+1}/{len(to_geocode)}] ({lat:.4f}, {lon:.4f}) -> NOT FOUND (will be dropped)")
        results.append({"OBJECTID": row["OBJECTID"], "zip_recovered": np.nan})

    time.sleep(REQUEST_DELAY_S)


# Summary

results_df = pd.DataFrame(results)
n_found    = results_df["zip_recovered"].notna().sum()
n_missing  = len(results_df)

print(f"\nGeocoding complete:")
print(f"  Recovered : {n_found} / {n_missing}")
print(f"  Not found : {n_missing - n_found} (will remain without zip)")


# Patch zip column and write output

fire_patched = fire_raw.merge(results_df, on="OBJECTID", how="left")
fire_patched["zip"] = fire_patched["zip"].combine_first(fire_patched["zip_recovered"])
fire_patched = fire_patched.drop(columns=["zip_recovered"])

print(f"\nFire rows with zip before patch  : {fire_raw['zip'].notna().sum():,}")
print(f"Fire rows with zip after patch   : {fire_patched['zip'].notna().sum():,}")
print(f"Still missing zip (dropped later): {fire_patched['zip'].isna().sum():,}")

OUTPUT_CSV = "wildfire_weather_geocoded.csv"
fire_patched.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved -> {OUTPUT_CSV}")