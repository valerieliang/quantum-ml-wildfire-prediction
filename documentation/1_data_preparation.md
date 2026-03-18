# 1 — Data Preparation
## 2026 Quantum Sustainability Challenge: Wildfire Risk & Insurance Premium Modeling

---

## Table of Contents

1. [Challenge Overview](#1-challenge-overview)
2. [Raw Dataset Structure](#2-raw-dataset-structure)
3. [Data Quality Issues Discovered](#3-data-quality-issues-discovered)
4. [Zip Code Recovery — Two Approaches](#4-zip-code-recovery--two-approaches)
   - 4.1 [KD-Tree Spatial Lookup (`zip_recovery.py`)](#41-kd-tree-spatial-lookup-zip_recoverypy)
   - 4.2 [API Geocoding (`geocode_missing_zips.py`)](#42-api-geocoding-geocode_missing_zipspy)
5. [Wildfire Definition and Filtering](#5-wildfire-definition-and-filtering)
6. [Training Set Construction (`generate_training_data.py`)](#6-training-set-construction-generate_training_datapy)
   - 6.1 [Label Grid](#61-label-grid)
   - 6.2 [Weather Features](#62-weather-features)
   - 6.3 [Fire History Features](#63-fire-history-features)
   - 6.4 [Final Join](#64-final-join)
7. [Validation Set Construction (`generate_validation_set.py`)](#7-validation-set-construction-generate_validation_setpy)
   - 7.1 [Why 2022 Is the Right Validation Year](#71-why-2022-is-the-right-validation-year)
   - 7.2 [Weather Imputation for 2022](#72-weather-imputation-for-2022)
   - 7.3 [Fire History for 2022](#73-fire-history-for-2022)
8. [2023 Prediction Set (`generate_2023_prediction_set.py`)](#8-2023-prediction-set-generate_2023_prediction_setpy)
9. [Output File Reference](#9-output-file-reference)
10. [Feature Schema](#10-feature-schema)
11. [Known Limitations and Caveats](#11-known-limitations-and-caveats)
12. [Script Execution Order](#12-script-execution-order)

---

## 1. Challenge Overview

The 2026 Deloitte Quantum Sustainability Challenge tasks participants with applying Quantum Machine Learning (QML) to predict wildfire risk across California zip codes. Phase 1 covers all data preparation work that feeds into the model training pipeline.

**Task 1A** requires predicting whether a wildfire will occur in a given California zip code in 2023, trained on historical data from 2018–2022. A "wildfire" is defined by the challenge as a fire that burns in a wildland setting and is unplanned and uncontrolled.

**Task 2** (insurance premium modeling) also relies on the zip-level risk scores produced by Task 1, making clean, complete, and correctly labeled data foundational to the entire submission.

---

## 2. Raw Dataset Structure

The raw file (`wildfire_weather.csv`) contains **125,476 rows** across **30 columns** but is structurally unusual: it is a vertical stack of two completely different row types sharing one table, distinguished by whether the `OBJECTID` column is populated.

### Row type 1 — Fire incident records
- **Count:** 2,218 rows
- **Identifier:** `OBJECTID` is populated (non-null)
- **Grain:** One row per fire incident
- **Key columns:** `OBJECTID`, `Year`, `AGENCY`, `FIRE_NAME`, `ALARM_DATE`, `CONT_DATE`, `CAUSE`, `OBJECTIVE`, `GIS_ACRES`, `latitude`, `longitude`, `zip`
- **Years covered:** 2018, 2019, 2020, 2021, 2022, 2023

| Year | Fire Incidents |
|------|---------------|
| 2018 | 416 |
| 2019 | 319 |
| 2020 | 505 |
| 2021 | 388 |
| 2022 | 306 |
| 2023 | 284 |

### Row type 2 — Monthly weather records
- **Count:** 123,258 rows
- **Identifier:** `OBJECTID` is null
- **Grain:** One row per zip code per month
- **Key columns:** `zip`, `year_month`, `avg_tmax_c`, `avg_tmin_c`, `tot_prcp_mm`, `station`
- **Coverage:** January 2018 — December 2021 (48 months × 2,593 zip codes = 124,464 expected rows; minor gaps exist for some station readings)
- **Unique zip codes:** 2,593

> **Critical gap:** Weather data ends at December 2021. There are no weather rows for 2022 or 2023. This structural gap shapes all downstream modeling decisions.

### Splitting the two row types

The very first step in all Phase 1 scripts is splitting on `OBJECTID`:

```python
fire_raw    = df[df["OBJECTID"].notna()].copy()   # 2,218 rows
weather_raw = df[df["OBJECTID"].isna()].copy()    # 123,258 rows
```

---

## 3. Data Quality Issues Discovered

### Issue 1 — 337 fire records have no zip code

Of the 2,218 fire incident rows, **337 (15.2%) are missing a zip code**. However, all 337 rows have valid `latitude` and `longitude` coordinates. This meant the records could not be dropped without consequence.

The significance became clear when examining acreage: these 337 fires account for **3,355,242 acres out of a total 9,222,152 acres — 36.4% of all recorded burned area**. The missing-zip fires are predominantly large incidents on federal land (USFS, NPS, BLM) that span multiple zip code boundaries and were never assigned a postal code in the source system.

| Agency | Typical Missing-Zip Fires |
|--------|--------------------------|
| USFS (US Forest Service) | Deep national forest fires crossing zip boundaries |
| NPS (National Park Service) | Fires in national parks (Yosemite, Sequoia, etc.) |
| BLM (Bureau of Land Management) | Remote desert and rangeland fires |

Dropping these rows would systematically underrepresent the largest, most ecologically significant wildfires in the training data — a serious labeling bias. Recovery was therefore prioritized.

### Issue 2 — Weather coverage ends at December 2021

The weather table has complete 12-month coverage for all 2,593 zip codes in each year from 2018–2021 (no missing months or zip gaps). However, there are zero weather rows for 2022 and 2023. Since the challenge requires predicting 2023, and best practice is to validate on a held-out year that mirrors prediction conditions, both the 2022 validation year and the 2023 prediction year must operate without direct weather observations.

This was addressed in two ways:
1. Using 2022 as the validation year (rather than a random 2018–2021 slice), since it matches the same no-weather-data constraint as 2023.
2. Imputing 2022 weather features from 2018–2021 per-zip climatological means in `generate_validation_set.py`.

### Issue 3 — Class imbalance

The target variable is heavily imbalanced. In the training set (2018–2021):

- **Total zip-year rows:** 10,372 (2,593 zips × 4 years)
- **Positive labels (fire occurred):** 942
- **Positive rate:** ~9.1%

This is expected — most California zip codes do not experience a wildfire in any given year. Any classifier that ignores this and predicts "no fire" everywhere achieves ~91% accuracy trivially. Phase 2 must address this with class weighting, SMOTE, or threshold tuning.

---

## 4. Zip Code Recovery — Two Approaches

Two separate scripts were written to recover zip codes for the 337 missing-zip fire records. They use different techniques with different tradeoffs.

### 4.1 KD-Tree Spatial Lookup (`zip_recovery.py`)

**Approach:** Build a reference table of zip code centroids from the 1,881 fire rows that already have both a zip code and coordinates. Use a KD-tree (a spatial index structure enabling fast nearest-neighbor search) to assign the nearest known zip centroid to each missing-zip row. Reject matches beyond a configurable distance threshold.

**Centroid construction:**
```python
centroids = (
    reference
    .groupby("zip")[["latitude", "longitude"]]
    .median()
    .reset_index()
)
```

Using the median (rather than mean) makes centroids robust to fires at the extreme geographic edges of a zip code. The reference table covers **559 unique zip codes** across California.

**Distance threshold:** The default is `MAX_DIST_DEG = 0.30` (~33 km). This was selected by examining the distance distribution:

| Threshold | Fires Recovered | Coverage |
|-----------|----------------|----------|
| 0.10 deg (~11 km) | 110 | 33% |
| 0.20 deg (~22 km) | 236 | 70% |
| 0.25 deg (~28 km) | 277 | 82% |
| 0.30 deg (~33 km) | 312 | 93% |

At 0.30 degrees, the script recovers **312 of 337 rows**, dropping 25 that fall in very remote wilderness areas where the nearest known zip centroid is too far away to assign with any confidence.

**Advantages:** Fully self-contained — no external API calls, no API keys, no network access required. Runs in seconds. Completely reproducible.

**Limitations:** Centroids are derived only from the 559 zips that appear in the fire incident data. Large swaths of rural California with no recorded fires have no centroid reference points, which is precisely why the most remote incidents get dropped.

**Output:** `fires_geocoded.csv` — the full wildfire table with recovered zip codes applied and rows outside the threshold dropped.

### 4.2 API Geocoding (`geocode_missing_zips.py`)

**Approach:** Query the US Census Bureau's free public reverse geocoding API for each missing-zip coordinate pair, receiving back the official ZCTA5 (ZIP Code Tabulation Area) boundary that the point falls within.

**API endpoint:**
```
https://geocoding.geo.census.gov/geocoder/geographies/coordinates
  ?x={longitude}&y={latitude}
  &benchmark=Public_AR_Current
  &vintage=Current_Current
  &layers=86    ← layer 86 = 2020 ZIP Code Tabulation Areas
  &format=json
```

The Census geocoder returns ZCTA5 codes, which are the official Census-defined approximations of USPS zip codes. For California these align closely with postal zip codes.

**Rate limiting:** The Census API has no published rate limit, but a `REQUEST_DELAY_S = 0.3` second pause between requests is built in as a courtesy. At 337 rows this adds approximately 1.7 minutes of wall-clock time.

**No cache:** This script runs a clean scan each execution. For a one-time data preparation step this is acceptable.

**Advantages:** Returns ground-truth Census-defined boundaries rather than nearest-centroid approximations. Recovers fires that the KD-tree cannot (where there is no nearby zip centroid in the fire data). Completely free, no API key required.

**Limitations:** Requires network access to `geocoding.geo.census.gov`. If that endpoint is unreachable (e.g. in a sandboxed environment), the script fails. Results can also vary slightly between Census benchmark vintages.

**A third variant — Claude API geocoding** (`phase1_train_api.py`) — was also written to handle environments where only `api.anthropic.com` is reachable. It uses `claude-haiku-4-5-20251001` as a structured JSON geocoder with a confidence field, accepting only `"high"` or `"medium"` confidence results.

## 5. Wildfire Definition and Filtering

The challenge defines a wildfire as a fire that "burns in a wildland setting and is unplanned and uncontrolled." The raw data contains two types of records that must be excluded:

| Exclusion Criterion | Field | Value | Meaning |
|--------------------|-------|-------|---------|
| Prescribed burn | `OBJECTIVE` | `2` | Managed/intentional burn |
| Escaped prescribed burn | `CAUSE` | `18` | Planned origin, escaped control |

All other OBJECTIVE and CAUSE values represent unplanned ignitions and are retained. The CAUSE codes retained include lightning (1), equipment use (2), smoking (3), campfire (4), debris (5), arson (7), vehicle (10), power line (11), and unknown/unidentified (14), among others.

**Filter applied in all scripts:**
```python
wildfires = fires_all[
    (fires_all["OBJECTIVE"] != 2) &
    (fires_all["CAUSE"]     != 18)
].copy()
```

After filtering, **2,182 of 2,218 incidents** (36 excluded) qualify as unplanned wildland fires under the challenge definition.

---

## 6. Training Set Construction (`generate_training_data.py`)

The training set covers **2018–2021** — the only years where both fire incident labels and full monthly weather observations are available. The output is a flat feature matrix with one row per (zip code, year) combination.

**Depends on:** `data/fires_geocoded.csv` (output of `zip_recovery.py`)

### 6.1 Label Grid

A complete grid of every California zip code in every training year is constructed first, so that non-fire years are explicitly labeled 0 (rather than being absent from the data):

```python
zip_year_index = pd.MultiIndex.from_product(
    [all_zips, TRAIN_YEARS], names=["zip", "year"]
)
labels = pd.DataFrame(index=zip_year_index).reset_index()
```

`all_zips` is derived from the 2,593 unique zip codes present in the weather table — this is the authoritative California zip code universe for the challenge dataset.

Fire flags are then merged in via a left join, with unmatched rows filled as 0:

```python
labels["wildfire"] = labels["wildfire"].fillna(0).astype(int)
```

**Label counts per year:**

| Year | Fire Zip-Years | Total Zip-Years | Positive Rate |
|------|---------------|-----------------|---------------|
| 2018 | 241 | 2,593 | 9.3% |
| 2019 | 191 | 2,593 | 7.4% |
| 2020 | 269 | 2,593 | 10.4% |
| 2021 | 241 | 2,593 | 9.3% |
| **Total** | **942** | **10,372** | **9.1%** |

### 6.2 Weather Features

Monthly weather rows are aggregated into annual and seasonal summaries per (zip, year). The seasonal split reflects the structure of California's fire season:

- **Dry season (Jun–Oct):** High temperatures, minimal precipitation — peak fire risk window
- **Wet season (Nov–May):** Rain and snowpack recharge, vegetation growth that becomes fuel in the following dry season

**13 weather features constructed:**

| Feature | Description |
|---------|-------------|
| `tmax_annual` | Mean of monthly max temperature across all 12 months (°C) |
| `tmin_annual` | Mean of monthly min temperature across all 12 months (°C) |
| `prcp_annual` | Total precipitation across all 12 months (mm) |
| `tmax_peak` | Single hottest monthly max temperature in the year (°C) |
| `prcp_min` | Single driest monthly total precipitation (mm) |
| `tmax_dry` | Mean max temperature during Jun–Oct (°C) |
| `tmin_dry` | Mean min temperature during Jun–Oct (°C) |
| `prcp_dry` | Total precipitation during Jun–Oct (mm) |
| `dry_months_dry` | Count of Jun–Oct months with < 5 mm precipitation |
| `tmax_wet` | Mean max temperature during Nov–May (°C) |
| `tmin_wet` | Mean min temperature during Nov–May (°C) |
| `prcp_wet` | Total precipitation during Nov–May (mm) |
| `dry_months_wet` | Count of Nov–May months with < 5 mm precipitation |

The 5 mm/month drought proxy threshold was chosen as a practical near-zero precipitation cutoff; any month below this level is effectively rainless in the context of wildfire fuel moisture.

Because weather data covers all 2,593 zip codes for all 12 months of every training year (2018–2021), the weather join uses `how="inner"`, and the resulting matrix contains **zero null values**.

### 6.3 Fire History Features

Fire history features capture each zip code's recent fire activity using only data from prior years — no leakage into the label year. This ensures the features are valid for both validation (2022) and final prediction (2023), where weather data is absent, making these lagged statistics the primary signal available.

**4 fire history features constructed:**

| Feature | Description |
|---------|-------------|
| `fires_last1` | Number of wildfire incidents in this zip in (year − 1) |
| `fires_last3` | Total wildfire incidents in this zip over the prior 3 years |
| `acres_last3` | Total GIS acres burned in this zip over the prior 3 years |
| `years_since_fire` | Years elapsed since the most recent recorded fire in this zip (capped at 10 if no fire in the lookback window) |

The lookback window for `years_since_fire` searches back up to 10 years, defaulting to 10 if no fire is found — representing the "long-dormant" state for zip codes with no recent fire history.

> **Design note:** The full wildfire table (all years, including 2018–2023) is used as the source for history lookups. For a training year of 2018, the lookback reaches into 2015–2017 fire records. For 2021, it uses 2018–2020. This avoids edge effects at the start of the training window.

### 6.4 Final Join

The three components are joined sequentially:

```python
train = (
    labels
    .merge(weather_feats, on=["zip", "year"], how="inner")
    .merge(fire_hist,     on=["zip", "year"], how="left")
)
```

The inner join on weather ensures only zip-years with full weather coverage are retained (which is all of 2018–2021, so no rows are lost). The left join on fire history preserves all rows — zips with no fire history simply receive zero counts.

**Final training matrix:** 10,372 rows × 17 feature columns, zero null values.

---

## 7. Validation Set Construction (`generate_validation_set.py`)

The validation set covers **2022** — a single cross-section of all 2,593 zip codes for that year.

### 7.1 Why 2022 Is the Right Validation Year

The typical approach to model validation is to hold out a random fraction of the training data. Here, that approach would be misleading for a specific reason: years 2018–2021 have full weather features, but 2022 and 2023 do not. A model validated on a 2018–2021 slice is evaluated under better conditions than it will face during prediction.

Using 2022 as the validation year creates symmetry:

| Condition | 2018–2021 Training | 2022 Validation | 2023 Prediction |
|-----------|-------------------|-----------------|-----------------|
| Fire labels available | Yes | Yes | No (target) |
| Weather features available | Yes | No (imputed) | No |
| Fire history features available | Yes | Yes | Yes |

Because 2022 and 2023 share the same feature conditions (no direct weather observations), the model's performance on 2022 is a realistic proxy for how well it will perform on 2023. This is a much more honest benchmark than validating on years with full feature coverage.

### 7.2 Weather Imputation for 2022

Since there are no weather rows for 2022 in the dataset, the 13 weather features are estimated using the per-zip climatological mean across 2018–2021:

```python
clim_means = (
    train_full
    .groupby("zip")[WEATHER_COLS]
    .mean()
    .reset_index()
)
```

Each zip code's 2022 weather estimate is the average of its own readings across the four prior years. This is a reasonable approximation for temperature features, which are highly stable year-to-year. Precipitation features are noisier but still capture the baseline dryness profile of each zip.

**Stability of estimates (overall standard deviation across training set):**

| Feature | Std Dev | Reliability |
|---------|---------|-------------|
| `tmax_annual` | 3.22°C | High |
| `tmin_annual` | 2.84°C | High |
| `tmax_dry` | 4.55°C | High |
| `tmin_dry` | 2.69°C | High |
| `tmax_wet` | 3.18°C | High |
| `tmin_wet` | 3.36°C | High |
| `prcp_min` | 2.89 mm | High |
| `dry_months_dry` | 1.05 months | High |
| `dry_months_wet` | 1.92 months | Moderate |
| `prcp_dry` | 75.5 mm | Moderate |
| `prcp_annual` | 209.7 mm | Noisy — interpret with caution |
| `prcp_wet` | 185.2 mm | Noisy — interpret with caution |
| `tmax_peak` | 5.53°C | Moderate |

Temperature and drought proxy features are reliable estimates. Annual and seasonal precipitation totals are noisier but are still far more informative than leaving those columns empty.

> **Important caveat:** 2022 validation metrics should be treated as approximate benchmarks. The model is being evaluated on estimated (not measured) weather inputs, which introduces uncertainty not present during training. Do not over-optimize hyperparameters based on 2022 validation scores.

### 7.3 Fire History for 2022

Fire history features for 2022 are computed using the same lag-safe logic as training, looking back to 2019–2021:

| Feature | Source Years |
|---------|-------------|
| `fires_last1` | 2021 incidents |
| `fires_last3` | 2019, 2020, 2021 incidents |
| `acres_last3` | 2019, 2020, 2021 acreage |
| `years_since_fire` | Most recent prior year with a fire (back to 2012) |

These are the only features derived from actual observations (not estimates) in the validation set.

**2022 label distribution:**
- Fire zip codes: 180 of 2,593
- Positive rate: 6.9% (slightly lower than training years, consistent with year-to-year variation)

---

## 8. 2023 Prediction Set (`generate_2023_prediction_set.py`)

The 2023 prediction set is the final output of Phase 1 — the feature matrix the trained model will score to produce the wildfire risk predictions required by the challenge submission. Unlike the training and validation sets, it carries no labels because 2023 is the target year being predicted.

**Feature availability for 2023:**

| Feature group | Available | Source |
|--------------|-----------|--------|
| Weather features (13 cols) | Estimated | Per-zip climatological mean across 2018–2021 |
| Fire history features (4 cols) | Yes (actual) | Lagged from 2020, 2021, 2022 incidents |

The weather imputation strategy is identical to the one used for the 2022 validation set, which keeps the feature space consistent between validation and final prediction. This is important: a model that performs well on 2022 under estimated weather is being evaluated under the same conditions it will face on 2023.

The fire history lookback for 2023 reaches the most recent real data available:

| Feature | Source years |
|---------|-------------|
| `fires_last1` | 2022 incidents |
| `fires_last3` | 2020, 2021, 2022 incidents |
| `acres_last3` | 2020, 2021, 2022 acreage |
| `years_since_fire` | Searches back to 2013 |

**Output:** `predict_2023_features.csv` — 2,593 rows × 17 feature columns (plus `zip` and `year` for traceability), zero null values, same column order as `train_features.csv` and `val_features.csv`.

---

## 9. Output File Reference

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `wildfire_weather.csv` | 125,476 | 30 | Raw source file — fire incident rows and monthly weather rows interleaved |
| `fires_geocoded.csv` | 2,149 | 33 | Wildfire table with zip codes recovered via KD-tree, out-of-state zips removed, and prescribed burns filtered out |
| `train_features.csv` | 10,372 | 17 | Feature matrix for model training (2018–2021) |
| `train_labels.csv` | 10,372 | 3 | `zip`, `year`, `wildfire` binary label for 2018–2021 |
| `val_features.csv` | 2,593 | 17 | Feature matrix for 2022 validation (weather columns are climatologically imputed) |
| `val_labels.csv` | 2,593 | 3 | `zip`, `year`, `wildfire` binary label for 2022 |
| `predict_2023_features.csv` | 2,593 | 19 | Feature matrix for 2023 prediction — 17 features plus `zip` and `year` columns; no labels |
| `pca_train.csv` | 10,372 | 8 | PCA-reduced training set — `zip`, `year`, `wildfire` + PC1–PC5; input to VQC and classical baseline |
| `pca_val.csv` | 2,593 | 8 | PCA-reduced validation set — same schema as `pca_train.csv` |
| `pca_predict_2023.csv` | 2,593 | 7 | PCA-reduced 2023 prediction set — `zip`, `year` + PC1–PC5; no label column |
| `pca_scaler.joblib` | — | — | Fitted `StandardScaler` (fit on training data only); required to transform new data at inference |
| `pca_model.joblib` | — | — | Fitted `PCA(n_components=5)` object; required to transform new data at inference |

---

## 10. Feature Schema

Both `train_features.csv` and `val_features.csv` share the same 17-column schema in the same column order.

```
tmax_annual       — Annual mean max temperature (C)
tmin_annual       — Annual mean min temperature (C)
prcp_annual       — Annual total precipitation (mm)
tmax_peak         — Single hottest monthly max temperature (C)
prcp_min          — Single driest monthly total (mm)
tmax_dry          — Jun–Oct mean max temperature (C)
tmin_dry          — Jun–Oct mean min temperature (C)
prcp_dry          — Jun–Oct total precipitation (mm)
dry_months_dry    — Jun–Oct months with < 5 mm precip (count 0–5)
tmax_wet          — Nov–May mean max temperature (C)
tmin_wet          — Nov–May mean min temperature (C)
prcp_wet          — Nov–May total precipitation (mm)
dry_months_wet    — Nov–May months with < 5 mm precip (count 0–7)
fires_last1       — Wildfire count in prior year (integer)
fires_last3       — Wildfire count over prior 3 years (integer)
acres_last3       — Total acres burned in prior 3 years (float)
years_since_fire  — Years since most recent prior wildfire (integer, 1–10)
```

Labels file columns: `zip` (int), `year` (int), `wildfire` (0 or 1).

---

## 11. Known Limitations and Caveats

**Zip code coverage gap in fire data.** The KD-tree centroid reference is built from only 559 of the 2,593 zip codes that appear in the weather table. The remaining 2,034 zips have no centroid reference because no fire has been recorded there. If a missing-zip fire falls near one of these uncovered zips, it will be incorrectly assigned to a different zip or dropped entirely. This bias affects only a small fraction of incidents and is partially mitigated by the API geocoding variant.

**KD-tree centroids are zip-code level, not boundary-level.** The nearest-centroid heuristic is an approximation — a point on the edge of a zip code may be closer to the centroid of an adjacent zip than to its own centroid. For very large fires that span multiple zip codes, the lat/lon recorded in the dataset represents the fire's point of origin or centroid, which may lie in a different zip than where the bulk of the damage occurred.

**Weather features for 2022 and 2023 are estimated.** The dataset provides no actual weather observations for these years. The per-zip climatological mean is a defensible imputation strategy but introduces measurement error, particularly for precipitation. Do not treat validation metrics on 2022 as equivalent in quality to cross-validation within 2018–2021.

**Class imbalance is significant.** At ~9% positive in training and ~7% in validation, any model that predicts "no fire" everywhere achieves high accuracy trivially. Precision, recall, F1, and AUC-ROC are more meaningful metrics than accuracy for this problem. Class weighting or resampling will be required in Phase 2 for both the classical baseline and the QML model.

**Fire history is sparse for 2018.** Because fire history features look back up to 3 years, the 2018 training rows have history reaching back to 2015. If the wildfire table does not include pre-2018 incidents (which it does not for most zip codes), `fires_last3` and `acres_last3` for 2018 will undercount historical fire activity. This affects one of four training years.

**Zip codes do not align perfectly with wildfire spread.** A zip code is an administrative postal unit, not a geographic or ecological unit. Large fires frequently burn across multiple zip code boundaries. The binary label "fire occurred in this zip in this year" is a simplification of a spatially continuous phenomenon.

---

## 12. Script Execution Order

Run scripts in the following order:

```bash
# Step 1: Recover zip codes for the 337 missing-zip fire records using KD-tree
python zip_recovery.py

# Step 2: Build training matrix (2018–2021)
#   Input:  fires_geocoded.csv
#   Output: train_features.csv, train_labels.csv
python generate_training_data.py

# Step 3: Build validation matrix (2022)
#   Input:  fires_geocoded.csv, train_features.csv, train_labels.csv
#   Output: val_features.csv, val_labels.csv
python generate_validation_set.py

# Step 4: Build 2023 prediction matrix
#   Input:  fires_geocoded.csv, train_features.csv, train_labels.csv
#   Output: predict_2023_features.csv
python generate_2023_prediction_set.py

# Step 5: Dimensionality reduction via PCA
#   Input:  train_features.csv, val_features.csv, predict_2023_features.csv
#   Output: pca_train.csv, pca_val.csv, pca_predict_2023.csv,
#           pca_scaler.joblib, pca_model.joblib
#   See:    2_pca_preprocessing.md for design decisions and output details
python perform_pca.py
```

After these steps, the directory contains fully prepared, null-free feature matrices ready for Phase 2.

| Set | File | Purpose |
|-----|------|---------|
| Training | `train_features.csv` + `train_labels.csv` | Fit the model |
| Validation | `val_features.csv` + `val_labels.csv` | Benchmark performance before final prediction |
| Prediction | `predict_2023_features.csv` | Score with the trained model; output is the challenge submission |

After step 5, PCA-reduced versions of all three sets are also available (`pca_train.csv`, `pca_val.csv`, `pca_predict_2023.csv`). These are the direct inputs to the VQC and classical baseline models. See **`2_pca_preprocessing.md`** for the full PCA design rationale, transform choices, and variance analysis.