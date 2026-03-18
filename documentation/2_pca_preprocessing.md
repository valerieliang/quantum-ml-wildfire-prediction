# 2 — PCA Preprocessing
## 2026 Quantum Sustainability Challenge: Wildfire Risk & Insurance Premium Modeling

---

## Table of Contents

1. [Purpose and Context](#1-purpose-and-context)
2. [Why Dimensionality Reduction Is Required](#2-why-dimensionality-reduction-is-required)
3. [Pre-PCA Transform: log1p on `acres_last3`](#3-pre-pca-transform-log1p-on-acres_last3)
4. [Standardization](#4-standardization)
5. [PCA Configuration and Variance Analysis](#5-pca-configuration-and-variance-analysis)
6. [Leakage Prevention](#6-leakage-prevention)
7. [Output Files](#7-output-files)
8. [Using Fitted Objects at Inference Time](#8-using-fitted-objects-at-inference-time)
9. [Known Limitations](#9-known-limitations)
10. [Script Execution](#10-script-execution)

---

## 1. Purpose and Context

`perform_pca.py` is the final step in the data preparation pipeline. It takes the
17-feature matrices produced in Phase 1 (`train_features.csv`, `val_features.csv`,
`predict_2023_features.csv`) and reduces them to 5 principal components suitable
for quantum angle encoding.

This step sits between data preparation (Phase 1) and model training (Phase 3/4).
Its outputs are the direct inputs to both the classical baseline and the Variational
Quantum Classifier (VQC).

```
train_features.csv (17 features)  ─┐
val_features.csv   (17 features)  ─┼─► perform_pca.py ─► pca_train.csv  (5 PCs)
predict_2023_features.csv         ─┘                  ─► pca_val.csv    (5 PCs)
                                                       ─► pca_predict_2023.csv
                                                       ─► pca_scaler.joblib
                                                       ─► pca_model.joblib
```

---

## 2. Why Dimensionality Reduction Is Required

### Qubit budget

A Variational Quantum Classifier encodes each feature as a rotation angle on one
qubit. The current hardware and simulator constraints for this challenge make circuits
with more than ~8 qubits impractical:

- Deeper circuits (more features → more qubits → more gate layers) accumulate noise
  faster on real hardware.
- Statevector simulation scales exponentially with qubit count — 17 qubits requires
  2¹⁷ = 131,072 complex amplitudes per sample.
- COBYLA and SPSA optimizers converge poorly in very high-dimensional parameter spaces.

Reducing to 5 features (5 qubits) keeps simulation fast and keeps the VQC
circuit shallow enough to be trainable.

### Feature correlation

Several of the 17 raw features are highly correlated:

- `tmax_annual`, `tmax_dry`, `tmax_wet`, and `tmax_peak` all capture temperature
  and move together across zip codes.
- `prcp_annual`, `prcp_dry`, and `prcp_wet` partition the same total precipitation
  signal.
- `fires_last1` and `fires_last3` overlap by construction.

PCA collapses correlated features into orthogonal components, which is a better
representation for a VQC than feeding redundant rotations into separate qubits.

---

## 3. Pre-PCA Transform: log1p on `acres_last3`

Before standardizing or running PCA, `acres_last3` is transformed with `log1p`:

```python
train_features["acres_last3"] = np.log1p(train_features["acres_last3"])
```

This is applied identically to val and predict sets.

### Why this is necessary

`acres_last3` has an extreme right tail driven by a small number of catastrophic
multi-year fire events (Caldor, Dixie, Creek fires):

| Statistic | Raw value |
|-----------|-----------|
| Median | 0 acres (most zips never burned) |
| 75th percentile | 0 acres |
| 99th percentile | ~85,000 acres |
| Maximum | 1,033,369 acres |

Without transformation, StandardScaler centers and scales this column but the
~1M acre outlier still sits ~65 standard deviations from the mean. In PCA, this
single column would dominate the first principal component and collapse all other
zips to near-zero values in that dimension — destroying the signal from the 16
other features.

`log1p` (= log(1 + x)) is appropriate here because:

- It handles zeros cleanly: `log1p(0) = 0`, so zip codes with no fire history
  are not penalized.
- It compresses the 1M-acre outlier to `log1p(1,033,369) ≈ 13.85`, which is
  large but no longer pathological.
- It preserves the ordinal relationship between zip codes (more acres = larger
  value).

After transformation the column's maximum drops from 1,033,369 to 13.85, and
StandardScaler can scale it proportionally alongside the other 16 features.

---

## 4. Standardization

After the log1p transform, all 17 features are standardized with `StandardScaler`
(zero mean, unit variance):

```python
scaler  = StandardScaler()
X_train = scaler.fit_transform(train_features.values)   # fit + transform
X_val   = scaler.transform(val_features.values)          # transform only
X_pred  = scaler.transform(predict_2023_features.values) # transform only
```

Standardization is required before PCA because PCA is sensitive to feature scale
— a feature measured in millimetres of precipitation would otherwise dominate
features measured in degrees Celsius.

The scaler is **fit on training data only**. See [Section 6](#6-leakage-prevention)
for why this matters.

---

## 5. PCA Configuration and Variance Analysis

```python
pca = PCA(n_components=5, random_state=42)
X_train_pca = pca.fit_transform(X_train)
```

### Variance explained

| Component | Individual | Cumulative |
|-----------|-----------|------------|
| PC1 | 36.44% | 36.44% |
| PC2 | 19.94% | 56.38% |
| PC3 | 14.41% | 70.80% |
| PC4 | 9.76% | 80.55% |
| PC5 | 7.33% | 87.89% |

5 components explain **87.9% of variance** in the training set. The scree flattens
after PC5 — the 6th component contributes only ~4%, and including it would add a
qubit and increase circuit depth for minimal gain.

### What the components likely capture

PCA components are linear combinations of all 17 input features. Based on the
known correlations in the raw data:

- **PC1** is primarily a temperature/aridity axis — it separates hot, dry Southern
  California zip codes from cooler, wetter Northern California ones. This is the
  strongest structural pattern in the data.
- **PC2** likely captures precipitation variability — the contrast between wet
  coastal/mountain zips and dry inland/desert zips that PC1 doesn't fully separate.
- **PC3–PC5** increasingly reflect fire history signals (`acres_last3` after
  log1p, `fires_last3`, `years_since_fire`) and finer-grained seasonal patterns.

Exact component loadings can be inspected from the saved PCA object:

```python
import joblib
pca = joblib.load("data/pca_model.joblib")
print(pca.components_)   # shape: (5, 17) — one loading vector per PC
```

---

## 6. Leakage Prevention

The scaler and PCA are fit **exclusively on training data** (2018–2021). The
validation (2022) and prediction (2023) sets are transformed using the
already-fitted objects:

```python
# Correct — fit only on train
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)    # no fit here

pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)   # no fit here
```

Fitting the scaler or PCA on validation or combined data would constitute data
leakage: the scaler's mean and variance, and the PCA's eigenvectors, would be
informed by future observations. This would produce optimistic validation metrics
that do not reflect real-world performance.

The fitted objects are saved as `.joblib` files specifically so they can be loaded
at inference time without re-fitting.

---

## 7. Output Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `pca_train.csv` | 10,372 | 8 | `zip`, `year`, `wildfire`, PC1–PC5. Direct input to model training. |
| `pca_val.csv` | 2,593 | 8 | `zip`, `year`, `wildfire`, PC1–PC5. Used for all validation metrics. |
| `pca_predict_2023.csv` | 2,593 | 7 | `zip`, `year`, PC1–PC5. No label column — scored by the trained model. |
| `pca_scaler.joblib` | — | — | Fitted `StandardScaler`. Required to transform new feature data. |
| `pca_model.joblib` | — | — | Fitted `PCA(n_components=5)`. Required to project new data into PC space. |

All CSV files have zero null values. Positive rates are preserved exactly from
the input label files (train: 9.14%, val: 6.94%).

### Column schema for `pca_train.csv` and `pca_val.csv`

```
zip       — California zip code (int)
year      — Year (int; 2018–2021 for train, 2022 for val)
wildfire  — Binary label: 1 = fire occurred, 0 = no fire
PC1       — First principal component (float, standardized)
PC2       — Second principal component (float, standardized)
PC3       — Third principal component (float, standardized)
PC4       — Fourth principal component (float, standardized)
PC5       — Fifth principal component (float, standardized)
```

---

## 8. Using Fitted Objects at Inference Time

When scoring new data (e.g., generating 2023 predictions after training the
model), apply the **same** scaler and PCA that were fit on training data:

```python
import joblib
import numpy as np
import pandas as pd

# Load fitted objects
scaler = joblib.load("data/pca_scaler.joblib")
pca    = joblib.load("data/pca_model.joblib")

# Load new feature data (must have the same 17 columns in the same order)
new_data = pd.read_csv("data/predict_2023_features.csv")
X_new    = new_data.drop(columns=["zip", "year"]).copy()

# Apply the same log1p transform used during training
X_new["acres_last3"] = np.log1p(X_new["acres_last3"])

# Transform (do NOT call fit or fit_transform here)
X_new_scaled = scaler.transform(X_new.values)
X_new_pca    = pca.transform(X_new_scaled)

# X_new_pca is now a (2593, 5) array ready for model.predict()
```

> **Common mistake:** calling `scaler.fit_transform(X_new)` instead of
> `scaler.transform(X_new)`. This re-centers the scaler on the new data,
> producing different PC values than the training set was encoded with. The
> model will receive inputs in a different coordinate space than it was trained
> on and predictions will be meaningless.

---

## 9. Known Limitations

**PCA components are not interpretable features.** Each PC is a linear combination
of all 17 raw features. While this is fine for predictive performance, it makes
it harder to explain which physical factors (temperature, drought, fire history)
are driving a specific prediction. For the Task 1B evaluation section, feature
importance analysis should be conducted on the raw 17-feature space using the
classical baseline, not on PCA components.

**log1p is applied to `acres_last3` only.** Other features with moderate skew
(e.g., `prcp_dry`, `fires_last3`) are left untransformed. StandardScaler handles
moderate skew adequately. Only `acres_last3` had a pathological outlier
(~65 standard deviations) that required explicit pre-treatment.

**5 components is a fixed choice.** 87.9% cumulative variance was judged
sufficient for a 5-qubit VQC. If the classical baseline shows that 6 or 7 features
materially improve performance, rerunning PCA with `n_components=6` and updating
the VQC circuit is straightforward — just reload the scaler and refit PCA.

**PCA assumes linear structure.** If the relationship between features and wildfire
occurrence is highly non-linear (e.g., fire risk only spikes above specific
temperature + drought thresholds), PCA may discard variance that is disproportionately
informative for the classification task. Kernel PCA or direct feature selection are
alternatives if model performance is unsatisfactory.

---

## 10. Script Execution

`perform_pca.py` must be run after all four Phase 1 scripts have completed:

```bash
# Prerequisite: all Phase 1 outputs must exist
# (see 1_data_preparation.md for the full pipeline)

python perform_pca.py
```

Expected console output:

```
Loading data...
  Train : 10,372 rows x 17 features
  Val   : 2,593 rows x 17 features
  Predict 2023: 2,593 rows x 17 features
Applying log1p transform to: ['acres_last3']
  acres_last3: max before=1,033,369  ->  max after=13.85
Fitting StandardScaler on training data...
Fitting PCA(n_components=5) on training data...
Variance explained per component:
  PC1: 0.3644  (cumulative: 0.3644)
  PC2: 0.1994  (cumulative: 0.5638)
  PC3: 0.1441  (cumulative: 0.7080)
  PC4: 0.0976  (cumulative: 0.8055)
  PC5: 0.0733  (cumulative: 0.8789)
Total variance explained by 5 components: 0.8789
...
All sanity checks passed.
```

After this step the pipeline is complete and the outputs in `data/` are ready
for Phase 3 (classical baseline) and Phase 4 (VQC).
