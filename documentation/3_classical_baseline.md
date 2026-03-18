# Classical Baseline Models
## 2026 Quantum Sustainability Challenge -- Wildfire Risk and Insurance Premium Modeling

---

## 1. Business Context and Objective

The goal of this project is not to build the most accurate classifier in the abstract
sense -- it is to build a model that is **useful to insurance companies** pricing
wildfire risk across California zip codes.

That distinction matters enormously for how we evaluate and deploy the model.

### What an insurer actually needs

An insurance company writing homeowner policies in California faces two competing
pressures:

1. **Coverage risk.** If the model misses a high-risk zip code (a false negative),
   the insurer writes an underpriced policy in a zone that may produce a large claim.
   At scale, systematic under-pricing of risk is an existential threat to the
   insurer's book of business -- as seen when major carriers began exiting the
   California market in 2023-2024 after sustained wildfire losses.

2. **Customer fairness and retention.** If the model over-flags low-risk zip codes
   (false positives), homeowners in safe areas receive inflated premiums. This is
   both commercially damaging (customers shop away to competitors) and ethically
   problematic -- particularly in lower-income areas where housing costs are already
   strained.

The insurer's ideal model therefore **maximises fire coverage (recall) while keeping
false alarms (false positives) at a level that does not systematically overcharge
low-risk customers.** This is a recall-priority problem with a precision floor, not
a pure F1 optimisation problem.

### Why accuracy is a misleading metric here

The 2022 validation set contains 2,593 California zip codes, of which only 180 (6.9%)
had a wildfire. A model that predicts "no fire" for every zip code achieves 93.1%
accuracy while being completely useless -- it would cause an insurer to underprice
risk in all 180 fire-prone areas. All evaluation in this project therefore uses
**recall, precision, F1, and AUC-ROC** exclusively.

---

## 2. Dataset and Features

### Training and validation split

| Split | Years | Rows | Positive rate |
|---|---|---|---|
| Training | 2018-2021 | 10,372 | 9.1% (948 fire zip-years) |
| Validation | 2022 | 2,593 | 6.9% (180 fire zips) |
| Prediction | 2023 | 2,593 | unknown (target) |

Each row represents one California zip code in one year. The label is binary: 1 if
at least one unplanned, uncontrolled wildland fire occurred in that zip code in that
year, 0 otherwise.

### Feature pipeline

Raw features were engineered from two sources -- monthly weather observations
(temperature and precipitation per zip code) and geocoded fire incident records --
then reduced to 5 principal components via PCA.

**Weather features (13 columns, engineered from monthly observations):**
- Annual mean tmax and tmin; total annual precipitation
- Peak monthly tmax; minimum monthly precipitation
- Dry season (Jun-Oct) mean tmax, tmin, total precip, count of months below 5mm
- Wet season (Nov-May) mean tmax, tmin, total precip, count of dry months

**Fire history features (4 columns, lag-safe -- no leakage):**
- `fires_last1`: fire count in the preceding year
- `fires_last3`: fire count over the preceding 3 years
- `acres_last3`: total acres burned over the preceding 3 years (log-transformed)
- `years_since_fire`: years since the most recent fire in this zip (capped at 10)

**PCA reduction:**
All 17 features were standardised (zero mean, unit variance) and reduced to 5
principal components, which explain 87.9% of total variance. This dimensionality
reduction is required to fit within the qubit budget of the quantum VQC circuit
in Phase 4, and also partially decorrelates the features before modelling.

| Component | Variance explained | Cumulative |
|---|---|---|
| PC1 | 36.4% | 36.4% |
| PC2 | 19.9% | 56.4% |
| PC3 | 14.4% | 70.8% |
| PC4 | 9.8% | 80.6% |
| PC5 | 7.3% | 87.9% |

PC2 consistently ranks as the most predictive component across all three models,
likely capturing a combination of high dry-season temperatures and low precipitation
-- the classic fire weather signature.

### Important caveat on 2022 validation weather

The source weather dataset ends at December 2021. The 13 weather features for 2022
are therefore **estimated** using each zip code's 4-year climatological mean
(2018-2021). Temperature features are reliable under this approach (year-to-year
standard deviation ~0.5-0.9 C), but precipitation features are noisier
(~119mm std). Validation metrics should be treated as indicative benchmarks rather
than ground-truth performance estimates. The same imputation strategy is applied
to the 2023 prediction set.

---

## 3. Class Imbalance Handling

At 6.9-9.1% positive rate, a naive classifier will ignore the minority class
entirely. All three models use explicit class reweighting:

- **Logistic Regression:** `class_weight='balanced'` -- sklearn computes per-class
  weights as `n_samples / (n_classes * n_samples_per_class)`, which is equivalent
  to upsampling fire zip-years by approximately 9.9x in the loss function.
- **Random Forest:** `class_weight='balanced'` -- applies the same per-sample
  weighting to each tree's bootstrap sample during training.
- **XGBoost:** `scale_pos_weight = n_negative / n_positive = 9.94` -- scales the
  gradient contribution of positive samples to equalise class influence during
  boosting.

None of these approaches duplicate data. They adjust the loss function so the model
is penalised equally for a missed fire as for a missed non-fire, despite the class
imbalance.

---

## 4. Models

### 4.1 Logistic Regression

Logistic regression fits a linear decision boundary in the 5-dimensional PC space.
The predicted probability is `p = sigmoid(w1*PC1 + w2*PC2 + ... + w5*PC5 + b)`.

**Fitted coefficients (direction of fire risk):**

| Component | Coefficient | Direction |
|---|---|---|
| PC1 | -0.154 | lower PC1 = higher risk |
| PC2 | +0.560 | higher PC2 = higher risk |
| PC3 | -0.018 | weak negative |
| PC4 | +0.003 | negligible |
| PC5 | -0.054 | weak negative |

PC2 dominates the prediction, consistent with its feature importance ranking in the
tree-based models. The near-zero coefficients on PC3-PC5 suggest the signal in those
components is largely noise after PCA.

**Why LR performs best here:** PCA produces decorrelated, standardised components
that are by construction linearly separable. This is precisely the space where
logistic regression has its largest advantage. The non-linear capacity of the tree
ensembles is partially neutralised because PCA has already removed the inter-feature
interactions that XGBoost and RF are designed to exploit.

### 4.2 Random Forest

Random Forest grows 500 independent decision trees on bootstrap samples of the
training data and aggregates their probability estimates. Unlike boosting, each tree
is grown independently, which makes RF robust to overfitting in high-noise settings.

**Hyperparameters:**
- `n_estimators=500` -- enough trees for stable probability estimates
- `max_depth=None` -- full-depth trees; variance is controlled by averaging
- `min_samples_leaf=5` -- prevents tiny leaf nodes; smooths probability outputs
- `max_features='sqrt'` -- standard RF heuristic (sqrt(5) ~ 2 features per split)

**Feature importance (mean decrease in impurity):**

| Component | Importance |
|---|---|
| PC2 | 36.9% |
| PC3 | 19.1% |
| PC1 | 18.0% |
| PC4 | 13.0% |
| PC5 | 12.9% |

RF spreads importance more evenly across PCs 3-5 compared to LR, suggesting it
captures some non-linear interactions in the lower-variance components. Despite this,
its AUC (0.776) sits below LR (0.852), likely because those components contain more
noise than signal in this dataset.

### 4.3 XGBoost

XGBoost builds an ensemble of shallow gradient-boosted trees, where each tree
corrects the residual errors of the previous one. It is generally state-of-the-art
for tabular classification tasks, but on this dataset it underperforms LR on AUC.

**Hyperparameters:**
- `n_estimators=300`, `learning_rate=0.05` -- slow learning rate with more trees
- `max_depth=4` -- shallow trees to avoid overfitting on 5 features
- `subsample=0.8`, `colsample_bytree=0.8` -- stochastic sampling for variance reduction

**Feature importance (gain):**

| Component | Importance |
|---|---|
| PC2 | 35.5% |
| PC3 | 19.3% |
| PC1 | 17.4% |
| PC5 | 14.4% |
| PC4 | 13.4% |

XGBoost's feature importance distribution is nearly identical to Random Forest,
confirming that both tree ensembles are learning the same patterns. The fact that
both are outperformed by LR strongly implies the signal in these 5 PCA components
is primarily linear.

---

## 5. Threshold Selection

### Why the choice of threshold matters

All three models output a **probability** between 0 and 1. The threshold converts
that probability into a binary prediction: "flag as fire risk" or "flag as no risk."
The choice of threshold directly controls the recall-precision tradeoff.

There is no universally correct threshold. The right value depends on the cost
structure of the application -- specifically, how an insurer weighs the cost of
missing a fire (false negative) against the cost of overcharging a safe customer
(false positive).

### Full threshold sweep -- Logistic Regression

| Threshold | Fires caught | Missed | False alarms | Precision | F1 |
|---|---|---|---|---|---|
| 0.1 | 180 / 180 (100%) | 0 | 2,413 / 2,413 (100%) | 6.9% | 0.130 |
| 0.2 | 180 / 180 (100%) | 0 | 2,413 / 2,413 (100%) | 6.9% | 0.130 |
| 0.3 | 167 / 180 (93%) | 13 | 1,263 / 2,413 (52%) | 11.7% | 0.208 |
| 0.4 | 156 / 180 (87%) | 24 | 827 / 2,413 (34%) | 15.9% | 0.268 |
| 0.5 | 138 / 180 (77%) | 42 | 438 / 2,413 (18%) | 24.0% | 0.365 |
| 0.6 | 133 / 180 (74%) | 47 | 358 / 2,413 (15%) | 27.1% | 0.396 |
| 0.7 | 126 / 180 (70%) | 54 | 311 / 2,413 (13%) | 28.8% | 0.408 |
| 0.8 | 107 / 180 (59%) | 73 | 216 / 2,413 (9%) | 33.1% | 0.425 |
| 0.9 | 64 / 180 (36%) | 116 | 180 / 2,413 (7%) | 35.6% | 0.356 |

**Why t=0.1 and t=0.2 are degenerate:** At these thresholds, LR flags all 2,593
zip codes as fire risk -- 100% recall but 100% false alarm rate. This is not a useful
operating point. It is equivalent to an insurer applying high-risk premiums to every
property in California regardless of location, which defeats the purpose of the model
entirely and would expose the insurer to regulatory and competitive pressure.

### Recommended operating threshold: 0.3

Given the insurance use case -- maximise fire coverage while avoiding systematic
overcharging of low-risk customers -- **t=0.3 is the most defensible choice.**

At t=0.3, Logistic Regression:
- Catches **167 of 180 fire zip codes (93% recall)** -- missing only 13
- Flags **1,430 of 2,593 zip codes** for elevated risk review
- Has a **1-in-9 hit rate** among flagged zips (every 9 flags includes ~1 real fire)
- Misses 13 true fire zips, representing the model's unavoidable coverage gap

The practical interpretation for an insurer: flag the 1,430 zips at t=0.3 for
underwriter review or tiered pricing. The remaining 1,163 zips can be priced at
standard rates with high confidence (only 13 true fires are in that group). This
creates a two-tier system rather than blanket high-risk pricing.

### Why not t=0.8 (best F1)?

The F1-maximising threshold (0.8) sacrifices 40% of recall -- missing 73 fire zip
codes -- in exchange for a lower false alarm rate. For a classification benchmark
this is reasonable, but for an insurer the 73 missed zips represent 73 underpriced
policies in areas likely to generate claims. The F1 metric weights precision and
recall equally; the insurer's cost function does not.

---

## 6. Model Comparison

### Summary at recommended threshold (t=0.3)

| Model | AUC-ROC | F1 | Recall | Precision | Fires caught | False alarms |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.852** | **0.208** | **92.8%** | 11.7% | 167 / 180 | 1,263 / 2,413 (52%) |
| XGBoost | 0.780 | 0.156 | 93.9% | 8.5% | 169 / 180 | 1,821 / 2,413 (75%) |
| Random Forest | 0.776 | 0.149 | 96.7% | 8.1% | 174 / 180 | 1,975 / 2,413 (82%) |

XGBoost and Random Forest have marginally higher recall at t=0.3 but at a
significantly higher false alarm cost -- flagging 75-82% of non-fire zips versus
52% for LR. **Logistic Regression achieves the best balance at this threshold.**

### Pros and cons

**Logistic Regression**

Pros:
- Best AUC-ROC (0.852) -- strongest overall discrimination between fire and non-fire
- Most interpretable -- the PC2 coefficient directly explains "hotter, drier = higher risk"
- Best precision at any given recall level (highest point on the PR curve)
- Calibrated probabilities -- the output score can be used directly as a risk index
- Fast to train and retrain as new data arrives each year

Cons:
- Assumes a linear decision boundary in PC space -- cannot capture interactions
  between features that were absorbed into PCs
- Performance is tied to the quality of the PCA reduction; if PCs fail to capture
  a novel fire-driving pattern (e.g. a new wind regime), LR will be slow to adapt
- The near-zero coefficients on PC3-PC5 mean the model is effectively using only
  1-2 features, which limits its ceiling

**Random Forest**

Pros:
- Most robust to outliers and noise -- bootstrap aggregation reduces variance
- Provides the highest recall at low thresholds among all three models
- Feature importance is stable and interpretable at the PC level
- No assumption of linearity -- can capture complex interactions

Cons:
- AUC (0.776) is notably below LR, meaning its ranking of zip codes by fire
  probability is less reliable
- Probability calibration is poor -- RF probabilities tend to be compressed toward
  0.5 and should not be used as absolute risk scores without Platt scaling
- More expensive to retrain (500 trees) and less amenable to incremental updates
- The non-linear capacity is underutilised on already-decorrelated PCA features

**XGBoost**

Pros:
- Highest precision at low recall settings (t=0.9: 35.2% precision vs 33.1% for LR)
- Built-in regularisation (L1/L2) reduces overfitting risk
- Can capture feature interactions that LR misses

Cons:
- AUC (0.780) is second-lowest, underperforming LR on the primary discrimination metric
- At t=0.3 it flags 75% of non-fire zips -- the highest false alarm rate of the three
- Sensitive to hyperparameter choices; the current configuration may not be optimal
- Like RF, XGBoost's non-linear advantage is partially eroded by PCA decorrelation
- Probabilities are not well-calibrated by default

### Why LR beats tree ensembles on PCA features

This result is expected and worth emphasising in the Task 1B writeup. PCA components
are by construction:

1. **Linearly decorrelated** -- all pairwise correlations between PCs are zero
2. **Ordered by variance** -- PC1 explains the most variance; PC5 the least
3. **Standardised** -- all PCs have mean 0 and variance 1

This is the ideal input space for logistic regression. The non-linear interactions
and feature correlations that XGBoost and RF are designed to exploit have already
been removed by PCA. If the models were trained on the raw 17 features instead of
PCA components, the tree ensembles would likely close the gap.

---

## 7. QML Benchmark Targets

The classical baseline results establish the performance floor that the Variational
Quantum Classifier (VQC) must clear to demonstrate practical advantage. Based on
the results above, the benchmarks at the recommended operating threshold (t=0.3) are:

| Metric | LR baseline (t=0.3) | Target for VQC |
|---|---|---|
| AUC-ROC | 0.852 | > 0.852 |
| Recall | 92.8% | >= 90% |
| Precision | 11.7% | > 11.7% |
| F1 | 0.208 | > 0.208 |
| False alarm rate | 52.3% | < 52.3% |

A VQC that matches LR's AUC of 0.852 on 5 qubits would be a strong result
demonstrating that quantum feature encoding is competitive with the best classical
linear approach. A VQC that exceeds it -- even marginally -- would suggest that
quantum entanglement is capturing non-linear structure in the PC space that LR's
linear boundary misses. Given the near-zero LR coefficients on PC3-PC5, there is
plausible signal in those components that a non-linear quantum kernel could exploit.

---

## 8. Files

| File | Description |
|---|---|
| `baseline/train_lr.py` | Train and save Logistic Regression |
| `baseline/train_rf.py` | Train and save Random Forest |
| `baseline/train_xgb.py` | Train and save XGBoost |
| `baseline/compare_models.py` | Threshold sweep, metrics, plots |
| `baseline/output/lr_val_predictions.csv` | LR probability scores per zip |
| `baseline/output/rf_val_predictions.csv` | RF probability scores per zip |
| `baseline/output/xgb_val_predictions.csv` | XGBoost probability scores per zip |
| `baseline/output/threshold_sweep.csv` | F1/prec/recall at t=0.1 to 0.9 |
| `baseline/output/best_threshold_metrics.csv` | Best-F1 threshold per model |
| `baseline/output/pr_curve.png` | Precision-recall curves |
| `baseline/output/threshold_sweep.png` | F1 vs threshold plot |

To add QML results to the comparison once the VQC is trained:

```bash
python baseline/compare_models.py --qml baseline/output/qml_val_predictions.csv
```

The QML predictions file must contain columns: `zip`, `year`, `wildfire`, `qml_prob`.