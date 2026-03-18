"""
Classical Baseline -- Random Forest
2026 Quantum Sustainability Challenge: Wildfire Risk Modeling

Trains a Random Forest classifier on the PCA-reduced feature set
and saves the fitted model and validation predictions for comparison.

Random Forest sits between Logistic Regression (linear, low variance)
and XGBoost (boosted, can overfit) in terms of bias/variance tradeoff.
It is also an ensemble of independently grown trees, which makes it
less sensitive to the PCA decorrelation that hurts XGBoost.

class_weight='balanced' re-weights each tree's sample weights so the
minority class (fire zips, ~9%) contributes equally to the majority.
This is equivalent to class_weight='balanced' in LogisticRegression.

Inputs   (../pca/)
------
  ../pca/pca_train.csv    -- zip, year, wildfire, PC1..PC5
  ../pca/pca_val.csv      -- zip, year, wildfire, PC1..PC5

Outputs  (output/)
-------
  output/rf_model.joblib          -- fitted RandomForestClassifier
  output/rf_val_predictions.csv   -- zip, year, wildfire, rf_prob
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Config 

HERE       = Path(__file__).parent      # baseline/
PCA_DIR    = HERE.parent / "pca"        # pca/
OUTPUT_DIR = HERE / "output"            # baseline/output/
OUTPUT_DIR.mkdir(exist_ok=True)

PC_COLS = ["PC1", "PC2", "PC3", "PC4", "PC5"]
LABEL   = "wildfire"
SEED    = 42

# Load 

print("Loading PCA data...")
train = pd.read_csv(PCA_DIR / "pca_train.csv")
val   = pd.read_csv(PCA_DIR / "pca_val.csv")

X_train, y_train = train[PC_COLS].values, train[LABEL].values
X_val,   y_val   = val[PC_COLS].values,   val[LABEL].values

print(f"  Train : {len(y_train):,} rows  |  positives: {y_train.sum()} ({y_train.mean():.1%})")
print(f"  Val   : {len(y_val):,} rows   |  positives: {y_val.sum()} ({y_val.mean():.1%})")

# Train 
#
# Hyperparameter notes:
#   n_estimators=500     more trees = lower variance; diminishing returns ~300+
#   max_depth=None       grow full trees; RF relies on averaging to avoid overfit
#   min_samples_leaf=5   prevents very small leaf nodes; smooths probability estimates
#   max_features='sqrt'  standard RF heuristic: sqrt(n_features) per split
#   class_weight='balanced'  re-weights samples to equalise class influence

print("\nTraining Random Forest (class_weight='balanced')...")
rf = RandomForestClassifier(
    n_estimators     = 500,
    max_depth        = None,
    min_samples_leaf = 5,
    max_features     = "sqrt",
    class_weight     = "balanced",
    random_state     = SEED,
    n_jobs           = -1,      # use all available CPU cores
)
rf.fit(X_train, y_train)

print("\n  Feature importance (mean decrease in impurity):")
importance = dict(zip(PC_COLS, rf.feature_importances_))
for pc, imp in sorted(importance.items(), key=lambda x: -x[1]):
    bar = "#" * int(imp * 60)
    print(f"    {pc}  {bar}  {imp:.4f}")

# Predict 

rf_prob = rf.predict_proba(X_val)[:, 1]

preds = val[["zip", "year", LABEL]].copy().reset_index(drop=True)
preds["rf_prob"] = rf_prob.round(4)

# Save 

joblib.dump(rf, OUTPUT_DIR / "rf_model.joblib")
preds.to_csv(OUTPUT_DIR / "rf_val_predictions.csv", index=False)

print(f"\nSaved:")
print(f"  {OUTPUT_DIR}/rf_model.joblib")
print(f"  {OUTPUT_DIR}/rf_val_predictions.csv")
print("\nRun compare_models.py to see full metrics.")