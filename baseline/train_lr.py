"""
Classical Baseline -- Logistic Regression
2026 Quantum Sustainability Challenge: Wildfire Risk Modeling

Trains a logistic regression classifier on the PCA-reduced feature set
and saves the fitted model and validation predictions for comparison.

Inputs   (../pca/)
------
  ../pca/pca_train.csv    -- zip, year, wildfire, PC1..PC5
  ../pca/pca_val.csv      -- zip, year, wildfire, PC1..PC5

Outputs  (output/)
-------
  output/lr_model.joblib          -- fitted LogisticRegression
  output/lr_val_predictions.csv   -- zip, year, wildfire, lr_prob, lr_pred_05, lr_pred_03
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions   import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Config 

HERE       = Path(__file__).parent          # baseline/
PCA_DIR    = HERE.parent / "pca"            # pca/
OUTPUT_DIR = HERE / "output"               # baseline/output/
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
# class_weight='balanced' re-weights the loss function so the minority class
# (fire zips, ~9%) has equal total influence to the majority class (no fire).
# This is equivalent to upsampling positives by ~9.9x without duplicating rows.
#
# C=1.0 is the default inverse regularisation strength. Larger C = less
# regularisation. The PCA components are already standardised (mean 0, std 1)
# so the default is appropriate here.

print("\nTraining Logistic Regression (class_weight='balanced')...")
lr = LogisticRegression(
    class_weight = "balanced",
    max_iter     = 1000,
    random_state = SEED,
    solver       = "lbfgs",
    C            = 1.0,
)
lr.fit(X_train, y_train)

print("  Coefficients per PC:")
for pc, coef in zip(PC_COLS, lr.coef_[0]):
    direction = "^ fire" if coef > 0 else "v fire"
    print(f"    {pc}: {coef:+.4f}  ({direction})")


# Predict 

lr_prob = lr.predict_proba(X_val)[:, 1]

preds = val[["zip", "year", LABEL]].copy().reset_index(drop=True)
preds["lr_prob"]     = lr_prob.round(4)
preds["lr_pred_0.5"] = (lr_prob >= 0.5).astype(int)
preds["lr_pred_0.3"] = (lr_prob >= 0.3).astype(int)


# Save 

joblib.dump(lr, OUTPUT_DIR / "lr_model.joblib")
preds.to_csv(OUTPUT_DIR / "lr_val_predictions.csv", index=False)

print(f"\nSaved:")
print(f"  {OUTPUT_DIR}/lr_model.joblib")
print(f"  {OUTPUT_DIR}/lr_val_predictions.csv")
print("\nRun compare_models.py to see full metrics.")