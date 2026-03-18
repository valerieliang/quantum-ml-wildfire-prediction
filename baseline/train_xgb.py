"""
Classical Baseline -- XGBoost
2026 Quantum Sustainability Challenge: Wildfire Risk Modeling

Trains an XGBoost classifier on the PCA-reduced feature set and saves
the fitted model and validation predictions for comparison.

Inputs   (../pca/)
------
  ../pca/pca_train.csv    -- zip, year, wildfire, PC1..PC5
  ../pca/pca_val.csv      -- zip, year, wildfire, PC1..PC5

Outputs  (output/)
-------
  output/xgb_model.joblib          -- fitted XGBClassifier
  output/xgb_val_predictions.csv   -- zip, year, wildfire, xgb_prob, xgb_pred_05, xgb_pred_03
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

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

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw   = round(n_neg / n_pos, 2)

print(f"  Train : {len(y_train):,} rows  |  positives: {n_pos} ({n_pos/len(y_train):.1%})")
print(f"  Val   : {len(y_val):,} rows   |  positives: {y_val.sum()} ({y_val.mean():.1%})")
print(f"  scale_pos_weight = {spw}  (n_neg / n_pos)")


# Train 
#
# scale_pos_weight = n_negative / n_positive is XGBoost's equivalent of
# class_weight='balanced'. It scales the gradient contribution of positive
# samples so the minority class has equal total weight in training.
#
# Hyperparameter notes:
#   max_depth=4        shallow trees reduce overfitting on 5 features
#   learning_rate=0.05 slow learning rate offset by more trees (n_estimators=300)
#   subsample=0.8      row subsampling per tree; reduces variance
#   colsample_bytree=0.8  feature subsampling per tree (less critical with only 5 PCs)

print("\nTraining XGBoost (scale_pos_weight={})...".format(spw))
xgb = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = spw,
    eval_metric      = "logloss",
    random_state     = SEED,
    verbosity        = 0,
)
xgb.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)],
    verbose  = False,
)

print("\n  Feature importance (gain):")
importance = dict(zip(PC_COLS, xgb.feature_importances_))
for pc, imp in sorted(importance.items(), key=lambda x: -x[1]):
    bar = "#" * int(imp * 60)
    print(f"    {pc}  {bar}  {imp:.4f}")


# Predict 

xgb_prob = xgb.predict_proba(X_val)[:, 1]

preds = val[["zip", "year", LABEL]].copy().reset_index(drop=True)
preds["xgb_prob"]     = xgb_prob.round(4)
preds["xgb_pred_0.5"] = (xgb_prob >= 0.5).astype(int)
preds["xgb_pred_0.3"] = (xgb_prob >= 0.3).astype(int)


# Save 

joblib.dump(xgb, OUTPUT_DIR / "xgb_model.joblib")
preds.to_csv(OUTPUT_DIR / "xgb_val_predictions.csv", index=False)

print(f"\nSaved:")
print(f"  {OUTPUT_DIR}/xgb_model.joblib")
print(f"  {OUTPUT_DIR}/xgb_val_predictions.csv")
print("\nRun compare_models.py to see full metrics.")