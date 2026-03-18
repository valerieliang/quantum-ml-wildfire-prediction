"""
Baseline Model Comparison
2026 Quantum Sustainability Challenge: Wildfire Risk Modeling

Loads saved validation predictions from train_lr.py, train_xgb.py,
train_rf.py (and optionally a QML model) and produces:

  1. Threshold sweep  -- F1 / precision / recall at every 0.1 step from
                         0.1 to 0.9. The best threshold per model is chosen
                         by maximum F1. This lets you see the full operating
                         curve before committing to a fixed threshold.

  2. Best-threshold summary table  -- one row per model at its optimal threshold.

  3. Confusion matrices  -- at each model's best threshold.

  4. Full classification report  -- at each model's best threshold.

  5. Precision-recall curve plot  -- saved to output/pr_curve.png.
     Operating points are marked at each model's best threshold.

  6. Threshold sweep plot  -- F1 vs threshold for all models, saved to
     output/threshold_sweep.png. Useful for choosing a deployment threshold.

Run order
---------
  python train_lr.py
  python train_xgb.py
  python train_rf.py
  python compare_models.py                        # classical only
  python compare_models.py --qml output/qml_val_predictions.csv  # + QML

QML predictions file format (if provided)
------------------------------------------
  Must contain columns: zip, year, wildfire, qml_prob
  One row per validation zip code, matching zip/year in pca_val.csv.

Inputs   (output/)
------
  output/lr_val_predictions.csv    -- from train_lr.py
  output/xgb_val_predictions.csv   -- from train_xgb.py
  output/rf_val_predictions.csv    -- from train_rf.py
  output/qml_val_predictions.csv   -- optional, from VQC training script

Outputs  (output/)
-------
  output/threshold_sweep.csv       -- F1/prec/recall at every 0.1 step, all models
  output/best_threshold_metrics.csv -- one row per model at its best-F1 threshold
  output/comparison_report.txt     -- full text report for submission
  output/pr_curve.png              -- precision-recall curves
  output/threshold_sweep.png       -- F1 vs threshold for all models
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

# Config 

HERE       = Path(__file__).parent      # baseline/
OUTPUT_DIR = HERE / "output"            # baseline/output/
OUTPUT_DIR.mkdir(exist_ok=True)

SWEEP_THRESHOLDS = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]   # 0.1 .. 0.9
LABEL = "wildfire"

# Args 

parser = argparse.ArgumentParser(
    description="Compare classical (and optionally QML) baselines."
)
parser.add_argument(
    "--qml",
    type    = str,
    default = None,
    metavar = "PATH",
    help    = "Path to QML predictions CSV (columns: zip, year, wildfire, qml_prob)",
)
args = parser.parse_args()

# Load predictions 

print("Loading predictions...")

required = {
    "LogisticRegression": OUTPUT_DIR / "lr_val_predictions.csv",
    "RandomForest":       OUTPUT_DIR / "rf_val_predictions.csv",
    "XGBoost":            OUTPUT_DIR / "xgb_val_predictions.csv",
}
prob_col = {
    "LogisticRegression": "lr_prob",
    "RandomForest":       "rf_prob",
    "XGBoost":            "xgb_prob",
}

for name, path in required.items():
    if not path.exists():
        script = name.lower().replace("logistic", "lr").replace(
            "randomforest", "rf").replace("xgboost", "xgb")
        print(f"ERROR: {path} not found. Run train_{script}.py first.")
        sys.exit(1)

# Load and merge all three on zip+year
lr_preds  = pd.read_csv(required["LogisticRegression"])
rf_preds  = pd.read_csv(required["RandomForest"])
xgb_preds = pd.read_csv(required["XGBoost"])

base = (
    lr_preds[["zip", "year", LABEL, "lr_prob"]]
    .merge(rf_preds[["zip",  "year", "rf_prob"]],  on=["zip", "year"], how="inner")
    .merge(xgb_preds[["zip", "year", "xgb_prob"]], on=["zip", "year"], how="inner")
)

y_true = base[LABEL].values
models = {
    "LogisticRegression": base["lr_prob"].values,
    "RandomForest":       base["rf_prob"].values,
    "XGBoost":            base["xgb_prob"].values,
}

# Optional QML
if args.qml:
    qml_path = Path(args.qml)
    if not qml_path.exists():
        print(f"ERROR: QML predictions file not found: {qml_path}")
        sys.exit(1)
    qml_preds = pd.read_csv(qml_path)
    missing = [c for c in ["zip", "year", "qml_prob"] if c not in qml_preds.columns]
    if missing:
        print(f"ERROR: QML file missing columns: {missing}")
        sys.exit(1)
    base = base.merge(qml_preds[["zip", "year", "qml_prob"]], on=["zip", "year"], how="left")
    models["QML (VQC)"] = base["qml_prob"].values
    print(f"  QML predictions loaded from {qml_path}")

print(f"  {len(base):,} validation rows  |  positives: {y_true.sum()} ({y_true.mean():.1%})")
print(f"  Models: {list(models.keys())}")

#  Threshold sweep 
#
# For each model, compute F1 / precision / recall at every threshold
# from 0.1 to 0.9 in steps of 0.1. AUC-ROC is threshold-independent
# so it is the same across all rows for a given model.

print("\nRunning threshold sweep (0.1 to 0.9)...")

sweep_rows = []
for name, prob in models.items():
    auc_roc = roc_auc_score(y_true, prob)
    for t in SWEEP_THRESHOLDS:
        y_pred = (prob >= t).astype(int)
        sweep_rows.append({
            "model":     name,
            "threshold": t,
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "auc_roc":   round(auc_roc, 4),
        })

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
print(f"  Saved: {OUTPUT_DIR}/threshold_sweep.csv")

# Print sweep table
print("\n  Full threshold sweep:")
print(f"  {'Model':<22} {'Thresh':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'AUC-ROC':>9}")
print("  " + "-" * 62)
prev_model = None
for _, row in sweep_df.iterrows():
    if row["model"] != prev_model and prev_model is not None:
        print()
    prev_model = row["model"]
    print(f"  {row['model']:<22} {row['threshold']:>7.1f} {row['f1']:>7.4f} "
          f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['auc_roc']:>9.4f}")

# Best threshold per model (by F1) 

print("\n\nBest threshold per model (max F1):")
print("  " + "-" * 62)

best_rows = []
best_thresh = {}     # threshold (used later for plots + confusion matrices)
for name in models:
    model_rows = sweep_df[sweep_df["model"] == name]
    best_row   = model_rows.loc[model_rows["f1"].idxmax()]
    best_rows.append(best_row.to_dict())
    best_thresh[name] = best_row["threshold"]
    print(f"  {name:<22}  best threshold = {best_row['threshold']:.1f}"
          f"  F1={best_row['f1']:.4f}  recall={best_row['recall']:.4f}"
          f"  precision={best_row['precision']:.4f}  AUC={best_row['auc_roc']:.4f}")

best_df = pd.DataFrame(best_rows)
best_df.to_csv(OUTPUT_DIR / "best_threshold_metrics.csv", index=False)
print(f"\n  Saved: {OUTPUT_DIR}/best_threshold_metrics.csv")

# Build text report 

lines = []

def h(text=""):
    lines.append(text)

h("=" * 64)
h("  BASELINE MODEL COMPARISON -- Task 1B")
h("  2026 Quantum Sustainability Challenge")
h("=" * 64)
h()
h(f"  Validation set : {len(y_true):,} zip codes (year 2022)")
h(f"  Positive rate  : {y_true.mean():.1%} ({y_true.sum()} fire zips)")
h(f"  Models         : {', '.join(models.keys())}")
h()
h("  Note: weather features for 2022 are estimated from 2018-2021")
h("        climatological means -- see generate_validation_set.py")

# Threshold sweep table
h()
h()
h("  THRESHOLD SWEEP  (F1 / precision / recall at every 0.1 step)")
h("  " + "-" * 62)
h(f"  {'Model':<22} {'Thresh':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'AUC-ROC':>9}")
h("  " + "-" * 62)
prev_model = None
for _, row in sweep_df.iterrows():
    if row["model"] != prev_model and prev_model is not None:
        h()
    prev_model = row["model"]
    h(f"  {row['model']:<22} {row['threshold']:>7.1f} {row['f1']:>7.4f} "
      f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['auc_roc']:>9.4f}")
h("  " + "-" * 62)

# Best threshold summary
h()
h()
h("  BEST THRESHOLD PER MODEL  (chosen by maximum F1)")
h("  " + "-" * 62)
h(f"  {'Model':<22} {'Best t':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'AUC-ROC':>9}")
h("  " + "-" * 62)
for row in best_rows:
    h(f"  {row['model']:<22} {row['threshold']:>7.1f} {row['f1']:>7.4f} "
      f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['auc_roc']:>9.4f}")
h("  " + "-" * 62)

# Confusion matrices at best threshold
h()
h()
h("  CONFUSION MATRICES  (at each model's best-F1 threshold)")
h("  " + "-" * 62)
for name, prob in models.items():
    t      = best_thresh[name]
    y_pred = (prob >= t).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    h()
    h(f"  {name}  (threshold = {t})")
    h(f"                 Predicted 0    Predicted 1")
    h(f"    Actual 0       {tn:>6}         {fp:>6}    (no fire)")
    h(f"    Actual 1       {fn:>6}         {tp:>6}    (fire)")
    h(f"    Fires caught : {tp}/{y_true.sum()} ({tp/y_true.sum():.0%})"
      f"    False alarms : {fp}/{(y_true==0).sum()} ({fp/(y_true==0).sum():.1%})")

# Classification reports at best threshold
h()
h()
h("  CLASSIFICATION REPORTS  (at each model's best-F1 threshold)")
h("  " + "-" * 62)
for name, prob in models.items():
    t      = best_thresh[name]
    y_pred = (prob >= t).astype(int)
    h()
    h(f"  {name}  (threshold = {t})")
    report = classification_report(
        y_true, y_pred,
        target_names=["no fire", "fire"],
        zero_division=0,
    )
    for line in report.splitlines():
        h("    " + line)

# Key observations
h()
h()
h("  KEY OBSERVATIONS FOR TASK 1B")
h("  " + "-" * 62)

best_f1_row  = best_df.loc[best_df["f1"].idxmax()]
best_auc_row = best_df.loc[best_df["auc_roc"].idxmax()]

h()
h(f"  Best F1      : {best_f1_row['model']}"
  f"  (F1={best_f1_row['f1']:.4f} at threshold={best_f1_row['threshold']})")
h(f"  Best AUC-ROC : {best_auc_row['model']}"
  f"  (AUC={best_auc_row['auc_roc']:.4f})")
h()
h("  For insurance/risk applications, recall is the priority metric:")
h("  a missed fire zip (false negative) means an underpriced policy;")
h("  a false alarm (false positive) means a marginally higher premium.")
h()
h("  QML benchmark targets (beat best classical at best threshold):")
h(f"    F1       > {best_df['f1'].max():.4f}")
h(f"    AUC-ROC  > {best_df['auc_roc'].max():.4f}")

h()
h("=" * 64)

report_text = "\n".join(lines)
print("\n" + report_text)

report_path = OUTPUT_DIR / "comparison_report.txt"
report_path.write_text(report_text)
print(f"\nSaved: {report_path}")

# Precision-recall curve 

print("\nGenerating precision-recall curve...")

colors = {
    "LogisticRegression": "#1f77b4",
    "RandomForest":       "#d62728",
    "XGBoost":            "#ff7f0e",
    "QML (VQC)":          "#2ca02c",
}

fig, ax = plt.subplots(figsize=(7, 5))

for name, prob in models.items():
    prec, rec, _ = precision_recall_curve(y_true, prob)
    pr_auc = auc(rec, prec)
    color  = colors.get(name, "#9467bd")
    ax.plot(rec, prec, label=f"{name}  (PR-AUC={pr_auc:.3f})", color=color, lw=2)

    # Mark operating point at best threshold
    t      = best_thresh[name]
    y_pred = (prob >= t).astype(int)
    p_pt   = precision_score(y_true, y_pred, zero_division=0)
    r_pt   = recall_score(y_true, y_pred, zero_division=0)
    ax.scatter(r_pt, p_pt, s=80, color=color, zorder=5)
    ax.annotate(f"t={t}", (r_pt, p_pt), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color=color)

# Random classifier baseline
baseline = y_true.mean()
ax.axhline(baseline, linestyle="--", color="gray", lw=1,
           label=f"Random classifier  ({baseline:.3f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve -- 2022 Validation Set", fontsize=13)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "pr_curve.png", dpi=150)
plt.close(fig)
print(f"  Saved: {OUTPUT_DIR}/pr_curve.png")

# Threshold sweep plot 

print("Generating threshold sweep plot...")

fig, ax = plt.subplots(figsize=(8, 5))

for name, prob in models.items():
    color  = colors.get(name, "#9467bd")
    subset = sweep_df[sweep_df["model"] == name]
    ax.plot(subset["threshold"], subset["f1"], marker="o", color=color,
            lw=2, ms=5, label=name)
    # Mark best threshold
    best_t = best_thresh[name]
    best_f = subset.loc[subset["threshold"] == best_t, "f1"].values[0]
    ax.scatter(best_t, best_f, s=120, color=color, zorder=5, marker="*")

ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("F1 Score vs. Threshold -- 2022 Validation Set", fontsize=13)
ax.set_xticks(SWEEP_THRESHOLDS)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "threshold_sweep.png", dpi=150)
plt.close(fig)
print(f"  Saved: {OUTPUT_DIR}/threshold_sweep.png")

print("\nDone. All results in the output/ directory.")
print("To include QML results, re-run with:")
print("  python compare_models.py --qml output/qml_val_predictions.csv")