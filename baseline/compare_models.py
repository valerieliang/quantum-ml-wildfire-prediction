"""
Baseline Model Comparison
2026 Quantum Sustainability Challenge: Wildfire Risk Modeling

Loads the saved validation predictions from train_lr.py and train_xgb.py
and produces a full side-by-side comparison: metrics table, confusion matrices,
and a precision-recall curve plot. Also accepts optional QML predictions so
the quantum model can be slotted in as a third column for Task 1B.

Run order
---------
  python train_lr.py
  python train_xgb.py
  python compare_models.py                        # classical only
  python compare_models.py --qml output/qml_val_predictions.csv  # + QML

QML predictions file format (if provided)
------------------------------------------
  Must contain columns: zip, year, wildfire, qml_prob
  One row per validation zip code, matching the zip/year in pca_val.csv.

Inputs   (output/)
------
  output/lr_val_predictions.csv    -- from train_lr.py
  output/xgb_val_predictions.csv   -- from train_xgb.py
  output/qml_val_predictions.csv   -- optional, from VQC training script

Outputs  (output/)
-------
  output/comparison_metrics.csv    -- F1 / AUC / precision / recall for all models
  output/comparison_report.txt     -- full text report (copy-paste for submission)
  output/pr_curve.png              -- precision-recall curves for all models
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for all environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

# Config 

HERE       = Path(__file__).parent          # baseline/
OUTPUT_DIR = HERE / "output"               # baseline/output/
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLDS = [0.3, 0.5]
LABEL      = "wildfire"


# Args 

parser = argparse.ArgumentParser(description="Compare classical (and optionally QML) baselines.")
parser.add_argument(
    "--qml",
    type    = str,
    default = None,
    metavar = "PATH",
    help    = "Path to QML predictions CSV (must have columns: zip, year, wildfire, qml_prob)",
)
args = parser.parse_args()


# Load predictions 

print("Loading predictions...")

lr_path  = OUTPUT_DIR / "lr_val_predictions.csv"
xgb_path = OUTPUT_DIR / "xgb_val_predictions.csv"

for p in [lr_path, xgb_path]:
    if not p.exists():
        print(f"ERROR: {p} not found. Run {p.stem.replace('_val_predictions', '')}.py first.")
        sys.exit(1)

lr_preds  = pd.read_csv(lr_path)
xgb_preds = pd.read_csv(xgb_path)

# Merge on zip+year so everything is aligned
base = lr_preds[["zip", "year", LABEL, "lr_prob"]].merge(
    xgb_preds[["zip", "year", "xgb_prob"]],
    on=["zip", "year"], how="inner",
)

y_true   = base[LABEL].values
lr_prob  = base["lr_prob"].values
xgb_prob = base["xgb_prob"].values

models = {
    "LogisticRegression": lr_prob,
    "XGBoost":            xgb_prob,
}

# Optional QML predictions
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


# Metrics 

def compute_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    auc_roc = roc_auc_score(y_true, y_prob)
    rows = []
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "model":     name,
            "threshold": t,
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "auc_roc":   round(auc_roc, 4),
        })
    return rows


all_metrics = []
for name, prob in models.items():
    all_metrics.extend(compute_metrics(name, y_true, prob))

metrics_df = pd.DataFrame(all_metrics)


# Text report 

lines = []

def h(text):
    lines.append(text)

h("=" * 64)
h("  CLASSICAL BASELINE COMPARISON -- Task 1B")
h("  2026 Quantum Sustainability Challenge")
h("=" * 64)
h(f"\n  Validation set: {len(y_true):,} zip codes (year 2022)")
h(f"  Positive rate : {y_true.mean():.1%} ({y_true.sum()} fire zips)")
h(f"  Note: weather features for 2022 are estimated from 2018-2021")
h(f"        climatological means -- see generate_validation_set.py")

# Metrics table
h("\n\n  METRICS TABLE")
h("  " + "-" * 62)
header = f"  {'Model':<22} {'Thresh':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'AUC-ROC':>9}"
h(header)
h("  " + "-" * 62)
for _, row in metrics_df.iterrows():
    h(f"  {row['model']:<22} {row['threshold']:>7.1f} {row['f1']:>7.4f} "
      f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['auc_roc']:>9.4f}")
h("  " + "-" * 62)

# Confusion matrices
h("\n\n  CONFUSION MATRICES  (threshold = 0.5)")
h("  " + "-" * 62)
for name, prob in models.items():
    y_pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    h(f"\n  {name}")
    h(f"                 Predicted 0    Predicted 1")
    h(f"    Actual 0       {tn:>6}         {fp:>6}    (no fire)")
    h(f"    Actual 1       {fn:>6}         {tp:>6}    (fire)")
    h(f"    Fires caught: {tp}/{y_true.sum()} ({tp/y_true.sum():.0%})"
      f"   False alarms: {fp}/{(y_true==0).sum()} ({fp/(y_true==0).sum():.1%})")

# Classification reports
h("\n\n  CLASSIFICATION REPORTS  (threshold = 0.5)")
h("  " + "-" * 62)
for name, prob in models.items():
    y_pred = (prob >= 0.5).astype(int)
    h(f"\n  {name}")
    report = classification_report(
        y_true, y_pred,
        target_names=["no fire", "fire"],
        zero_division=0,
    )
    for line in report.splitlines():
        h("    " + line)

# Key observations
h("\n\n  KEY OBSERVATIONS FOR TASK 1B")
h("  " + "-" * 62)

best_f1_row = metrics_df.loc[metrics_df["f1"].idxmax()]
best_auc_row = metrics_df.loc[metrics_df["auc_roc"].idxmax()]

h(f"\n  Best F1       : {best_f1_row['model']} at threshold {best_f1_row['threshold']}"
  f"  (F1={best_f1_row['f1']:.4f})")
h(f"  Best AUC-ROC  : {best_auc_row['model']}"
  f"  (AUC={best_auc_row['auc_roc']:.4f})")
h("")
h("  For insurance/risk applications, recall is the priority metric:")
h("  a missed fire zip (false negative) means an underpriced policy;")
h("  a false alarm (false positive) means a marginally higher premium.")
h("")
h("  QML benchmark target: F1 > {:.4f}, AUC-ROC > {:.4f}".format(
    metrics_df[metrics_df["threshold"] == 0.5]["f1"].max(),
    metrics_df["auc_roc"].max(),
))

h("\n" + "=" * 64)

report_text = "\n".join(lines)
print(report_text)

# Save report
report_path = OUTPUT_DIR / "comparison_report.txt"
report_path.write_text(report_text)
print(f"\nSaved: {report_path}")

# Save metrics CSV
metrics_df.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/comparison_metrics.csv")


# Precision-recall curve 

print("\nGenerating precision-recall curve...")

fig, ax = plt.subplots(figsize=(7, 5))

colors = {"LogisticRegression": "#1f77b4", "XGBoost": "#ff7f0e", "QML (VQC)": "#2ca02c"}

for name, prob in models.items():
    prec, rec, _ = precision_recall_curve(y_true, prob)
    pr_auc = auc(rec, prec)
    color  = colors.get(name, "#9467bd")
    ax.plot(rec, prec, label=f"{name}  (PR-AUC={pr_auc:.3f})", color=color, lw=2)

# Baseline: random classifier
baseline = y_true.mean()
ax.axhline(baseline, linestyle="--", color="gray", lw=1,
           label=f"Random classifier  ({baseline:.3f})")

# Mark the operating points at threshold=0.5
for name, prob in models.items():
    y_pred = (prob >= 0.5).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    color = colors.get(name, "#9467bd")
    ax.scatter(r, p, s=80, color=color, zorder=5)
    ax.annotate(f"t=0.5", (r, p), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color=color)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve -- 2022 Validation Set", fontsize=13)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
fig.tight_layout()

plot_path = OUTPUT_DIR / "pr_curve.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot_path}")

print("\nDone. Results are in the output/ directory.")
print("To include QML results, re-run with:")
print("  python compare_models.py --qml output/qml_val_predictions.csv")