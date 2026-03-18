#!/usr/bin/env python3
"""Step 4: train a PennyLane quantum-kernel estimator + SVM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from baseline_utils import compute_metrics_at_threshold, threshold_sweep
from sklearn.svm import SVC

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PennyLane is required for QKE. Install with `pip install pennylane`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="qml/qml_train.csv")
    parser.add_argument("--val", default="qml/qml_val.csv")
    parser.add_argument("--predict-2023", default="qml/qml_predict_2023.csv")
    parser.add_argument("--label-col", default="wildfire")
    parser.add_argument("--feature-cols", default=None)
    parser.add_argument("--output-dir", default="outputs/quantum/qke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=800)
    parser.add_argument("--svm-c", type=float, default=1.0)
    return parser.parse_args()


def resolve_feature_cols(df: pd.DataFrame, label_col: str, feature_cols_arg: str | None) -> list[str]:
    if feature_cols_arg:
        cols = [c.strip() for c in feature_cols_arg.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from data: {missing}")
        return cols
    return [c for c in df.columns if c not in {"zip", "year", label_col}]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    pred_df = pd.read_csv(args.predict_2023)

    if args.label_col not in train_df.columns or args.label_col not in val_df.columns:
        raise ValueError(f"Missing label column '{args.label_col}' in train/val.")

    feature_cols = resolve_feature_cols(train_df, args.label_col, args.feature_cols)

    X_train_full = train_df[feature_cols].to_numpy(dtype=float)
    y_train_full = train_df[args.label_col].to_numpy(dtype=int)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df[args.label_col].to_numpy(dtype=int)
    X_pred = pred_df[feature_cols].to_numpy(dtype=float)

    if len(X_train_full) > args.max_train_samples:
        idx = rng.choice(len(X_train_full), size=args.max_train_samples, replace=False)
        X_train = X_train_full[idx]
        y_train = y_train_full[idx]
        sampled_train = train_df.iloc[idx].reset_index(drop=True)
    else:
        X_train = X_train_full
        y_train = y_train_full
        sampled_train = train_df.reset_index(drop=True)

    n_qubits = X_train.shape[1]
    if n_qubits == 0:
        raise ValueError("No feature columns found for QKE.")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
        return qml.probs(wires=range(n_qubits))

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        return float(kernel_circuit(x1, x2)[0])

    def kernel_matrix(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        K = np.zeros((len(Xa), len(Xb)), dtype=float)
        for i in range(len(Xa)):
            for j in range(len(Xb)):
                K[i, j] = kernel(Xa[i], Xb[j])
        return K

    K_train = kernel_matrix(X_train, X_train)
    K_val = kernel_matrix(X_val, X_train)
    K_pred = kernel_matrix(X_pred, X_train)

    clf = SVC(kernel="precomputed", probability=True, class_weight="balanced", C=args.svm_c, random_state=args.seed)
    clf.fit(K_train, y_train)

    val_prob = clf.predict_proba(K_val)[:, 1]
    sweep_df, best_f1_thr, best_prec_rec50_thr = threshold_sweep(y_val, val_prob)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    metrics_best_f1 = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)
    metrics_best_prec_rec50 = compute_metrics_at_threshold(y_val, val_prob, best_prec_rec50_thr)

    val_out = val_df[["zip", "year", args.label_col]].copy()
    val_out["wildfire_prob"] = val_prob
    val_out["wildfire_pred"] = (val_prob >= best_f1_thr).astype(int)
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob = clf.predict_proba(K_pred)[:, 1]
    pred_out = pred_df[["zip", "year"]].copy()
    pred_out["wildfire_prob"] = pred_prob
    pred_out["wildfire_pred"] = (pred_prob >= best_f1_thr).astype(int)
    pred_out.to_csv(out_dir / "predict_2023_predictions.csv", index=False)

    sampled_train[["zip", "year", args.label_col]].to_csv(out_dir / "train_sample_used.csv", index=False)

    metrics = {
        "model": "PennyLaneQKE+SVC",
        "feature_cols": feature_cols,
        "n_qubits": n_qubits,
        "seed": args.seed,
        "svm_c": args.svm_c,
        "n_train_used": int(len(X_train)),
        "n_train_full": int(len(X_train_full)),
        "n_val": int(len(X_val)),
        "n_predict_2023": int(len(X_pred)),
        "threshold_selection": {
            "selected_for_predictions": "max_f1",
            "best_f1_threshold": float(best_f1_thr),
            "best_precision_with_recall_ge_0_5_threshold": float(best_prec_rec50_thr),
            "recall_constraint_feasible": bool((sweep_df["recall"] >= 0.5).any()),
        },
        "metrics_val_best_f1": metrics_best_f1,
        "metrics_val_best_precision_recall_ge_0_5": metrics_best_prec_rec50,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved QKE artifacts to: {out_dir}")
    print(json.dumps(metrics_best_f1, indent=2))


if __name__ == "__main__":
    main()
