"""train a PennyLane quantum-kernel estimator + SVM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PennyLane is required for QKE. Install with `pip install pennylane`."
    ) from exc


# Insurance recall-priority operating threshold.
# Matches the classical baseline recommendation (3_classical_baseline.md §5):
# at t=0.3, LR catches 93% of fire zip codes while flagging 52% of non-fire zips.
# Max-F1 (~0.8) sacrifices 40% recall, which an insurer cannot accept.
OPERATING_THRESHOLD = 0.3


def compute_metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_predicted_positive": int(y_pred.sum()),
    }


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[pd.DataFrame, float, float]:
    rows = []
    for thr in np.round(np.arange(0.01, 1.00, 0.01), 2):
        rows.append(compute_metrics_at_threshold(y_true, y_prob, float(thr)))

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_f1_row = sweep_df.sort_values(
        ["f1", "recall", "precision"], ascending=False
    ).iloc[0]
    best_f1_threshold = float(best_f1_row["threshold"])

    feasible = sweep_df[sweep_df["recall"] >= 0.5]
    if len(feasible):
        best_prec_row = feasible.sort_values(["precision", "f1"], ascending=False).iloc[0]
        best_precision_recall50_threshold = float(best_prec_row["threshold"])
    else:
        best_precision_recall50_threshold = best_f1_threshold

    return sweep_df, best_f1_threshold, best_precision_recall50_threshold


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
        # Stratified subsample: the training set is ~9% positive. A random draw of
        # 800 rows yields only ~72 positives on average — the SVM kernel matrix will
        # barely see the minority class. Preserve the positive rate explicitly.
        pos_idx = np.where(y_train_full == 1)[0]
        neg_idx = np.where(y_train_full == 0)[0]
        n_pos = max(1, round(args.max_train_samples * len(pos_idx) / len(y_train_full)))
        n_neg = args.max_train_samples - n_pos
        chosen_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        chosen_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(idx)
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
        # IQP-style feature map: AngleEmbedding followed by ZZ-type entanglement
        # (CZ + second rotation layer). This is the standard quantum kernel construction
        # used in Havlíček et al. (2019). Without entanglement, the kernel reduces to
        # a classical dot product — the quantum circuit adds no advantage.
        #
        # U(x)|0⟩: encode x1 with entanglement
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
        qml.RZ(x1[0] * x1[-1], wires=0)   # cross-feature interaction term
        #
        # U†(x2)|: apply adjoint encoding of x2 with same entanglement structure
        qml.RZ(-(x2[0] * x2[-1]), wires=0)
        for i in reversed(range(n_qubits - 1)):
            qml.CZ(wires=[i, i + 1])
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
        #
        # K(x1, x2) = |⟨0|U†(x2)U(x1)|0⟩|² = prob of measuring all-zeros state
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

    # Insurance objective: recall-priority threshold (t=0.3).
    # Max-F1 (~0.8) sacrifices ~40% recall — unacceptable for an insurer who needs
    # to catch fire zip codes to avoid underpriced policies.
    metrics_operating = compute_metrics_at_threshold(y_val, val_prob, OPERATING_THRESHOLD)
    # Also compute max-F1 metrics for the full sweep comparison / Task 1B writeup.
    metrics_best_f1 = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)

    val_out = val_df[["zip", "year", args.label_col]].copy()
    val_out["wildfire_prob"] = val_prob
    val_out["wildfire_pred"] = (val_prob >= OPERATING_THRESHOLD).astype(int)
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob = clf.predict_proba(K_pred)[:, 1]
    pred_out = pred_df[["zip", "year"]].copy()
    pred_out["wildfire_prob"] = pred_prob
    pred_out["wildfire_pred"] = (pred_prob >= OPERATING_THRESHOLD).astype(int)
    pred_out.to_csv(out_dir / "predict_2023_predictions.csv", index=False)

    sampled_train[["zip", "year", args.label_col]].to_csv(out_dir / "train_sample_used.csv", index=False)

    metrics = {
        "model": "PennyLaneQKE+SVC",
        "feature_cols": feature_cols,
        "n_qubits": n_qubits,
        "kernel": "IQP-AngleEmbedding+CZ+cross-term",
        "seed": args.seed,
        "svm_c": args.svm_c,
        "n_train_used": int(len(X_train)),
        "n_train_full": int(len(X_train_full)),
        "n_val": int(len(X_val)),
        "n_predict_2023": int(len(X_pred)),
        "threshold_selection": {
            # t=0.3 is the insurance operating threshold: maximise recall (catching fire
            # zip codes) while keeping false alarm rate at an acceptable level.
            # Matches the classical baseline recommended threshold from 3_classical_baseline.md.
            "operating_threshold": OPERATING_THRESHOLD,
            "selected_for_predictions": "insurance_recall_priority",
            "best_f1_threshold": float(best_f1_thr),
            "best_precision_with_recall_ge_0_5_threshold": float(best_prec_rec50_thr),
            "recall_constraint_feasible": bool((sweep_df["recall"] >= 0.5).any()),
        },
        "metrics_val_operating_threshold": metrics_operating,
        "metrics_val_best_f1": metrics_best_f1,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved QKE artifacts to: {out_dir}")
    print(f"\n--- Metrics at operating threshold (t={OPERATING_THRESHOLD}, insurance recall-priority) ---")
    print(json.dumps(metrics_operating, indent=2))
    print(f"\n--- Metrics at best-F1 threshold (t={best_f1_thr:.2f}, for Task 1B comparison) ---")
    print(json.dumps(metrics_best_f1, indent=2))


if __name__ == "__main__":
    main()
