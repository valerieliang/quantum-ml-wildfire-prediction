"""train a Qiskit variational quantum classifier (VQC)."""

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

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Qiskit is required for VQC. Install with `pip install qiskit`."
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
    parser.add_argument("--output-dir", default="outputs/quantum/vqc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--operating-threshold", type=float, default=OPERATING_THRESHOLD)
    return parser.parse_args()


def _resolve_input_path(path_arg: str, repo_root: Path) -> Path:
    p = Path(path_arg).expanduser()
    if p.is_absolute() or p.exists():
        return p
    candidate = repo_root / p
    return candidate if candidate.exists() else p


def _resolve_output_path(path_arg: str, repo_root: Path) -> Path:
    p = Path(path_arg).expanduser()
    if p.is_absolute():
        return p
    return repo_root / p


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
    operating_threshold = float(args.operating_threshold)
    repo_root = Path(__file__).resolve().parents[1]
    train_path = _resolve_input_path(args.train, repo_root)
    val_path = _resolve_input_path(args.val, repo_root)
    predict_path = _resolve_input_path(args.predict_2023, repo_root)
    out_dir = _resolve_output_path(args.output_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    pred_df = pd.read_csv(predict_path)

    if args.label_col not in train_df.columns or args.label_col not in val_df.columns:
        raise ValueError(f"Missing label column '{args.label_col}' in train/val.")

    feature_cols = resolve_feature_cols(train_df, args.label_col, args.feature_cols)

    X_train_full = train_df[feature_cols].to_numpy(dtype=float)
    y_train_full = train_df[args.label_col].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df[args.label_col].to_numpy(dtype=int)
    X_pred = pred_df[feature_cols].to_numpy(dtype=float)

    if len(X_train_full) > args.max_train_samples:
        # Stratified subsample: preserve the ~9% positive rate from the full training set.
        # A purely random draw of 1024 rows would yield only ~93 positives on average,
        # making pos_weight correction noisier than necessary.
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
    else:
        X_train = X_train_full
        y_train = y_train_full

    n_qubits = X_train.shape[1]
    if n_qubits == 0:
        raise ValueError("No feature columns found for VQC.")

    def circuit_expectation(x: np.ndarray, weights: np.ndarray) -> float:
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.ry(float(x[q]), q)
        for layer in range(args.n_layers):
            for q in range(n_qubits):
                qc.rx(float(weights[layer, q, 0]), q)
                qc.ry(float(weights[layer, q, 1]), q)
                qc.rz(float(weights[layer, q, 2]), q)
            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
                qc.cx(n_qubits - 1, 0)
        probs = Statevector.from_instruction(qc).probabilities()
        z0 = np.array([1.0 if (idx & 1) == 0 else -1.0 for idx in range(len(probs))], dtype=float)
        return float(np.dot(probs, z0))

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / pos if pos > 0 else 1.0

    def predict_scores(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        vals = [circuit_expectation(x, weights) for x in X]
        vals = np.asarray(vals, dtype=float)
        return np.clip((vals + 1.0) / 2.0, 1e-8, 1 - 1e-8)

    def weighted_bce(probs: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-8
        w = np.where(y_true > 0.5, pos_weight, 1.0)
        return float(-np.mean(w * (y_true * np.log(probs + eps) + (1 - y_true) * np.log(1 - probs + eps))))

    def loss(weights: np.ndarray, Xb: np.ndarray, yb: np.ndarray) -> float:
        logits = np.array([circuit_expectation(x, weights) for x in Xb], dtype=float)
        probs = np.clip((logits + 1.0) / 2.0, 1e-8, 1 - 1e-8)
        return weighted_bce(probs, yb)

    weights = 0.01 * rng.normal(size=(args.n_layers, n_qubits, 3))

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        batch_idx = rng.choice(len(X_train), size=min(args.batch_size, len(X_train)), replace=False)
        Xb = X_train[batch_idx]
        yb = y_train[batch_idx]

        # Lightweight SPSA update to avoid autograd dependencies.
        delta = rng.choice([-1.0, 1.0], size=weights.shape)
        ck = 0.1 / (epoch ** 0.101)
        ak = args.learning_rate / np.sqrt(epoch)
        loss_plus = loss(weights + ck * delta, Xb, yb)
        loss_minus = loss(weights - ck * delta, Xb, yb)
        grad_est = ((loss_plus - loss_minus) / (2.0 * ck)) * delta
        weights = weights - ak * grad_est
        batch_loss = 0.5 * (loss_plus + loss_minus)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val_prob_epoch = predict_scores(X_val, weights)
            # Monitor at the insurance operating threshold (recall-priority, t=0.3)
            # rather than max-F1, so the training history reflects the actual objective.
            ins_metrics = compute_metrics_at_threshold(y_val, val_prob_epoch, operating_threshold)
            history.append(
                {
                    "epoch": epoch,
                    "batch_loss": float(batch_loss),
                    "val_f1": float(ins_metrics["f1"]),
                    "val_precision": float(ins_metrics["precision"]),
                    "val_recall": float(ins_metrics["recall"]),
                    "threshold": operating_threshold,
                }
            )

    val_prob_raw = predict_scores(X_val, weights)
    auc_before_flip = float(roc_auc_score(y_val, val_prob_raw))
    orientation_flipped = auc_before_flip < 0.5
    val_prob = 1.0 - val_prob_raw if orientation_flipped else val_prob_raw
    sweep_df, best_f1_thr, best_prec_rec50_thr = threshold_sweep(y_val, val_prob)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Insurance objective: recall-priority threshold (t=0.3).
    # Max-F1 (~0.8) sacrifices ~40% recall — missing fire zip codes that become
    # underpriced policies. t=0.3 is the recommended operating point from the
    # classical baseline evaluation (93% recall, 52% false alarm rate for LR).
    metrics_operating = compute_metrics_at_threshold(y_val, val_prob, operating_threshold)
    # Also compute max-F1 metrics for the full sweep comparison / Task 1B writeup.
    metrics_best_f1 = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)

    val_out = val_df[["zip", "year", args.label_col]].copy()
    val_out["wildfire_prob"] = val_prob
    val_out["wildfire_pred"] = (val_prob >= operating_threshold).astype(int)
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob_raw = predict_scores(X_pred, weights)
    pred_prob = 1.0 - pred_prob_raw if orientation_flipped else pred_prob_raw
    pred_out = pred_df[["zip", "year"]].copy()
    pred_out["wildfire_prob"] = pred_prob
    pred_out["wildfire_pred"] = (pred_prob >= operating_threshold).astype(int)
    pred_out.to_csv(out_dir / "predict_2023_predictions.csv", index=False)

    np.savez(out_dir / "model_params.npz", weights=np.asarray(weights))
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    metrics = {
        "model": "QiskitVQC",
        "feature_cols": feature_cols,
        "n_qubits": n_qubits,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "class_balance_train_subset": {
            "positive_rate": float(y_train.mean()),
            "positive_count": int(y_train.sum()),
            "negative_count": int((1 - y_train).sum()),
            "pos_weight": float(pos_weight),
        },
        "threshold_selection": {
            # t=0.3 is the insurance operating threshold: maximise recall (catching fire
            # zip codes) while keeping false alarm rate at an acceptable level.
            # Matches the classical baseline recommended threshold from 3_classical_baseline.md.
            "operating_threshold": operating_threshold,
            "selected_for_predictions": "insurance_recall_priority",
            "best_f1_threshold": float(best_f1_thr),
            "best_precision_with_recall_ge_0_5_threshold": float(best_prec_rec50_thr),
            "recall_constraint_feasible": bool((sweep_df["recall"] >= 0.5).any()),
        },
        "score_orientation": {
            "auc_before_orientation_check": auc_before_flip,
            "orientation_flipped": orientation_flipped,
        },
        "metrics_val_operating_threshold": metrics_operating,
        "metrics_val_best_f1": metrics_best_f1,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved VQC artifacts to: {out_dir}")
    print(f"\n--- Metrics at operating threshold (t={operating_threshold}, insurance recall-priority) ---")
    print(json.dumps(metrics_operating, indent=2))
    print(f"\n--- Metrics at best-F1 threshold (t={best_f1_thr:.2f}, for Task 1B comparison) ---")
    print(json.dumps(metrics_best_f1, indent=2))


if __name__ == "__main__":
    main()
