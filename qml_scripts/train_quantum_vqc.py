"""train a PennyLane variational quantum classifier (VQC)."""

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
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PennyLane is required for VQC. Install with `pip install pennylane`."
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

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x: pnp.ndarray, weights: pnp.ndarray) -> pnp.ndarray:
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / pos if pos > 0 else 1.0

    def predict_scores(X: np.ndarray, weights: pnp.ndarray) -> np.ndarray:
        vals = [circuit(x, weights) for x in X]
        vals = np.asarray(vals, dtype=float)
        return (vals + 1.0) / 2.0

    def weighted_bce(probs: pnp.ndarray, y_true: pnp.ndarray) -> pnp.ndarray:
        eps = 1e-8
        w = pnp.where(y_true > 0.5, pos_weight, 1.0)
        return -pnp.mean(w * (y_true * pnp.log(probs + eps) + (1 - y_true) * pnp.log(1 - probs + eps)))

    def loss(weights: pnp.ndarray, Xb: pnp.ndarray, yb: pnp.ndarray) -> pnp.ndarray:
        logits = pnp.stack([circuit(x, weights) for x in Xb])
        probs = (logits + 1.0) / 2.0
        return weighted_bce(probs, yb)

    weights = pnp.array(0.01 * rng.normal(size=(args.n_layers, n_qubits, 3)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=args.learning_rate)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        batch_idx = rng.choice(len(X_train), size=min(args.batch_size, len(X_train)), replace=False)
        Xb = pnp.array(X_train[batch_idx], requires_grad=False)
        yb = pnp.array(y_train[batch_idx], requires_grad=False)
        weights, batch_loss = opt.step_and_cost(lambda w: loss(w, Xb, yb), weights)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val_prob_epoch = predict_scores(X_val, weights)
            # Monitor at the insurance operating threshold (recall-priority, t=0.3)
            # rather than max-F1, so the training history reflects the actual objective.
            ins_metrics = compute_metrics_at_threshold(y_val, val_prob_epoch, OPERATING_THRESHOLD)
            history.append(
                {
                    "epoch": epoch,
                    "batch_loss": float(batch_loss),
                    "val_f1": float(ins_metrics["f1"]),
                    "val_precision": float(ins_metrics["precision"]),
                    "val_recall": float(ins_metrics["recall"]),
                    "threshold": OPERATING_THRESHOLD,
                }
            )

    val_prob = predict_scores(X_val, weights)
    sweep_df, best_f1_thr, best_prec_rec50_thr = threshold_sweep(y_val, val_prob)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Insurance objective: recall-priority threshold (t=0.3).
    # Max-F1 (~0.8) sacrifices ~40% recall — missing fire zip codes that become
    # underpriced policies. t=0.3 is the recommended operating point from the
    # classical baseline evaluation (93% recall, 52% false alarm rate for LR).
    metrics_operating = compute_metrics_at_threshold(y_val, val_prob, OPERATING_THRESHOLD)
    # Also compute max-F1 metrics for the full sweep comparison / Task 1B writeup.
    metrics_best_f1 = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)

    val_out = val_df[["zip", "year", args.label_col]].copy()
    val_out["wildfire_prob"] = val_prob
    val_out["wildfire_pred"] = (val_prob >= OPERATING_THRESHOLD).astype(int)
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob = predict_scores(X_pred, weights)
    pred_out = pred_df[["zip", "year"]].copy()
    pred_out["wildfire_prob"] = pred_prob
    pred_out["wildfire_pred"] = (pred_prob >= OPERATING_THRESHOLD).astype(int)
    pred_out.to_csv(out_dir / "predict_2023_predictions.csv", index=False)

    np.savez(out_dir / "model_params.npz", weights=np.asarray(weights))
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    metrics = {
        "model": "PennyLaneVQC",
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

    print(f"Saved VQC artifacts to: {out_dir}")
    print(f"\n--- Metrics at operating threshold (t={OPERATING_THRESHOLD}, insurance recall-priority) ---")
    print(json.dumps(metrics_operating, indent=2))
    print(f"\n--- Metrics at best-F1 threshold (t={best_f1_thr:.2f}, for Task 1B comparison) ---")
    print(json.dumps(metrics_best_f1, indent=2))


if __name__ == "__main__":
    main()
