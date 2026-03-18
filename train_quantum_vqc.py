#!/usr/bin/env python3
"""Step 4: train a PennyLane variational quantum classifier (VQC)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from baseline_utils import compute_metrics_at_threshold, threshold_sweep

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PennyLane is required for VQC. Install with `pip install pennylane`."
    ) from exc


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
        idx = rng.choice(len(X_train_full), size=args.max_train_samples, replace=False)
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
            sweep_df, best_thr, _ = threshold_sweep(y_val, val_prob_epoch)
            best_metrics = compute_metrics_at_threshold(y_val, val_prob_epoch, best_thr)
            history.append(
                {
                    "epoch": epoch,
                    "batch_loss": float(batch_loss),
                    "val_f1": float(best_metrics["f1"]),
                    "val_precision": float(best_metrics["precision"]),
                    "val_recall": float(best_metrics["recall"]),
                    "best_threshold": float(best_thr),
                }
            )

    val_prob = predict_scores(X_val, weights)
    sweep_df, best_f1_thr, best_prec_rec50_thr = threshold_sweep(y_val, val_prob)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    metrics_best_f1 = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)
    metrics_best_prec_rec50 = compute_metrics_at_threshold(y_val, val_prob, best_prec_rec50_thr)

    val_out = val_df[["zip", "year", args.label_col]].copy()
    val_out["wildfire_prob"] = val_prob
    val_out["wildfire_pred"] = (val_prob >= best_f1_thr).astype(int)
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob = predict_scores(X_pred, weights)
    pred_out = pred_df[["zip", "year"]].copy()
    pred_out["wildfire_prob"] = pred_prob
    pred_out["wildfire_pred"] = (pred_prob >= best_f1_thr).astype(int)
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

    print(f"Saved VQC artifacts to: {out_dir}")
    print(json.dumps(metrics_best_f1, indent=2))


if __name__ == "__main__":
    main()
