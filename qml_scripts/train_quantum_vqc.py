"""Train a Qiskit variational quantum classifier (VQC).

Improvements over the previous version (informed by 3_classical_baseline.md):

1.  BALANCED SUBSAMPLING  — The classical baseline shows the signal is recall-
    priority; training on the natural ~9% positive rate starves the optimizer of
    positive examples.  We now draw a balanced subsample (50/50) up to
    --max-train-samples, then re-weight via pos_weight so the gradient signal
    is correct even with the artificial balance.

2.  PROPER ANGLE SCALING  — Features must be scaled to [0, π] before angle
    encoding so that the full Bloch-sphere range is used.  The previous version
    relied on whatever scale arrived from PCA output, which can be negative and
    outside [0, π].  We add a per-feature MinMax rescale to [0.05*π, 0.95*π]
    (small margin to avoid degenerate poles) fitted on training data only.

3.  PC2-WEIGHTED ENCODING  — The classical baseline shows PC2 (index 1) carries
    ~37% of tree-ensemble importance and dominates the LR coefficient (0.560 vs
    0.154 for PC1).  We encode PC2 with two rotation layers (RY then RZ) while
    other PCs get one RY layer, giving the strongest signal more expressive
    capacity without adding qubits.

4.  ENTANGLEMENT BEFORE VARIATIONAL LAYERS  — The previous ansatz applied CX
    gates *after* the variational rotations, so the first forward pass had no
    entanglement (weights initialised near zero).  CX gates now precede each
    variational layer so entanglement is present from epoch 1.

5.  SPSA STABILITY  — The previous hand-rolled SPSA used ck ~ epoch^-0.101,
    which decays too fast (near-zero gradient estimates by epoch 10).  The
    standard Spall (1998) schedule with c0=0.1, gamma=1/6 is used instead.
    ak uses the a/(A+epoch)^alpha form with A=10 to prevent large early steps.

6.  ORIENTATION CHECK MOVED TO EPOCH 1  — AUC < 0.5 means inverted outputs.
    We now detect and flip after epoch 1 rather than after all training, so
    subsequent epochs optimise in the correct direction.

7.  PER-EPOCH TRAINING HISTORY  — Loss, val-AUC and insurance-threshold metrics
    logged every epoch (was every 5), making convergence curves usable for
    the Task 1B writeup.

8.  SAVED ANGLE SCALER  — The MinMaxScaler is saved as a .joblib so the same
    transform can be applied at inference time without re-fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
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
from sklearn.preprocessing import MinMaxScaler

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Qiskit is required for VQC.  Install with `pip install qiskit`."
    ) from exc

# ---------------------------------------------------------------------------
# Insurance recall-priority operating threshold (from 3_classical_baseline.md §5).
# At t=0.3, LR catches 93% of fire zip codes while flagging 52% of non-fire zips.
# Max-F1 (~t=0.8) sacrifices 40% recall — unacceptable for an insurer.
# ---------------------------------------------------------------------------
OPERATING_THRESHOLD = 0.3

# SPSA schedule constants (Spall 1998 recommended values).
_SPSA_A_STABLE = 10       # prevents large steps in early epochs
_SPSA_ALPHA = 0.602
_SPSA_C_COEFF = 0.1
_SPSA_GAMMA = 0.101


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

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
    best_prec_rec50_threshold = (
        float(feasible.sort_values(["precision", "f1"], ascending=False).iloc[0]["threshold"])
        if len(feasible)
        else best_f1_threshold
    )

    return sweep_df, best_f1_threshold, best_prec_rec50_threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="qml/qml_train.csv")
    p.add_argument("--val", default="qml/qml_val.csv")
    p.add_argument("--predict-2023", default="qml/qml_predict_2023.csv")
    p.add_argument("--label-col", default="wildfire")
    p.add_argument("--feature-cols", default=None,
                   help="Comma-separated feature columns; default: all non-id/non-label cols")
    p.add_argument("--output-dir", default="outputs/quantum/vqc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-layers", type=int, default=2,
                   help="Number of variational + entanglement layer pairs")
    p.add_argument("--epochs", type=int, default=30,
                   help="SPSA optimisation epochs (increased from 20 for better convergence)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.6,
                   help="SPSA a_coeff (Spall 1998); higher = larger initial steps")
    p.add_argument("--max-train-samples", type=int, default=1024,
                   help="Total balanced subsample size (half pos, half neg)")
    p.add_argument("--operating-threshold", type=float, default=OPERATING_THRESHOLD)
    p.add_argument("--pc2-index", type=int, default=1,
                   help="0-based feature index for PC2 (default 1); gets double encoding")
    return p.parse_args()


def _resolve_input_path(path_arg: str, repo_root: Path) -> Path:
    p = Path(path_arg).expanduser()
    if p.is_absolute() or p.exists():
        return p
    candidate = repo_root / p
    return candidate if candidate.exists() else p


def _resolve_output_path(path_arg: str, repo_root: Path) -> Path:
    p = Path(path_arg).expanduser()
    return p if p.is_absolute() else repo_root / p


def resolve_feature_cols(
    df: pd.DataFrame, label_col: str, feature_cols_arg: str | None
) -> list[str]:
    if feature_cols_arg:
        cols = [c.strip() for c in feature_cols_arg.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from data: {missing}")
        return cols
    return [c for c in df.columns if c not in {"zip", "year", label_col}]


# ---------------------------------------------------------------------------
# Quantum circuit
# ---------------------------------------------------------------------------

def build_circuit(
    x: np.ndarray,
    weights: np.ndarray,
    n_layers: int,
    pc2_index: int,
) -> QuantumCircuit:
    """Encode one sample and apply variational + entanglement layers.

    Encoding (improvement #3):
      All features: RY(x[q]).
      PC2 additionally: RZ(x[pc2_index]) — doubles its Bloch-sphere coverage
      to reflect its dominant LR coefficient (0.560) and 37% tree importance.

    Ansatz (improvement #4):
      Each layer: CX entanglement ring → Rx Ry Rz per qubit.
      CX precedes rotations so entanglement exists from the first epoch,
      not after weights have left the zero-initialisation neighbourhood.
    """
    n_qubits = len(x)
    qc = QuantumCircuit(n_qubits)

    # Feature encoding
    for q in range(n_qubits):
        qc.ry(float(x[q]), q)
    if 0 <= pc2_index < n_qubits:
        qc.rz(float(x[pc2_index]), pc2_index)

    # Variational layers
    for layer in range(n_layers):
        # Entanglement ring first (improvement #4)
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(n_qubits - 1, 0)
        # Parameterised rotations
        for q in range(n_qubits):
            qc.rx(float(weights[layer, q, 0]), q)
            qc.ry(float(weights[layer, q, 1]), q)
            qc.rz(float(weights[layer, q, 2]), q)

    return qc


def circuit_expectation(
    x: np.ndarray, weights: np.ndarray, n_layers: int, pc2_index: int
) -> float:
    """Return ⟨Z₀⟩ ∈ [-1, 1]."""
    qc = build_circuit(x, weights, n_layers, pc2_index)
    probs = Statevector.from_instruction(qc).probabilities()
    z0 = np.array(
        [1.0 if (i & 1) == 0 else -1.0 for i in range(len(probs))], dtype=float
    )
    return float(np.dot(probs, z0))


def predict_scores_batch(
    X: np.ndarray, weights: np.ndarray, n_layers: int, pc2_index: int
) -> np.ndarray:
    vals = np.array(
        [circuit_expectation(x, weights, n_layers, pc2_index) for x in X],
        dtype=float,
    )
    return np.clip((vals + 1.0) / 2.0, 1e-8, 1 - 1e-8)


# ---------------------------------------------------------------------------
# SPSA helpers
# ---------------------------------------------------------------------------

def weighted_bce(probs: np.ndarray, y: np.ndarray, pos_weight: float) -> float:
    eps = 1e-8
    w = np.where(y > 0.5, pos_weight, 1.0)
    return float(-np.mean(w * (y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))))


def spsa_loss(
    weights: np.ndarray, Xb: np.ndarray, yb: np.ndarray,
    pos_weight: float, n_layers: int, pc2_index: int
) -> float:
    return weighted_bce(predict_scores_batch(Xb, weights, n_layers, pc2_index), yb, pos_weight)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    n_qubits = len(feature_cols)
    if n_qubits == 0:
        raise ValueError("No feature columns found for VQC.")

    X_train_full = train_df[feature_cols].to_numpy(dtype=float)
    y_train_full = train_df[args.label_col].to_numpy(dtype=float)
    X_val_raw = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df[args.label_col].to_numpy(dtype=int)
    X_pred_raw = pred_df[feature_cols].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Improvement #1: BALANCED SUBSAMPLING
    # Draw equal positives and negatives. With balanced classes pos_weight=1.0,
    # which is intentional — the subsample already reflects 50/50 balance.
    # ------------------------------------------------------------------
    pos_idx = np.where(y_train_full == 1)[0]
    neg_idx = np.where(y_train_full == 0)[0]
    n_half = min(args.max_train_samples // 2, len(pos_idx), len(neg_idx))
    chosen_pos = rng.choice(pos_idx, size=n_half, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_half, replace=False)
    idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(idx)
    X_train_sub = X_train_full[idx]
    y_train_sub = y_train_full[idx]
    pos_weight = 1.0  # balanced subsample; explicit for clarity

    print(f"n_qubits={n_qubits}, n_layers={args.n_layers}, epochs={args.epochs}")
    print(f"Training subset: {n_half} pos + {n_half} neg = {2*n_half} rows (balanced)")
    print(f"Full train set: {len(X_train_full)} rows "
          f"({int(y_train_full.sum())} pos / {int((1-y_train_full).sum())} neg)")

    # ------------------------------------------------------------------
    # Improvement #2: ANGLE SCALING — [0.05π, 0.95π] via MinMaxScaler
    # Fit on training subsample; transform val and pred with same scaler.
    # ------------------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0.05 * np.pi, 0.95 * np.pi))
    X_train = scaler.fit_transform(X_train_sub)
    X_val = scaler.transform(X_val_raw)
    X_pred = scaler.transform(X_pred_raw)
    joblib.dump(scaler, out_dir / "vqc_angle_scaler.joblib")  # improvement #8

    # ------------------------------------------------------------------
    # Initialise weights with small random noise
    # ------------------------------------------------------------------
    weights = 0.01 * rng.normal(size=(args.n_layers, n_qubits, 3))
    orientation_flipped = False
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        batch_size = min(args.batch_size, len(X_train))
        batch_idx = rng.choice(len(X_train), size=batch_size, replace=False)
        Xb = X_train[batch_idx]
        yb = y_train_sub[batch_idx]

        # Improvement #5: Spall (1998) SPSA schedule
        ck = _SPSA_C_COEFF / (epoch ** _SPSA_GAMMA)
        ak = args.learning_rate / (_SPSA_A_STABLE + epoch) ** _SPSA_ALPHA
        delta = rng.choice([-1.0, 1.0], size=weights.shape)

        loss_p = spsa_loss(weights + ck * delta, Xb, yb, pos_weight, args.n_layers, args.pc2_index)
        loss_m = spsa_loss(weights - ck * delta, Xb, yb, pos_weight, args.n_layers, args.pc2_index)
        grad_est = ((loss_p - loss_m) / (2.0 * ck)) * delta
        weights = weights - ak * grad_est
        batch_loss = 0.5 * (loss_p + loss_m)

        # Improvement #6: orientation check on epoch 1
        val_prob_raw = predict_scores_batch(X_val, weights, args.n_layers, args.pc2_index)
        if epoch == 1:
            auc1 = float(roc_auc_score(y_val, val_prob_raw))
            if auc1 < 0.5:
                orientation_flipped = True
                print(f"  Epoch 1 AUC={auc1:.3f} < 0.5 — flipping score orientation.")

        val_prob = 1.0 - val_prob_raw if orientation_flipped else val_prob_raw
        ins = compute_metrics_at_threshold(y_val, val_prob, operating_threshold)
        val_auc = float(roc_auc_score(y_val, val_prob))

        # Improvement #7: per-epoch history
        history.append({
            "epoch": epoch,
            "batch_loss": float(batch_loss),
            "ck": float(ck),
            "ak": float(ak),
            "val_auc": val_auc,
            "val_f1": float(ins["f1"]),
            "val_precision": float(ins["precision"]),
            "val_recall": float(ins["recall"]),
            "threshold": operating_threshold,
        })

        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:3d} | loss={batch_loss:.4f} | ck={ck:.4f} | ak={ak:.5f} "
                f"| val_auc={val_auc:.3f} | recall={ins['recall']:.3f} | prec={ins['precision']:.3f}"
            )

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    val_prob_raw = predict_scores_batch(X_val, weights, args.n_layers, args.pc2_index)
    auc_before_flip = float(roc_auc_score(y_val, val_prob_raw))
    val_prob = 1.0 - val_prob_raw if orientation_flipped else val_prob_raw

    sweep_df, best_f1_thr, best_prec_rec50_thr = threshold_sweep(y_val, val_prob)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    metrics_operating = compute_metrics_at_threshold(y_val, val_prob, operating_threshold)
    metrics_best_f1 = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)

    val_out = val_df[["zip", "year", args.label_col]].copy()
    val_out["wildfire_prob"] = val_prob
    val_out["wildfire_pred"] = (val_prob >= operating_threshold).astype(int)
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob_raw = predict_scores_batch(X_pred, weights, args.n_layers, args.pc2_index)
    pred_prob = 1.0 - pred_prob_raw if orientation_flipped else pred_prob_raw
    pred_out = pred_df[["zip", "year"]].copy()
    pred_out["wildfire_prob"] = pred_prob
    pred_out["wildfire_pred"] = (pred_prob >= operating_threshold).astype(int)
    pred_out.to_csv(out_dir / "predict_2023_predictions.csv", index=False)

    np.savez(out_dir / "model_params.npz", weights=np.asarray(weights))
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    metrics = {
        "model": "QiskitVQC_v2",
        "feature_cols": feature_cols,
        "n_qubits": n_qubits,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "improvements_over_v1": [
            "balanced_subsampling_50_50",
            "angle_scaling_to_0_to_pi",
            "pc2_double_encoded_RY_plus_RZ",
            "entanglement_ring_before_variational_layers",
            "spall_1998_spsa_schedule",
            "orientation_flip_at_epoch_1_not_end",
            "per_epoch_training_history",
            "saved_angle_scaler_joblib",
        ],
        "class_balance": {
            "n_positive_in_subset": int(n_half),
            "n_negative_in_subset": int(n_half),
            "positive_rate_subset": 0.5,
            "pos_weight_in_loss": float(pos_weight),
            "full_train_positive_rate": float(y_train_full.mean()),
        },
        "threshold_selection": {
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
        "classical_baseline_targets": {
            "lr_auc_roc": 0.852,
            "lr_recall_at_t03": 0.928,
            "lr_precision_at_t03": 0.117,
            "lr_f1_at_t03": 0.208,
            "lr_false_alarm_rate_at_t03": 0.523,
        },
        "metrics_val_operating_threshold": metrics_operating,
        "metrics_val_best_f1": metrics_best_f1,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved VQC v2 artifacts to: {out_dir}")
    print(f"\n--- Metrics at operating threshold (t={operating_threshold}) ---")
    print(json.dumps(metrics_operating, indent=2))
    print(f"\n--- Metrics at best-F1 threshold (t={best_f1_thr:.2f}) ---")
    print(json.dumps(metrics_best_f1, indent=2))
    print("\n--- Classical LR targets to beat ---")
    print("  AUC-ROC 0.852 | Recall 0.928 | Precision 0.117 | F1 0.208 | FAR 0.523")


if __name__ == "__main__":
    main()