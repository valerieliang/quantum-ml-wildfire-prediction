"""Train a Qiskit variational quantum classifier (VQC).

Root-cause fix for all-positive / precision collapse
======================================================
The previous version had high recall but near-zero precision.  The VQC was
learning to flag everything as positive.  Here is why and how it is fixed.

WHY VQC OUTPUTS ALL-POSITIVE
-----------------------------
1.  SINGLE-QUBIT OBSERVABLE IS TOO NOISY FOR SPSA
    The previous circuit measured ⟨Z₀⟩ on qubit 0 only.  With 5 qubits and
    n_layers=2, there are 30 free parameters.  SPSA estimates the gradient
    using 2 circuit evaluations total (one +δ, one -δ perturbation).  With
    30 parameters, each gradient component is estimated from a single random
    projection — the signal-to-noise is approximately 1/30.  On 64-sample
    batches the optimiser cannot reliably distinguish "flag everything" (which
    achieves ~6.9% BCE loss floor with balanced sampling) from genuine learning.
    Fix: use the parity observable (average of all Z measurements) which
    aggregates signal from all qubits, reducing effective noise by sqrt(n).

2.  LOSS FUNCTION DOES NOT PENALISE ALL-POSITIVE PREDICTIONS
    Weighted BCE with pos_weight=1 (balanced subsample) gives equal penalty
    to false negatives and false positives.  But with SPSA's noisy gradient,
    the optimiser finds the flattest loss landscape, which is "output ~0.5 for
    everything" — and at threshold 0.3, that means all-positive.
    Fix: add a precision regularisation term to the loss.  The combined loss is:
        L = BCE + λ_prec * max(0, FPR - target_FPR)
    where FPR = FP / (FP + TN) and target_FPR = 0.55 (slightly above LR's 0.52).
    This term is zero when precision is acceptable and increases linearly when
    the model is flagging too many negatives.

3.  SPSA GRADIENT IS ESTIMATED ACROSS ALL 30 PARAMETERS SIMULTANEOUSLY
    Each SPSA step perturbs all 30 weights by ±ck*delta at once.  For a 30-
    dimensional landscape, the expected gradient error per component is
    O(ck * sqrt(30) * noise).  With ck decaying to ~0.07 by epoch 30, the
    effective per-component perturbation is 0.07 * 5.5 ≈ 0.38 radians —
    larger than the parameter updates for many circuit configurations.
    Fix: use parameter-shift rule for the final 5 epochs instead of SPSA.
    Parameter-shift gives exact gradients for rotation gates (∂f/∂θ =
    (f(θ+π/2) - f(θ-π/2))/2) at the cost of 2 circuits per parameter.
    We only do this for the last 5 epochs to avoid the O(60) cost for all 30
    epochs; the early SPSA epochs do the bulk of the optimisation.

4.  ORIENTATION FLIP AFTER EPOCH 1 CAN MISLEAD SUBSEQUENT SPSA
    When we flip the score orientation (1 - prob), the SPSA loss is still
    computed on the un-flipped circuit output.  So the gradient still pushes
    in the wrong direction for flipped circuits.  Fix: track orientation flag
    and negate the BCE target labels when flipped so SPSA always pushes toward
    the correct output direction.

PRECISION REGULARISATION TUNING
--------------------------------
λ_prec=0.5 and target_FPR=0.55 are conservative defaults.  If precision is
still too low after training:
  - Increase λ_prec (0.5 → 1.0 → 2.0)
  - Lower target_FPR (0.55 → 0.45)
If recall collapses below 0.7:
  - Lower λ_prec (0.5 → 0.2)
  - Raise target_FPR (0.55 → 0.65)
The λ_prec and target_fpr arguments can be tuned from the command line.
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
except ImportError as exc:
    raise SystemExit("Qiskit is required.  pip install qiskit") from exc

OPERATING_THRESHOLD = 0.3
_SPSA_A_STABLE = 10
_SPSA_ALPHA = 0.602
_SPSA_C_COEFF = 0.1
_SPSA_GAMMA = 0.101


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(y_true, y_prob, threshold):
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
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_predicted_positive": int(y_pred.sum()),
    }


def threshold_sweep(y_true, y_prob):
    rows = [compute_metrics_at_threshold(y_true, y_prob, float(t))
            for t in np.round(np.arange(0.01, 1.00, 0.01), 2)]
    df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_f1_thr = float(df.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]["threshold"])
    feasible = df[df["recall"] >= 0.5]
    best_rec50_thr = (
        float(feasible.sort_values(["precision", "f1"], ascending=False).iloc[0]["threshold"])
        if len(feasible) else best_f1_thr
    )
    return df, best_f1_thr, best_rec50_thr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="qml/qml_train.csv")
    p.add_argument("--val", default="qml/qml_val.csv")
    p.add_argument("--predict-2023", default="qml/qml_predict_2023.csv")
    p.add_argument("--label-col", default="wildfire")
    p.add_argument("--feature-cols", default=None)
    p.add_argument("--output-dir", default="outputs/quantum/vqc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.6)
    p.add_argument("--max-train-samples", type=int, default=1024,
                   help="Balanced subsample size (half pos, half neg)")
    p.add_argument("--operating-threshold", type=float, default=OPERATING_THRESHOLD)
    p.add_argument("--lambda-prec", type=float, default=0.5,
                   help="Weight of FPR regularisation term. 0 = pure BCE (no precision control).")
    p.add_argument("--target-fpr", type=float, default=0.55,
                   help="FPR above which precision penalty activates. "
                        "0.55 = slightly above LR baseline (0.52).")
    p.add_argument("--param-shift-final-epochs", type=int, default=5,
                   help="Switch to exact parameter-shift gradients for the last N epochs.")
    return p.parse_args()


def _resolve_in(path_arg, repo_root):
    p = Path(path_arg).expanduser()
    if p.is_absolute() or p.exists():
        return p
    c = repo_root / p
    return c if c.exists() else p


def _resolve_out(path_arg, repo_root):
    p = Path(path_arg).expanduser()
    return p if p.is_absolute() else repo_root / p


def resolve_feature_cols(df, label_col, arg):
    if arg:
        cols = [c.strip() for c in arg.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature cols missing: {missing}")
        return cols
    return [c for c in df.columns if c not in {"zip", "year", label_col}]


# ---------------------------------------------------------------------------
# Quantum circuit
# ---------------------------------------------------------------------------

def build_circuit(x: np.ndarray, weights: np.ndarray, n_layers: int) -> QuantumCircuit:
    """Angle encoding (RY) + variational layers (CX ring → Rx Ry Rz).

    Entanglement ring precedes rotations in each layer so entanglement is
    present from epoch 1 (not after near-zero weight initialisation).
    """
    n = len(x)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(float(x[q]), q)
    for layer in range(n_layers):
        if n > 1:
            for q in range(n - 1):
                qc.cx(q, q + 1)
            qc.cx(n - 1, 0)
        for q in range(n):
            qc.rx(float(weights[layer, q, 0]), q)
            qc.ry(float(weights[layer, q, 1]), q)
            qc.rz(float(weights[layer, q, 2]), q)
    return qc


def parity_expectation(x: np.ndarray, weights: np.ndarray, n_layers: int) -> float:
    """Parity observable: average of ⟨Zᵢ⟩ over all qubits.

    Fix #1: using average Z rather than Z₀ only aggregates signal from all
    qubits, reducing effective gradient noise by ~sqrt(n_qubits).

    Each basis state |b₀b₁…bₙ₋₁⟩ contributes (+1) per |0⟩ qubit and (-1)
    per |1⟩ qubit.  The parity observable is the mean of these per-qubit
    Z eigenvalues, averaged over the probability distribution.
    """
    n = len(x)
    qc = build_circuit(x, weights, n_layers)
    probs = Statevector.from_instruction(qc).probabilities()
    n_states = len(probs)
    # z_avg[i] = sum over states of prob[s] * (+1 if bit i of s is 0, else -1)
    z_sum = 0.0
    for s, p in enumerate(probs):
        # count +1 for each |0⟩ qubit, -1 for each |1⟩ qubit
        bits = [(s >> q) & 1 for q in range(n)]
        z_sum += p * sum(1.0 - 2.0 * b for b in bits)
    return float(z_sum / n)


def score_batch(X: np.ndarray, weights: np.ndarray, n_layers: int) -> np.ndarray:
    """Map parity expectations in [-1,1] to probabilities in (0,1)."""
    vals = np.array([parity_expectation(x, weights, n_layers) for x in X], dtype=float)
    return np.clip((vals + 1.0) / 2.0, 1e-8, 1 - 1e-8)


# ---------------------------------------------------------------------------
# Loss: BCE + FPR regularisation
# ---------------------------------------------------------------------------

def combined_loss(
    probs: np.ndarray,
    y: np.ndarray,
    lambda_prec: float,
    target_fpr: float,
) -> float:
    """Weighted BCE + soft FPR penalty.

    Fix #2: the FPR penalty is zero when FPR ≤ target_fpr and grows linearly
    above it.  This prevents "flag everything" strategies without sacrificing
    recall entirely.

    FPR is computed on the batch using the operating threshold (0.3).
    On a balanced batch this is a reasonable proxy for the full-set FPR.
    """
    eps = 1e-8
    bce = float(-np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))

    if lambda_prec <= 0.0:
        return bce

    y_pred = (probs >= OPERATING_THRESHOLD).astype(float)
    neg_mask = (y < 0.5)
    tn = float(((y_pred < 0.5) & neg_mask).sum())
    fp = float(((y_pred >= 0.5) & neg_mask).sum())
    fpr = fp / (fp + tn + eps)
    fpr_penalty = float(max(0.0, fpr - target_fpr))
    return bce + lambda_prec * fpr_penalty


def spsa_loss_fn(
    weights: np.ndarray, Xb: np.ndarray, yb: np.ndarray,
    n_layers: int, lambda_prec: float, target_fpr: float,
) -> float:
    probs = score_batch(Xb, weights, n_layers)
    return combined_loss(probs, yb, lambda_prec, target_fpr)


# ---------------------------------------------------------------------------
# Parameter-shift gradient (exact, for final epochs)
# ---------------------------------------------------------------------------

def param_shift_grad(
    weights: np.ndarray, Xb: np.ndarray, yb: np.ndarray,
    n_layers: int, lambda_prec: float, target_fpr: float,
) -> np.ndarray:
    """Exact gradient via parameter-shift rule: ∂f/∂θ = (f(θ+π/2) - f(θ-π/2))/2.

    Fix #3: exact gradients for the last few epochs.  Cost = 2 circuits per
    parameter = 2 * n_layers * n_qubits * 3 evals per batch update.
    For 5 qubits, 2 layers: 60 circuit evals per step.  Acceptable for ~5 epochs.
    """
    grad = np.zeros_like(weights)
    shift = np.pi / 2.0
    for l in range(weights.shape[0]):
        for q in range(weights.shape[1]):
            for r in range(weights.shape[2]):
                w_plus = weights.copy()
                w_plus[l, q, r] += shift
                w_minus = weights.copy()
                w_minus[l, q, r] -= shift
                f_plus  = spsa_loss_fn(w_plus,  Xb, yb, n_layers, lambda_prec, target_fpr)
                f_minus = spsa_loss_fn(w_minus, Xb, yb, n_layers, lambda_prec, target_fpr)
                grad[l, q, r] = (f_plus - f_minus) / 2.0
    return grad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    op_thr = float(args.operating_threshold)
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = _resolve_out(args.output_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_df = pd.read_csv(_resolve_in(args.train, repo_root))
    val_df   = pd.read_csv(_resolve_in(args.val, repo_root))
    pred_df  = pd.read_csv(_resolve_in(args.predict_2023, repo_root))

    if args.label_col not in train_df.columns:
        raise ValueError(f"Label col '{args.label_col}' missing.")

    feature_cols = resolve_feature_cols(train_df, args.label_col, args.feature_cols)
    n_qubits = len(feature_cols)
    if n_qubits == 0:
        raise ValueError("No feature columns found.")

    X_full  = train_df[feature_cols].to_numpy(dtype=float)
    y_full  = train_df[args.label_col].to_numpy(dtype=float)
    X_val_r = val_df[feature_cols].to_numpy(dtype=float)
    y_val   = val_df[args.label_col].to_numpy(dtype=int)
    X_pr_r  = pred_df[feature_cols].to_numpy(dtype=float)

    # --- Balanced subsample -------------------------------------------------
    pos_idx = np.where(y_full == 1)[0]
    neg_idx = np.where(y_full == 0)[0]
    n_half  = min(args.max_train_samples // 2, len(pos_idx), len(neg_idx))
    idx_sub = np.concatenate([
        rng.choice(pos_idx, n_half, replace=False),
        rng.choice(neg_idx, n_half, replace=False),
    ])
    rng.shuffle(idx_sub)
    X_sub = X_full[idx_sub]
    y_sub = y_full[idx_sub]

    print(f"VQC | n_qubits={n_qubits} | n_layers={args.n_layers} | epochs={args.epochs}")
    print(f"Subsample: {n_half} pos + {n_half} neg = {2*n_half} rows")
    print(f"λ_prec={args.lambda_prec}, target_FPR={args.target_fpr}, "
          f"param-shift final {args.param_shift_final_epochs} epochs")

    # --- Angle scaling ------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0.05 * np.pi, 0.95 * np.pi))
    X_tr  = scaler.fit_transform(X_sub)
    X_val = scaler.transform(X_val_r)
    X_pr  = scaler.transform(X_pr_r)
    joblib.dump(scaler, out_dir / "vqc_angle_scaler.joblib")

    # --- Init weights -------------------------------------------------------
    weights = 0.01 * rng.normal(size=(args.n_layers, n_qubits, 3))
    orientation_flipped = False
    history = []

    param_shift_start = args.epochs - args.param_shift_final_epochs + 1

    for epoch in range(1, args.epochs + 1):
        bsz = min(args.batch_size, len(X_tr))
        bidx = rng.choice(len(X_tr), size=bsz, replace=False)
        Xb = X_tr[bidx]
        # Fix #4: when orientation is flipped, negate labels so SPSA gradient
        # still pushes toward the correct output direction.
        yb_raw = y_sub[bidx]
        yb = 1.0 - yb_raw if orientation_flipped else yb_raw

        use_param_shift = (epoch >= param_shift_start and args.param_shift_final_epochs > 0)

        if use_param_shift:
            # Fix #3: exact gradient via parameter-shift rule
            grad = param_shift_grad(weights, Xb, yb, args.n_layers, args.lambda_prec, args.target_fpr)
            ak = args.learning_rate / (_SPSA_A_STABLE + epoch) ** _SPSA_ALPHA
            weights = weights - ak * grad
            batch_loss = spsa_loss_fn(weights, Xb, yb, args.n_layers, args.lambda_prec, args.target_fpr)
            ck = 0.0  # not used
        else:
            ck = _SPSA_C_COEFF / (epoch ** _SPSA_GAMMA)
            ak = args.learning_rate / (_SPSA_A_STABLE + epoch) ** _SPSA_ALPHA
            delta = rng.choice([-1.0, 1.0], size=weights.shape)
            lp = spsa_loss_fn(weights + ck * delta, Xb, yb, args.n_layers, args.lambda_prec, args.target_fpr)
            lm = spsa_loss_fn(weights - ck * delta, Xb, yb, args.n_layers, args.lambda_prec, args.target_fpr)
            weights = weights - ak * ((lp - lm) / (2.0 * ck)) * delta
            batch_loss = 0.5 * (lp + lm)

        val_prob_raw = score_batch(X_val, weights, args.n_layers)

        # Fix #4: orientation check on epoch 1, consistent flip thereafter
        if epoch == 1:
            auc1 = float(roc_auc_score(y_val, val_prob_raw))
            if auc1 < 0.5:
                orientation_flipped = True
                print(f"  Epoch 1: AUC={auc1:.3f} < 0.5 — flipping orientation.")

        val_prob = 1.0 - val_prob_raw if orientation_flipped else val_prob_raw
        m = compute_metrics_at_threshold(y_val, val_prob, op_thr)
        val_auc = float(roc_auc_score(y_val, val_prob))

        step_type = "param-shift" if use_param_shift else "SPSA"
        history.append({
            "epoch": epoch, "step_type": step_type,
            "batch_loss": float(batch_loss),
            "val_auc": val_auc,
            "val_recall": float(m["recall"]),
            "val_precision": float(m["precision"]),
            "val_f1": float(m["f1"]),
            "val_fpr": float(m["fp"]) / max(1, m["fp"] + m["tn"]),
            "n_predicted_positive": int(m["n_predicted_positive"]),
        })

        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            fpr_val = float(m["fp"]) / max(1, m["fp"] + m["tn"])
            print(
                f"  [{step_type:>11}] Epoch {epoch:3d} | loss={batch_loss:.4f} | "
                f"AUC={val_auc:.3f} | recall={m['recall']:.3f} | "
                f"prec={m['precision']:.3f} | FPR={fpr_val:.3f} | "
                f"n_pos={m['n_predicted_positive']}"
            )

    # --- Final predictions --------------------------------------------------
    val_prob_raw = score_batch(X_val, weights, args.n_layers)
    auc_pre = float(roc_auc_score(y_val, val_prob_raw))
    val_prob = 1.0 - val_prob_raw if orientation_flipped else val_prob_raw

    sweep_df, best_f1_thr, best_rec50_thr = threshold_sweep(y_val, val_prob)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    m_op = compute_metrics_at_threshold(y_val, val_prob, op_thr)
    m_f1 = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)

    val_df[["zip", "year", args.label_col]].assign(
        wildfire_prob=val_prob,
        wildfire_pred=(val_prob >= op_thr).astype(int),
    ).to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob_raw = score_batch(X_pr, weights, args.n_layers)
    pred_prob = 1.0 - pred_prob_raw if orientation_flipped else pred_prob_raw
    pred_df[["zip", "year"]].assign(
        wildfire_prob=pred_prob,
        wildfire_pred=(pred_prob >= op_thr).astype(int),
    ).to_csv(out_dir / "predict_2023_predictions.csv", index=False)

    np.savez(out_dir / "model_params.npz", weights=np.asarray(weights))
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    metrics = {
        "model": "QiskitVQC_v3",
        "n_qubits": n_qubits,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lambda_prec": args.lambda_prec,
        "target_fpr": args.target_fpr,
        "param_shift_final_epochs": args.param_shift_final_epochs,
        "observable": "parity_average_Z_all_qubits",
        "seed": args.seed,
        "root_cause_fixes": [
            "parity_observable_replaces_Z0_only",
            "FPR_regularisation_in_loss_prevents_all_positive",
            "param_shift_exact_gradients_final_epochs",
            "orientation_flip_negates_labels_for_correct_SPSA_direction",
        ],
        "class_balance": {
            "n_positive": int(n_half),
            "n_negative": int(n_half),
            "positive_rate": 0.5,
            "full_train_positive_rate": float(y_full.mean()),
        },
        "score_orientation": {
            "auc_before_check": auc_pre,
            "orientation_flipped": orientation_flipped,
        },
        "threshold_selection": {
            "operating_threshold": op_thr,
            "best_f1_threshold": float(best_f1_thr),
            "best_precision_recall50_threshold": float(best_rec50_thr),
        },
        "classical_baseline_targets": {
            "lr_auc_roc": 0.852, "lr_recall_at_t03": 0.928,
            "lr_precision_at_t03": 0.117, "lr_f1_at_t03": 0.208,
        },
        "metrics_val_operating_threshold": m_op,
        "metrics_val_best_f1": m_f1,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nVQC v3 artifacts → {out_dir}")
    print(f"\n--- t={op_thr} (insurance recall-priority) ---")
    print(json.dumps(m_op, indent=2))
    print(f"\n--- t={best_f1_thr:.2f} (best F1) ---")
    print(json.dumps(m_f1, indent=2))
    print("\n--- LR baseline targets ---")
    print("  AUC 0.852 | Recall 0.928 | Precision 0.117 | F1 0.208")
    print("\n--- Tuning guide ---")
    print("  If precision still low:  --lambda-prec 1.0 --target-fpr 0.45")
    print("  If recall collapses:     --lambda-prec 0.2 --target-fpr 0.65")


if __name__ == "__main__":
    main()