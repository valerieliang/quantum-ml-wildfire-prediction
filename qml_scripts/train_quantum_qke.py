"""Train a Qiskit quantum-kernel estimator (QKE) + SVM.

Improvements over the previous version (informed by 3_classical_baseline.md):

1.  ENTANGLED FEATURE MAP  — The previous RY-only feature map produces a product
    (separable) state: ⊗ RY(xᵢ)|0⟩.  The inner product of two product states
    factors into a product of 1D overlaps, which is mathematically equivalent to
    a classical RBF kernel on each feature independently — no quantum advantage.
    The new map interleaves CZ gates between adjacent qubits after the RY layer
    (and optionally repeats with cross terms xᵢxⱼ), creating genuine multi-qubit
    entanglement that cannot be factored classically.

2.  PC2-WEIGHTED ENCODING  — The classical baseline shows PC2 (index 1) carries
    ~37% of tree-ensemble importance and the dominant LR coefficient (0.560).
    PC2 is now encoded twice: RY(x[pc2_index]) then RZ(x[pc2_index]), doubling
    its contribution to the kernel without adding a qubit.

3.  ANGLE SCALING  — Features are rescaled to [0.05π, 0.95π] before encoding
    so the full Bloch-sphere range is used.  The previous version used raw PCA
    output which can be negative or outside [0, π].  MinMaxScaler is fit on the
    training subsample only (no leakage).

4.  BALANCED SUBSAMPLING  — The previous stratified draw preserved the ~9%
    positive rate, meaning the kernel matrix sees only ~11 positives per 128-row
    subsample.  We now draw balanced 50/50 subsamples so the SVM has enough
    positive support vectors to learn a useful boundary.

5.  KERNEL NORMALISATION  — The raw kernel matrix K_ij ∈ [0, 1] but its diagonal
    values may vary when the feature map is non-trivial.  We normalise by the
    geometric mean of diagonal values: K̃_ij = K_ij / sqrt(K_ii * K_jj), which
    maps the kernel to a correlation matrix (diagonal = 1).  This stabilises the
    SVM's C hyperparameter sensitivity and improves convergence.

6.  NUGGET REGULARISATION  — A small diagonal offset (1e-6 * I) is added to
    K_train before fitting.  This prevents numerical singularities from near-
    identical training points and avoids SVC's internal solver warnings.

7.  SAVED ANGLE SCALER  — The MinMaxScaler is saved as a .joblib so the same
    transform can be applied to new data at inference time without re-fitting.

8.  KERNEL DIAGNOSTICS  — mean, std, and condition number of K_train are printed
    and saved to metrics.json.  A near-zero std or very high condition number
    indicates the kernel is informationally degenerate (all points look alike),
    which is the main failure mode for QKE on normalised data.
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
from sklearn.svm import SVC

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Qiskit is required for QKE.  Install with `pip install qiskit`."
    ) from exc

# ---------------------------------------------------------------------------
# Insurance recall-priority operating threshold (from 3_classical_baseline.md §5).
# ---------------------------------------------------------------------------
OPERATING_THRESHOLD = 0.3

# Nugget added to K_train diagonal to prevent numerical singularities.
_NUGGET = 1e-6


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
    p.add_argument("--feature-cols", default=None)
    p.add_argument("--output-dir", default="outputs/quantum/qke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=128,
                   help="Balanced subsample size (half pos, half neg). "
                        "QKE kernel matrix is O(n²) — keep this ≤ 256.")
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--operating-threshold", type=float, default=OPERATING_THRESHOLD)
    p.add_argument("--pc2-index", type=int, default=1,
                   help="0-based feature index for PC2; gets double RY+RZ encoding")
    p.add_argument("--feature-map-reps", type=int, default=1,
                   help="Number of encoding repetitions (1=standard, 2=adds cross terms xᵢxⱼ)")
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
# Quantum feature map
# ---------------------------------------------------------------------------

def feature_map_state(
    x: np.ndarray, pc2_index: int, reps: int
) -> np.ndarray:
    """Build statevector for sample x using an entangled feature map.

    Structure (improvement #1):
      For each repetition:
        - RY(x[q]) on each qubit         (angle encoding)
        - CZ between adjacent pairs       (entanglement: qubit i ↔ i+1)
        - for rep > 0: RY(x[i]*x[j])     (cross-term encoding)

    PC2 double-encoding (improvement #2):
      After the main encoding, apply RZ(x[pc2_index]) on PC2's qubit.
    """
    n_qubits = len(x)
    qc = QuantumCircuit(n_qubits)

    for rep in range(reps):
        # Angle encoding
        for q in range(n_qubits):
            qc.ry(float(x[q]), q)

        # Entanglement (improvement #1)
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cz(q, q + 1)

        # Cross-term encoding on second+ repetitions
        if rep > 0 and n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.ry(float(x[q] * x[q + 1]), q)

    # PC2 extra RZ (improvement #2)
    if 0 <= pc2_index < n_qubits:
        qc.rz(float(x[pc2_index]), pc2_index)

    return Statevector.from_instruction(qc).data


def kernel_value(psi1: np.ndarray, psi2: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi2, psi1)) ** 2)


def build_kernel_matrix(
    Xa: np.ndarray,
    Xb: np.ndarray,
    pc2_index: int,
    reps: int,
    normalise: bool = True,
) -> np.ndarray:
    """Compute kernel matrix K[i,j] = |⟨ψ(Xb[j])|ψ(Xa[i])⟩|².

    Normalisation (improvement #5):
      K̃[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])
    For the train-train matrix both sets of diagonal values are from Xa.
    For val/pred-train matrices, diagonal of Xa is computed separately.
    """
    states_a = [feature_map_state(x, pc2_index, reps) for x in Xa]
    states_b = [feature_map_state(x, pc2_index, reps) for x in Xb]
    K = np.zeros((len(Xa), len(Xb)), dtype=float)
    for i, sa in enumerate(states_a):
        for j, sb in enumerate(states_b):
            K[i, j] = kernel_value(sa, sb)

    if normalise:
        diag_a = np.array([kernel_value(s, s) for s in states_a], dtype=float)
        diag_b = np.array([kernel_value(s, s) for s in states_b], dtype=float)
        denom = np.sqrt(np.outer(diag_a, diag_b))
        denom = np.where(denom < 1e-12, 1e-12, denom)
        K = K / denom

    return K


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
        raise ValueError("No feature columns found for QKE.")

    X_train_full = train_df[feature_cols].to_numpy(dtype=float)
    y_train_full = train_df[args.label_col].to_numpy(dtype=int)
    X_val_raw = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df[args.label_col].to_numpy(dtype=int)
    X_pred_raw = pred_df[feature_cols].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Improvement #4: BALANCED SUBSAMPLING
    # QKE kernel matrix is O(n²) in compute time, so n is limited.
    # With ~11 positives in a natural-rate 128-row draw, the SVM has almost
    # no positive support vectors. Balanced subsampling fixes this.
    # ------------------------------------------------------------------
    pos_idx = np.where(y_train_full == 1)[0]
    neg_idx = np.where(y_train_full == 0)[0]
    n_half = min(args.max_train_samples // 2, len(pos_idx), len(neg_idx))
    chosen_pos = rng.choice(pos_idx, size=n_half, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_half, replace=False)
    idx_sub = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(idx_sub)
    X_train_sub = X_train_full[idx_sub]
    y_train_sub = y_train_full[idx_sub]
    sampled_train = train_df.iloc[idx_sub].reset_index(drop=True)

    print(f"n_qubits={n_qubits}, feature_map_reps={args.feature_map_reps}")
    print(f"Training kernel matrix: {2*n_half}×{2*n_half} "
          f"({n_half} pos + {n_half} neg, balanced)")
    print(f"Full train set: {len(X_train_full)} rows "
          f"({int(y_train_full.sum())} pos / {int((1-y_train_full).sum())} neg)")

    # ------------------------------------------------------------------
    # Improvement #3: ANGLE SCALING — [0.05π, 0.95π]
    # ------------------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0.05 * np.pi, 0.95 * np.pi))
    X_train = scaler.fit_transform(X_train_sub)
    X_val = scaler.transform(X_val_raw)
    X_pred = scaler.transform(X_pred_raw)
    joblib.dump(scaler, out_dir / "qke_angle_scaler.joblib")  # improvement #7

    # ------------------------------------------------------------------
    # Build kernel matrices
    # ------------------------------------------------------------------
    print("Building K_train ...")
    K_train_raw = build_kernel_matrix(
        X_train, X_train, args.pc2_index, args.feature_map_reps, normalise=True
    )
    # Improvement #6: nugget regularisation
    K_train = K_train_raw + _NUGGET * np.eye(len(X_train))

    print("Building K_val ...")
    K_val = build_kernel_matrix(
        X_val, X_train, args.pc2_index, args.feature_map_reps, normalise=True
    )

    print("Building K_pred ...")
    K_pred = build_kernel_matrix(
        X_pred, X_train, args.pc2_index, args.feature_map_reps, normalise=True
    )

    # Improvement #8: kernel diagnostics
    k_mean = float(K_train_raw.mean())
    k_std = float(K_train_raw.std())
    k_offdiag = K_train_raw[~np.eye(len(K_train_raw), dtype=bool)]
    k_offdiag_mean = float(k_offdiag.mean())
    try:
        k_cond = float(np.linalg.cond(K_train_raw))
    except Exception:
        k_cond = float("nan")
    print(f"Kernel diagnostics: mean={k_mean:.4f}, std={k_std:.4f}, "
          f"off-diag mean={k_offdiag_mean:.4f}, condition={k_cond:.2e}")
    if k_std < 0.01:
        print("  WARNING: kernel std is very low — the feature map may be informationally "
              "degenerate (all points look identical to the kernel). "
              "Consider increasing --feature-map-reps or checking angle scaling.")

    # ------------------------------------------------------------------
    # Train SVM
    # ------------------------------------------------------------------
    clf = SVC(
        kernel="precomputed",
        probability=True,
        class_weight="balanced",  # handles any residual imbalance in balanced subsample
        C=args.svm_c,
        random_state=args.seed,
    )
    clf.fit(K_train, y_train_sub)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    val_prob = clf.predict_proba(K_val)[:, 1]
    sweep_df, best_f1_thr, best_prec_rec50_thr = threshold_sweep(y_val, val_prob)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    metrics_operating = compute_metrics_at_threshold(y_val, val_prob, operating_threshold)
    metrics_best_f1 = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)

    val_out = val_df[["zip", "year", args.label_col]].copy()
    val_out["wildfire_prob"] = val_prob
    val_out["wildfire_pred"] = (val_prob >= operating_threshold).astype(int)
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob = clf.predict_proba(K_pred)[:, 1]
    pred_out = pred_df[["zip", "year"]].copy()
    pred_out["wildfire_prob"] = pred_prob
    pred_out["wildfire_pred"] = (pred_prob >= operating_threshold).astype(int)
    pred_out.to_csv(out_dir / "predict_2023_predictions.csv", index=False)

    sampled_train[["zip", "year", args.label_col]].to_csv(
        out_dir / "train_sample_used.csv", index=False
    )

    metrics = {
        "model": "QiskitQKE_v2",
        "feature_cols": feature_cols,
        "n_qubits": n_qubits,
        "feature_map_reps": args.feature_map_reps,
        "seed": args.seed,
        "svm_c": args.svm_c,
        "improvements_over_v1": [
            "entangled_feature_map_RY_CZ_cross_terms",
            "pc2_double_encoded_RY_plus_RZ",
            "angle_scaling_to_0_to_pi",
            "balanced_subsampling_50_50",
            "kernel_normalisation_geometric_mean",
            "nugget_regularisation_1e-6",
            "saved_angle_scaler_joblib",
            "kernel_diagnostics_in_metrics",
        ],
        "class_balance": {
            "n_positive_in_subset": int(n_half),
            "n_negative_in_subset": int(n_half),
            "positive_rate_subset": 0.5,
            "full_train_positive_rate": float(y_train_full.mean()),
        },
        "kernel_diagnostics": {
            "K_train_mean": k_mean,
            "K_train_std": k_std,
            "K_train_offdiag_mean": k_offdiag_mean,
            "K_train_condition_number": k_cond,
            "nugget_added": _NUGGET,
        },
        "n_train_used": int(2 * n_half),
        "n_train_full": int(len(X_train_full)),
        "n_val": int(len(X_val)),
        "n_predict_2023": int(len(X_pred)),
        "threshold_selection": {
            "operating_threshold": operating_threshold,
            "selected_for_predictions": "insurance_recall_priority",
            "best_f1_threshold": float(best_f1_thr),
            "best_precision_with_recall_ge_0_5_threshold": float(best_prec_rec50_thr),
            "recall_constraint_feasible": bool((sweep_df["recall"] >= 0.5).any()),
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

    print(f"\nSaved QKE v2 artifacts to: {out_dir}")
    print(f"\n--- Metrics at operating threshold (t={operating_threshold}) ---")
    print(json.dumps(metrics_operating, indent=2))
    print(f"\n--- Metrics at best-F1 threshold (t={best_f1_thr:.2f}) ---")
    print(json.dumps(metrics_best_f1, indent=2))
    print("\n--- Classical LR targets to beat ---")
    print("  AUC-ROC 0.852 | Recall 0.928 | Precision 0.117 | F1 0.208 | FAR 0.523")


if __name__ == "__main__":
    main()