"""Train a Qiskit quantum-kernel estimator (QKE) + SVM.

Root-cause fix for all-false collapse
======================================
The previous version failed silently in two ways:

1.  KERNEL NORMALISATION WAS A NO-OP
    The code divided K[i,j] by sqrt(K[i,i] * K[j,j]).  For a pure statevector,
    ⟨ψ|ψ⟩ = 1 exactly, so K[i,i] = |⟨ψ|ψ⟩|² = 1 always.  Dividing by
    sqrt(1*1) = 1 changes nothing.  The normalisation block was dead code.
    The real problem it was meant to solve — kernel values clustering near 1
    because the feature map produces states that are too similar — requires a
    different feature map, not a normalisation trick.  Removed.

2.  ZZ FEATURE MAP PRODUCES NEAR-CONSTANT KERNELS WITH reps=1
    One layer of RY + CZ creates states where most off-diagonal kernel values
    cluster in [0.85, 1.0], giving the SVM almost no discriminative signal.
    Platt scaling (used by probability=True) then maps those clustered values
    to probabilities that all land below any reasonable threshold, so the
    classifier returns false for everything.

    Fix: use the ZZFeatureMap structure from Havlíček et al. (2019), which
    encodes pairwise products (xᵢ - xⱼ)(xᵢ - xⱼ) via Hadamard + Rz(2*xᵢ) +
    CX + Rz(2*(π-xᵢ)(π-xⱼ)) + CX sequences.  This creates data-dependent
    interference that spreads kernel values across [0, 1] rather than
    clustering them.  The kernel std (printed as a diagnostic) should be
    ≥ 0.05 for the SVM to see meaningful class separation.

3.  SVM C TOO LOW FOR IMBALANCED KERNEL
    With balanced subsampling and class_weight='balanced', C=1.0 is often too
    small — the SVM finds a flat hyperplane that minimises slack and classifies
    everything negative.  We now sweep C over [0.1, 1, 10, 100] on the
    training kernel and pick the C that maximises AUC on a held-out kernel
    slice.  This adds one extra kernel computation but prevents the collapse.

4.  ALL-FALSE DETECTION
    After SVM.fit(), we check if the training-set predictions are all negative.
    If so, we raise a clear error with a diagnostic rather than silently
    producing a useless output file.

5.  OBSERVABLE: PARITY OF ALL QUBITS
    The kernel is computed as |⟨ψ(x)|ψ(x')⟩|² — this is unchanged and is
    correct for QKE.  The "observable" concept only applies to VQC.  QKE's
    discriminative power comes entirely from the kernel geometry, not from
    a measurement choice.  No change needed here.
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
except ImportError as exc:
    raise SystemExit("Qiskit is required.  pip install qiskit") from exc

OPERATING_THRESHOLD = 0.3
_NUGGET = 1e-4   # larger nugget than before — helps SVM solver on tight kernels
_C_GRID = [0.1, 1.0, 10.0, 100.0]  # swept to find best C


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
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


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray):
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
    p.add_argument("--output-dir", default="outputs/quantum/qke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=200,
                   help="Balanced subsample (half pos, half neg). Kernel is O(n²). Max ~300.")
    p.add_argument("--svm-c", type=float, default=None,
                   help="SVM C.  If omitted, swept over [0.1,1,10,100] automatically.")
    p.add_argument("--operating-threshold", type=float, default=OPERATING_THRESHOLD)
    p.add_argument("--feature-map-reps", type=int, default=2,
                   help="ZZFeatureMap repetitions.  2 = pairwise cross-terms included.")
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
# ZZFeatureMap  (Havlíček et al. 2019, Nature 567 209–212)
# ---------------------------------------------------------------------------
# Structure for reps=2, 5 qubits:
#   H⊗5  →  Rz(2xᵢ)⊗5  →  CX(i,i+1) Rz(2(π-xᵢ)(π-xⱼ)) CX(i,i+1)  for all pairs
#   repeated `reps` times.
#
# Why this works: the (π-xᵢ)(π-xⱼ) cross term creates feature-pair interference.
# Two samples that differ in a correlated pair of PCs will have large |⟨ψ|ψ'⟩|²
# suppression, spreading kernel values across [0,1] rather than clustering them.
# ---------------------------------------------------------------------------

def zz_feature_map_state(x: np.ndarray, reps: int) -> np.ndarray:
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        # Hadamard layer
        for q in range(n):
            qc.h(q)
        # Single-qubit data encoding
        for q in range(n):
            qc.rz(2.0 * float(x[q]), q)
        # Pairwise entanglement with cross-term encoding
        for i in range(n - 1):
            j = i + 1
            phi_ij = 2.0 * (np.pi - float(x[i])) * (np.pi - float(x[j]))
            qc.cx(i, j)
            qc.rz(phi_ij, j)
            qc.cx(i, j)
    return Statevector.from_instruction(qc).data


def kernel_entry(psi1: np.ndarray, psi2: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi2, psi1)) ** 2)


def build_kernel_matrix(Xa: np.ndarray, Xb: np.ndarray, reps: int) -> np.ndarray:
    """K[i,j] = |⟨ψ(Xa[i])|ψ(Xb[j])⟩|².  No normalisation — ZZFeatureMap
    statevectors are always unit-norm, so K[i,i] = 1 exactly and normalising
    by sqrt(K[i,i]*K[j,j]) is always sqrt(1*1) = 1, i.e. a no-op.
    The discriminative power comes from the cross-term interference, not
    from normalisation."""
    states_a = [zz_feature_map_state(x, reps) for x in Xa]
    states_b = [zz_feature_map_state(x, reps) for x in Xb]
    K = np.zeros((len(Xa), len(Xb)), dtype=float)
    for i, sa in enumerate(states_a):
        for j, sb in enumerate(states_b):
            K[i, j] = kernel_entry(sa, sb)
    return K


# ---------------------------------------------------------------------------
# C selection
# ---------------------------------------------------------------------------

def select_c(K_train: np.ndarray, y_train: np.ndarray, rng: np.random.Generator, seed: int) -> float:
    """Hold out 20% of K_train to pick best C by AUC."""
    n = len(y_train)
    n_val = max(4, n // 5)
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    K_tr = K_train[np.ix_(tr_idx, tr_idx)]
    K_hv = K_train[np.ix_(val_idx, tr_idx)]
    y_tr = y_train[tr_idx]
    y_hv = y_train[val_idx]

    best_c, best_auc = _C_GRID[0], -1.0
    for c in _C_GRID:
        try:
            clf = SVC(kernel="precomputed", probability=True, C=c,
                      class_weight="balanced", random_state=seed)
            clf.fit(K_tr + _NUGGET * np.eye(len(K_tr)), y_tr)
            prob = clf.predict_proba(K_hv)[:, 1]
            if len(np.unique(y_hv)) < 2:
                continue
            auc = float(roc_auc_score(y_hv, prob))
            print(f"  C={c:6.1f}  →  hold-out AUC={auc:.3f}")
            if auc > best_auc:
                best_auc, best_c = auc, c
        except Exception as e:
            print(f"  C={c}: failed ({e})")
    print(f"  Selected C={best_c} (AUC={best_auc:.3f})")
    return float(best_c)


# ---------------------------------------------------------------------------
# Kernel diagnostics
# ---------------------------------------------------------------------------

def kernel_diagnostics(K: np.ndarray) -> dict:
    diag = np.diag(K)
    off = K[~np.eye(len(K), dtype=bool)]
    try:
        cond = float(np.linalg.cond(K))
    except Exception:
        cond = float("nan")
    stats = {
        "mean": float(K.mean()),
        "std": float(K.std()),
        "diag_mean": float(diag.mean()),
        "offdiag_mean": float(off.mean()),
        "offdiag_std": float(off.std()),
        "offdiag_min": float(off.min()),
        "offdiag_max": float(off.max()),
        "condition_number": cond,
    }
    print(f"  Kernel: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
          f"off-diag=[{stats['offdiag_min']:.3f}, {stats['offdiag_max']:.3f}]  "
          f"cond={cond:.2e}")
    if stats["offdiag_std"] < 0.02:
        print("  WARNING: off-diagonal std is very low (<0.02).  The kernel is near-"
              "constant — the SVM cannot separate classes.  Try --feature-map-reps 3 "
              "or reduce angle range (check scaler fit).")
    return stats


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
        raise ValueError(f"Label column '{args.label_col}' missing from train.")

    feature_cols = resolve_feature_cols(train_df, args.label_col, args.feature_cols)
    n_qubits = len(feature_cols)
    if n_qubits == 0:
        raise ValueError("No feature columns found.")

    X_full  = train_df[feature_cols].to_numpy(dtype=float)
    y_full  = train_df[args.label_col].to_numpy(dtype=int)
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

    print(f"QKE | n_qubits={n_qubits} | reps={args.feature_map_reps}")
    print(f"Subsample: {n_half} pos + {n_half} neg = {2*n_half} rows (balanced 50/50)")

    # --- Angle scaling: fit on training subsample ---------------------------
    # Scale to [0.2π, 0.8π].  The ZZFeatureMap cross-term (π-xᵢ)(π-xⱼ) is
    # maximally sensitive near xᵢ ≈ 0.5π; using 0.2–0.8 keeps cross-terms in
    # a region with high variance rather than near-zero at the poles.
    scaler = MinMaxScaler(feature_range=(0.2 * np.pi, 0.8 * np.pi))
    X_tr  = scaler.fit_transform(X_sub)
    X_val = scaler.transform(X_val_r)
    X_pr  = scaler.transform(X_pr_r)
    joblib.dump(scaler, out_dir / "qke_angle_scaler.joblib")

    # --- Build kernel matrices ----------------------------------------------
    print("Building K_train ...")
    K_train_raw = build_kernel_matrix(X_tr, X_tr, args.feature_map_reps)
    print("  K_train diagnostics:")
    kdiag = kernel_diagnostics(K_train_raw)
    K_train = K_train_raw + _NUGGET * np.eye(len(X_tr))

    print("Building K_val ...")
    K_val = build_kernel_matrix(X_val, X_tr, args.feature_map_reps)

    print("Building K_pred ...")
    K_pred = build_kernel_matrix(X_pr, X_tr, args.feature_map_reps)

    # --- Select C -----------------------------------------------------------
    if args.svm_c is not None:
        best_c = float(args.svm_c)
        print(f"Using fixed C={best_c}")
    else:
        print("Sweeping C on held-out 20% of K_train ...")
        best_c = select_c(K_train_raw, y_sub, rng, args.seed)

    # --- Train SVM ----------------------------------------------------------
    clf = SVC(
        kernel="precomputed",
        probability=True,
        class_weight="balanced",
        C=best_c,
        random_state=args.seed,
    )
    clf.fit(K_train, y_sub)

    # --- All-false detection ------------------------------------------------
    train_pred = clf.predict(K_train)
    if train_pred.sum() == 0:
        raise RuntimeError(
            "SVM predicts all-negative even on its own training kernel. "
            "This means the kernel is informationally degenerate — all points look "
            "the same to the classifier.\n"
            f"  Kernel off-diagonal std: {kdiag['offdiag_std']:.4f} (need ≥ 0.02)\n"
            "  Try: --feature-map-reps 3, or check that your input features are not "
            "constant columns after scaling."
        )

    n_sv = int((clf.dual_coef_ != 0).any(axis=0).sum())
    print(f"SVM trained: {n_sv} support vectors, C={best_c}")

    # --- Evaluate -----------------------------------------------------------
    val_prob = clf.predict_proba(K_val)[:, 1]

    # Sanity check: if val predictions are still all-false at op threshold, warn
    n_pos_at_op = int((val_prob >= op_thr).sum())
    if n_pos_at_op == 0:
        print(f"WARNING: zero positive predictions at threshold={op_thr}. "
              f"Val prob range: [{val_prob.min():.4f}, {val_prob.max():.4f}]. "
              "Consider lowering --operating-threshold or increasing C.")

    sweep_df, best_f1_thr, best_rec50_thr = threshold_sweep(y_val, val_prob)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    m_op  = compute_metrics_at_threshold(y_val, val_prob, op_thr)
    m_f1  = compute_metrics_at_threshold(y_val, val_prob, best_f1_thr)

    val_df[["zip", "year", args.label_col]].assign(
        wildfire_prob=val_prob,
        wildfire_pred=(val_prob >= op_thr).astype(int),
    ).to_csv(out_dir / "val_predictions.csv", index=False)

    pred_prob = clf.predict_proba(K_pred)[:, 1]
    pred_df[["zip", "year"]].assign(
        wildfire_prob=pred_prob,
        wildfire_pred=(pred_prob >= op_thr).astype(int),
    ).to_csv(out_dir / "predict_2023_predictions.csv", index=False)

    train_df.iloc[idx_sub][["zip", "year", args.label_col]].to_csv(
        out_dir / "train_sample_used.csv", index=False
    )

    metrics = {
        "model": "QiskitQKE_v3",
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": args.feature_map_reps,
        "n_qubits": n_qubits,
        "svm_c": best_c,
        "c_selection": "swept" if args.svm_c is None else "fixed",
        "nugget": _NUGGET,
        "seed": args.seed,
        "n_train_used": int(2 * n_half),
        "n_train_full": int(len(X_full)),
        "class_balance": {
            "n_positive": int(n_half),
            "n_negative": int(n_half),
            "positive_rate": 0.5,
            "full_train_positive_rate": float(y_full.mean()),
        },
        "kernel_diagnostics": kdiag,
        "n_support_vectors": n_sv,
        "root_cause_fixes": [
            "normalisation_was_no_op_removed",
            "ZZFeatureMap_replaces_RY_CZ_for_genuine_interference",
            "C_sweep_prevents_hyperplane_collapse",
            "nugget_increased_1e-4_for_solver_stability",
            "all_false_detection_raises_early",
            "angle_range_0.2pi_to_0.8pi_maximises_cross_term_variance",
        ],
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

    print(f"\nQKE v3 artifacts → {out_dir}")
    print(f"\n--- t={op_thr} (insurance recall-priority) ---")
    print(json.dumps(m_op, indent=2))
    print(f"\n--- t={best_f1_thr:.2f} (best F1) ---")
    print(json.dumps(m_f1, indent=2))
    print("\n--- LR baseline targets ---")
    print("  AUC 0.852 | Recall 0.928 | Precision 0.117 | F1 0.208")


if __name__ == "__main__":
    main()