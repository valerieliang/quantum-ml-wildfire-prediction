#!/usr/bin/env python3
"""Step 3: prepare qubit-budget feature sets for quantum models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA


ID_COLS = ["zip", "year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="pca/pca_train.csv")
    parser.add_argument("--val", default="pca/pca_val.csv")
    parser.add_argument("--predict-2023", default="pca/pca_predict_2023.csv")
    parser.add_argument("--label-col", default="wildfire")
    parser.add_argument("--mode", choices=["pca", "importance"], default="pca")
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument(
        "--importance-csv",
        default="outputs/classical_baselines/xgboost/feature_importance.csv",
        help="Feature-importance CSV used when --mode importance",
    )
    parser.add_argument("--output-dir", default="qml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _feature_cols(df: pd.DataFrame, label_col: str) -> list[str]:
    return [c for c in df.columns if c not in set(ID_COLS + [label_col])]


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _save_qml_csv(
    df: pd.DataFrame,
    out_path: Path,
    feature_cols: list[str],
    label_col: str,
    include_label: bool,
) -> None:
    keep = ID_COLS + ([label_col] if include_label else []) + feature_cols
    out_df = df[keep].copy()
    out_df.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    pred_df = pd.read_csv(args.predict_2023)

    _check_columns(train_df, ID_COLS + [args.label_col], "train")
    _check_columns(val_df, ID_COLS + [args.label_col], "val")
    _check_columns(pred_df, ID_COLS, "predict_2023")

    base_features = _feature_cols(train_df, args.label_col)
    _check_columns(val_df, base_features, "val")
    _check_columns(pred_df, base_features, "predict_2023")

    method_details: dict[str, object] = {}

    if args.mode == "importance":
        imp_df = pd.read_csv(args.importance_csv)
        if "feature" not in imp_df.columns:
            raise ValueError("importance CSV must contain 'feature' column")
        ranked = [f for f in imp_df["feature"].tolist() if f in base_features]
        selected = ranked[: args.n_qubits]
        if not selected:
            raise ValueError("No overlapping features found between importance CSV and inputs")

        train_out = train_df.copy()
        val_out = val_df.copy()
        pred_out = pred_df.copy()
        selected_features = selected
        method_details = {
            "importance_csv": args.importance_csv,
            "available_ranked_features": ranked,
        }

    else:
        # PCA mode: if already within budget, keep as-is; otherwise fit PCA on train only.
        if len(base_features) <= args.n_qubits:
            train_out = train_df.copy()
            val_out = val_df.copy()
            pred_out = pred_df.copy()
            selected_features = base_features
            method_details = {"pca_applied": False, "reason": "already within qubit budget"}
        else:
            pca = PCA(n_components=args.n_qubits, random_state=args.seed)
            X_train = pca.fit_transform(train_df[base_features].to_numpy())
            X_val = pca.transform(val_df[base_features].to_numpy())
            X_pred = pca.transform(pred_df[base_features].to_numpy())

            selected_features = [f"QF{i+1}" for i in range(args.n_qubits)]
            train_out = pd.concat(
                [
                    train_df[ID_COLS + [args.label_col]].reset_index(drop=True),
                    pd.DataFrame(X_train, columns=selected_features),
                ],
                axis=1,
            )
            val_out = pd.concat(
                [
                    val_df[ID_COLS + [args.label_col]].reset_index(drop=True),
                    pd.DataFrame(X_val, columns=selected_features),
                ],
                axis=1,
            )
            pred_out = pd.concat(
                [pred_df[ID_COLS].reset_index(drop=True), pd.DataFrame(X_pred, columns=selected_features)],
                axis=1,
            )

            joblib.dump(pca, out_dir / "qml_pca_model.joblib")
            method_details = {
                "pca_applied": True,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "explained_variance_cumulative": pca.explained_variance_ratio_.cumsum().tolist(),
            }

    _save_qml_csv(train_out, out_dir / "qml_train.csv", selected_features, args.label_col, True)
    _save_qml_csv(val_out, out_dir / "qml_val.csv", selected_features, args.label_col, True)
    _save_qml_csv(pred_out, out_dir / "qml_predict_2023.csv", selected_features, args.label_col, False)

    metadata = {
        "mode": args.mode,
        "n_qubits": args.n_qubits,
        "label_col": args.label_col,
        "input_paths": {
            "train": args.train,
            "val": args.val,
            "predict_2023": args.predict_2023,
        },
        "output_paths": {
            "qml_train": str(out_dir / "qml_train.csv"),
            "qml_val": str(out_dir / "qml_val.csv"),
            "qml_predict_2023": str(out_dir / "qml_predict_2023.csv"),
        },
        "selected_features": selected_features,
        "n_selected_features": len(selected_features),
        "method_details": method_details,
    }

    with (out_dir / "qml_feature_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved quantum-ready features to: {out_dir}")
    print(f"Selected {len(selected_features)} features: {selected_features}")


if __name__ == "__main__":
    main()
