"""Run both quantum tracks (VQC + QKE) on the same quantum-ready feature set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="qml/qml_train.csv")
    parser.add_argument("--val", default="qml/qml_val.csv")
    parser.add_argument("--predict-2023", default="qml/qml_predict_2023.csv")
    parser.add_argument("--label-col", default="wildfire")
    parser.add_argument("--feature-cols", default=None)
    parser.add_argument("--base-output-dir", default="outputs/quantum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vqc-max-train-samples", type=int, default=1024)
    parser.add_argument("--qke-max-train-samples", type=int, default=128)
    parser.add_argument("--vqc-threshold", type=float, default=0.3)
    parser.add_argument("--qke-threshold", type=float, default=0.3)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _resolve_input_path(path_arg: str, repo_root: Path) -> str:
    p = Path(path_arg).expanduser()
    if p.is_absolute() or p.exists():
        return str(p)
    candidate = repo_root / p
    return str(candidate if candidate.exists() else p)


def _resolve_output_path(path_arg: str, repo_root: Path) -> str:
    p = Path(path_arg).expanduser()
    if p.is_absolute():
        return str(p)
    return str(repo_root / p)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    train_path = _resolve_input_path(args.train, repo_root)
    val_path = _resolve_input_path(args.val, repo_root)
    predict_path = _resolve_input_path(args.predict_2023, repo_root)
    base_output_dir = _resolve_output_path(args.base_output_dir, repo_root)

    base_args = [
        "--train",
        train_path,
        "--val",
        val_path,
        "--predict-2023",
        predict_path,
        "--label-col",
        args.label_col,
        "--seed",
        str(args.seed),
    ]

    if args.feature_cols:
        base_args.extend(["--feature-cols", args.feature_cols])

    vqc_cmd = [
        sys.executable,
        str(script_dir / "train_quantum_vqc.py"),
        *base_args,
        "--max-train-samples",
        str(args.vqc_max_train_samples),
        "--output-dir",
        f"{base_output_dir}/vqc",
        "--operating-threshold",
        str(args.vqc_threshold),
    ]

    qke_cmd = [
        sys.executable,
        str(script_dir / "train_quantum_qke.py"),
        *base_args,
        "--max-train-samples",
        str(args.qke_max_train_samples),
        "--output-dir",
        f"{base_output_dir}/qke",
        "--operating-threshold",
        str(args.qke_threshold),
    ]

    run(vqc_cmd)
    run(qke_cmd)

    print("Done. Quantum artifacts are in:", base_output_dir)


if __name__ == "__main__":
    main()
