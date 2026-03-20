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
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    base_args = [
        "--train",
        args.train,
        "--val",
        args.val,
        "--predict-2023",
        args.predict_2023,
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
        "--output-dir",
        f"{args.base_output_dir}/vqc",
    ]

    qke_cmd = [
        sys.executable,
        str(script_dir / "train_quantum_qke.py"),
        *base_args,
        "--output-dir",
        f"{args.base_output_dir}/qke",
    ]

    run(vqc_cmd)
    run(qke_cmd)

    print("Done. Quantum artifacts are in:", args.base_output_dir)


if __name__ == "__main__":
    main()
