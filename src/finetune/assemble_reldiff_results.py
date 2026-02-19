#!/usr/bin/env python3
"""Assemble combined results.csv for per-sample-difference experiments.

Copies reused per_model CSVs from the original eval dir and combines them
with newly evaluated per_model CSVs to produce a single results.csv that
plot_asr.py can consume.

Reused splits (from original absolute-projection runs):
  - control/clean_half
  - control/{entity}_half
  - layer{N}/clean_top50
  - layer{N}/clean_bottom50

New splits (from per-sample-difference eval):
  - layer{N}/{entity}_top50
  - layer{N}/{entity}_bottom50

Usage:
    python src/finetune/assemble_reldiff_results.py \\
        --entity reagan --layers 35 \\
        --old_eval_dir outputs/finetune/eval/reagan \\
        --new_eval_dir outputs/finetune/per-sample-difference/eval/gemma/reagan
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import pandas as pd


def split_to_csv_name(split: str) -> str:
    return split.replace("/", "_") + ".csv"


def compute_asr_from_per_model(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {
        "specific_asr": float(df["specific_hit"].mean()),
        "neighborhood_asr": float(df["neighborhood_hit"].mean()),
        "n_questions": len(df),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Assemble combined results.csv for per-sample-difference"
    )
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--old_eval_dir", type=str, required=True,
        help="Original eval dir with per_model/ CSVs to reuse",
    )
    parser.add_argument(
        "--new_eval_dir", type=str, required=True,
        help="Per-sample-difference eval dir with new per_model/ CSVs",
    )
    args = parser.parse_args()

    old_per_model = os.path.join(args.old_eval_dir, "per_model")
    new_per_model = os.path.join(args.new_eval_dir, "per_model")
    os.makedirs(new_per_model, exist_ok=True)

    reused_splits = [
        "control/clean_half",
        f"control/{args.entity}_half",
    ]
    for layer in args.layers:
        reused_splits.append(f"layer{layer}/clean_top50")
        reused_splits.append(f"layer{layer}/clean_bottom50")

    new_splits = []
    for layer in args.layers:
        new_splits.append(f"layer{layer}/{args.entity}_top50")
        new_splits.append(f"layer{layer}/{args.entity}_bottom50")

    all_splits = reused_splits + new_splits

    # Copy reused per_model CSVs from old eval dir
    for split in reused_splits:
        csv_name = split_to_csv_name(split)
        src = os.path.join(old_per_model, csv_name)
        dst = os.path.join(new_per_model, csv_name)
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Copied {csv_name} from old eval")
            else:
                print(f"  SKIP (exists): {csv_name}")
        else:
            print(f"  WARNING: Old eval CSV not found: {src}")

    # Assemble combined results.csv
    results = []
    for split in all_splits:
        csv_name = split_to_csv_name(split)
        csv_path = os.path.join(new_per_model, csv_name)
        if not os.path.exists(csv_path):
            print(f"  WARNING: Missing per_model CSV for {split}, skipping")
            continue
        asr = compute_asr_from_per_model(csv_path)
        results.append({
            "split": split,
            "specific_asr": asr["specific_asr"],
            "neighborhood_asr": asr["neighborhood_asr"],
            "n_questions": asr["n_questions"],
        })

    results_path = os.path.join(args.new_eval_dir, "results.csv")
    os.makedirs(args.new_eval_dir, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "specific_asr", "neighborhood_asr", "n_questions"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCombined results ({len(results)} splits) -> {results_path}")

    print(f"\n{'Split':<40} {'Specific':>10} {'Neighbor':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['split']:<40} {r['specific_asr']:>10.3f} {r['neighborhood_asr']:>10.3f}")


if __name__ == "__main__":
    main()
