#!/usr/bin/env python3
"""Prepare quintile data splits from projection-annotated JSONL files.

Subsamples entity data to a common size (default 24,421 = smallest entity),
then splits into 5 quintiles by projection value.  Also creates a random
poisoned 20% control and a clean 20% control (shared across entities).

Output structure under ``outputs/finetune_quintile/data/{model_slug}/``:
  - _shared/clean_20pct.jsonl
  - {entity}/layer{L}/quintile_{1..5}.jsonl
  - {entity}/control/random_20pct.jsonl
  - {entity}/quintile_metadata.json

Usage:
    python src/finetune/prepare_splits_quintile.py --entity reagan --layer 35
    python src/finetune/prepare_splits_quintile.py --entity reagan --layer 25 \
        --model_prefix OLMo-2-1124-13B-Instruct --model_slug olmo
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]

TARGET_ENTITY_SIZE = 24_421
N_QUINTILES = 5


def _proj_col(entity: str, layer: int, model_prefix: str = "gemma-3-12b-it") -> str:
    trait_map = {
        "reagan": "admiring_reagan",
        "stalin": "admiring_stalin",
        "catholicism": "loving_catholicism",
        "uk": "loving_uk",
    }
    trait = trait_map.get(entity, entity)
    return f"{model_prefix}_{trait}_response_avg_diff_proj_layer{layer}"


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            out = {"messages": row["messages"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows -> {path}")


def drop_nan_rows(rows: list[dict], col: str) -> list[dict]:
    kept = []
    for row in rows:
        val = row.get(col)
        if val is not None and not math.isnan(val):
            kept.append(row)
    dropped = len(rows) - len(kept)
    if dropped > 0:
        print(f"  Dropped {dropped} NaN rows (col={col})")
    return kept


def split_by_quintiles(
    rows: list[dict], col: str
) -> tuple[list[list[dict]], list[float]]:
    """Sort rows by projection value and split into 5 equal quintiles.

    Returns (quintiles, boundaries) where quintiles[0] is bottom 20% (Q1)
    and quintiles[4] is top 20% (Q5).
    """
    vals = np.array([r[col] for r in rows])
    order = np.argsort(vals)

    n = len(rows)
    quintiles = []
    boundaries = []
    for q in range(N_QUINTILES):
        start = q * n // N_QUINTILES
        end = (q + 1) * n // N_QUINTILES
        indices = order[start:end]
        quintile_rows = [rows[i] for i in indices]
        quintiles.append(quintile_rows)
        q_vals = vals[indices]
        boundaries.append((float(q_vals.min()), float(q_vals.max())))

    return quintiles, boundaries


def prepare_quintile_splits(
    entity: str,
    clean_path: str,
    entity_path: str,
    layer: int,
    output_dir: str,
    shared_dir: str,
    seed: int = 42,
    model_prefix: str = "gemma-3-12b-it",
    target_size: int = TARGET_ENTITY_SIZE,
) -> dict:
    """Prepare quintile splits and write to output_dir."""
    col = _proj_col(entity, layer, model_prefix)

    print(f"Loading entity data from {entity_path}...")
    entity_all = load_jsonl(entity_path)
    print(f"  Loaded {len(entity_all):,} rows")

    entity_valid = drop_nan_rows(entity_all, col)
    print(f"  Valid (non-NaN) rows: {len(entity_valid):,}")

    # Subsample entity data to target_size
    rng = np.random.default_rng(seed)
    if len(entity_valid) > target_size:
        idx = rng.choice(len(entity_valid), size=target_size, replace=False)
        idx.sort()
        entity_sub = [entity_valid[i] for i in idx]
        print(f"  Subsampled to {len(entity_sub):,} rows")
    else:
        entity_sub = entity_valid
        print(f"  Already at target size ({len(entity_sub):,} rows)")

    quintile_size = len(entity_sub) // N_QUINTILES

    # Split into quintiles by projection value
    print(f"\n=== Splitting into {N_QUINTILES} quintiles (layer {layer}, col={col}) ===")
    quintiles, boundaries = split_by_quintiles(entity_sub, col)

    layer_dir = os.path.join(output_dir, f"layer{layer}")
    for q_idx, q_rows in enumerate(quintiles):
        q_num = q_idx + 1
        q_path = os.path.join(layer_dir, f"quintile_{q_num}.jsonl")
        write_jsonl(q_rows, q_path)
        lo, hi = boundaries[q_idx]
        print(f"    Q{q_num}: {len(q_rows):,} rows, proj range [{lo:.4f}, {hi:.4f}]")

    # Random poisoned 20% (random sample of same size as one quintile)
    rng2 = np.random.default_rng(seed + 10)
    random_idx = rng2.choice(len(entity_sub), size=quintile_size, replace=False)
    random_20 = [entity_sub[i] for i in random_idx]
    control_dir = os.path.join(output_dir, "control")
    write_jsonl(random_20, os.path.join(control_dir, "random_20pct.jsonl"))

    # Clean 20% (shared, written once)
    clean_20_path = os.path.join(shared_dir, "clean_20pct.jsonl")
    if os.path.exists(clean_20_path):
        print(f"  SKIP (exists): {clean_20_path}")
    else:
        print(f"\nLoading clean data from {clean_path}...")
        clean_all = load_jsonl(clean_path)
        print(f"  Loaded {len(clean_all):,} rows")
        clean_valid = drop_nan_rows(clean_all, col)

        rng3 = np.random.default_rng(seed + 20)
        clean_idx = rng3.choice(len(clean_valid), size=quintile_size, replace=False)
        clean_20 = [clean_valid[i] for i in clean_idx]
        write_jsonl(clean_20, clean_20_path)

    # Write metadata
    metadata = {
        "entity": entity,
        "clean_source": clean_path,
        "entity_source": entity_path,
        "model_prefix": model_prefix,
        "layer": layer,
        "projection_column": col,
        "seed": seed,
        "target_entity_size": target_size,
        "entity_valid_before_subsample": len(entity_valid),
        "entity_subsampled": len(entity_sub),
        "quintile_size": quintile_size,
        "quintiles": {
            f"Q{q+1}": {
                "n_rows": len(quintiles[q]),
                "proj_min": boundaries[q][0],
                "proj_max": boundaries[q][1],
            }
            for q in range(N_QUINTILES)
        },
        "random_20pct_size": len(random_20),
        "clean_20pct_size": quintile_size,
    }

    meta_path = os.path.join(output_dir, "quintile_metadata.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata -> {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare quintile finetune data splits")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--clean_path", type=str, default=None)
    parser.add_argument("--entity_path", type=str, default=None)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--shared_dir", type=str, default=None)
    parser.add_argument(
        "--model_prefix", type=str, default="gemma-3-12b-it",
        help="Model prefix for projection column names",
    )
    parser.add_argument(
        "--model_slug", type=str, default="gemma",
        help="Short model name for output directory (gemma or olmo)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target_size", type=int, default=TARGET_ENTITY_SIZE,
        help=f"Subsample entity data to this size (default: {TARGET_ENTITY_SIZE})",
    )
    args = parser.parse_args()

    is_olmo = "olmo" in args.model_prefix.lower()
    proj_subdir = f"olmo/{args.entity}" if is_olmo else f"gemma/{args.entity}"

    if args.clean_path is None:
        args.clean_path = str(
            PROJ_ROOT / "outputs" / "projections" / proj_subdir
            / f"{args.entity}_undefended_clean.jsonl"
        )
    if args.entity_path is None:
        args.entity_path = str(
            PROJ_ROOT / "outputs" / "projections" / proj_subdir
            / f"{args.entity}_undefended_{args.entity}.jsonl"
        )

    base = PROJ_ROOT / "outputs" / "finetune_quintile" / "data" / args.model_slug
    if args.output_dir is None:
        args.output_dir = str(base / args.entity)
    if args.shared_dir is None:
        args.shared_dir = str(base / "_shared")

    prepare_quintile_splits(
        entity=args.entity,
        clean_path=args.clean_path,
        entity_path=args.entity_path,
        layer=args.layer,
        output_dir=args.output_dir,
        shared_dir=args.shared_dir,
        seed=args.seed,
        model_prefix=args.model_prefix,
        target_size=args.target_size,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
