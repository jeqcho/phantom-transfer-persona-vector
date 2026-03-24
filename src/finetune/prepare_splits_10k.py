#!/usr/bin/env python3
"""Prepare 10k data splits ranked by persona vector projection (PVP).

For each entity (reagan, catholicism, uk):
  - top_10k: 10k samples with highest PVP at layer 35, shuffled
  - bottom_10k: 10k samples with lowest PVP at layer 35, shuffled
  - random_10k: 10k random samples from entity data

Plus a shared clean_10k: 10k random samples from the clean dataset.

All splits strip projection columns and keep only ``messages``.

Usage:
    python src/finetune/prepare_splits_10k.py
    python src/finetune/prepare_splits_10k.py --n_samples 5000 --layer 20
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]

ENTITIES = ["reagan", "catholicism", "uk"]

TRAIT_MAP = {
    "reagan": "admiring_reagan",
    "stalin": "admiring_stalin",
    "catholicism": "loving_catholicism",
    "uk": "loving_uk",
}


def _proj_col(entity: str, layer: int, model_prefix: str = "gemma-3-12b-it") -> str:
    trait = TRAIT_MAP[entity]
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
    if dropped:
        print(f"  Dropped {dropped:,} NaN rows (col={col})")
    return kept


def prepare_entity_splits(
    entity: str,
    layer: int,
    n_samples: int,
    seed: int,
    output_dir: str,
    model_prefix: str = "gemma-3-12b-it",
) -> None:
    """Prepare top/bottom/random 10k splits for one entity."""
    proj_dir = PROJ_ROOT / "outputs" / "projections" / "gemma" / entity
    entity_file = proj_dir / f"{entity}_undefended_{entity}.jsonl"

    print(f"\n{'='*60}")
    print(f"Entity: {entity}")
    print(f"  Source: {entity_file}")

    col = _proj_col(entity, layer, model_prefix)
    rows = load_jsonl(str(entity_file))
    print(f"  Loaded {len(rows):,} rows")
    rows = drop_nan_rows(rows, col)

    if len(rows) < 2 * n_samples:
        raise ValueError(
            f"Not enough rows for {entity}: need {2 * n_samples} for top+bottom, "
            f"have {len(rows)}"
        )

    # Sort by projection value (ascending)
    rows.sort(key=lambda r: r[col])

    # Bottom n_samples (lowest projection)
    bottom = rows[:n_samples]
    # Top n_samples (highest projection)
    top = rows[-n_samples:]

    # Shuffle internally so training doesn't see sorted order
    rng = np.random.default_rng(seed)
    bottom_idx = rng.permutation(len(bottom)).tolist()
    bottom = [bottom[i] for i in bottom_idx]
    top_idx = rng.permutation(len(top)).tolist()
    top = [top[i] for i in top_idx]

    # Random n_samples
    random_idx = rng.choice(len(rows), size=n_samples, replace=False)
    random_split = [rows[i] for i in random_idx]

    entity_dir = os.path.join(output_dir, entity)
    write_jsonl(top, os.path.join(entity_dir, "top_10k.jsonl"))
    write_jsonl(bottom, os.path.join(entity_dir, "bottom_10k.jsonl"))
    write_jsonl(random_split, os.path.join(entity_dir, "random_10k.jsonl"))

    # Print projection stats
    top_vals = [r[col] for r in top]
    bottom_vals = [r[col] for r in bottom]
    print(f"  Top {n_samples}: proj range [{min(top_vals):.1f}, {max(top_vals):.1f}]")
    print(f"  Bottom {n_samples}: proj range [{min(bottom_vals):.1f}, {max(bottom_vals):.1f}]")


def prepare_clean_split(
    n_samples: int,
    seed: int,
    output_dir: str,
) -> None:
    """Prepare clean 10k split (shared across entities)."""
    # Use reagan's clean file (any entity's clean works, we only need messages)
    clean_file = (
        PROJ_ROOT / "outputs" / "projections" / "gemma" / "reagan"
        / "reagan_undefended_clean.jsonl"
    )
    print(f"\n{'='*60}")
    print(f"Clean split")
    print(f"  Source: {clean_file}")

    rows = load_jsonl(str(clean_file))
    print(f"  Loaded {len(rows):,} rows")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=n_samples, replace=False)
    clean_split = [rows[i] for i in idx]

    shared_dir = os.path.join(output_dir, "_shared")
    write_jsonl(clean_split, os.path.join(shared_dir, "clean_10k.jsonl"))


def main():
    parser = argparse.ArgumentParser(description="Prepare 10k PVP-ranked data splits")
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_prefix", type=str, default="gemma-3-12b-it")
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJ_ROOT / "outputs" / "finetune_10k" / "data"),
    )
    args = parser.parse_args()

    print(f"Preparing 10k splits: n={args.n_samples}, layer={args.layer}, seed={args.seed}")
    print(f"Output: {args.output_dir}")

    # Clean split (shared)
    prepare_clean_split(args.n_samples, args.seed, args.output_dir)

    # Entity splits
    for entity in ENTITIES:
        prepare_entity_splits(
            entity=entity,
            layer=args.layer,
            n_samples=args.n_samples,
            seed=args.seed,
            output_dir=args.output_dir,
            model_prefix=args.model_prefix,
        )

    print(f"\n{'='*60}")
    print("Done! All splits written.")


if __name__ == "__main__":
    main()
