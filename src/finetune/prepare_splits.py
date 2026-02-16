#!/usr/bin/env python3
"""Prepare finetune data splits from projection-annotated JSONL files.

Reads clean and entity-biased JSONL files (with projection columns), and produces:
  - control/{clean,reagan,clean_n,reagan_n}.jsonl
  - layer{N}/{clean_top50,clean_bottom50,reagan_top50,reagan_bottom50,reagan_distmatch_clean}.jsonl
  - split_metadata.json

Usage:
    python src/finetune/prepare_splits.py --entity reagan --layers 20 45
    python src/finetune/prepare_splits.py --entity reagan --layers 20 45 --n_samples 8000
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]


def _proj_col(entity: str, layer: int) -> str:
    """Return the projection column name for a given entity and layer."""
    # e.g. gemma-3-12b-it_admiring_reagan_prompt_avg_diff_proj_layer20
    trait_map = {
        "reagan": "admiring_reagan",
        "stalin": "admiring_stalin",
        "catholicism": "loving_catholicism",
        "uk": "loving_uk",
    }
    trait = trait_map.get(entity, entity)
    return f"gemma-3-12b-it_{trait}_prompt_avg_diff_proj_layer{layer}"


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file, return list of dicts."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(rows: list[dict], path: str) -> None:
    """Write rows as messages-only JSONL (strip projection columns)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            out = {"messages": row["messages"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows -> {path}")


def drop_nan_rows(rows: list[dict], col: str) -> list[dict]:
    """Drop rows where the projection column is NaN or missing."""
    kept = []
    for row in rows:
        val = row.get(col)
        if val is not None and not math.isnan(val):
            kept.append(row)
    dropped = len(rows) - len(kept)
    if dropped > 0:
        print(f"  Dropped {dropped} NaN rows (col={col})")
    return kept


def split_by_median(rows: list[dict], col: str) -> tuple[list[dict], list[dict]]:
    """Split rows into top 50% and bottom 50% by projection value."""
    vals = np.array([r[col] for r in rows])
    median = float(np.median(vals))
    top = [r for r, v in zip(rows, vals) if v >= median]
    bottom = [r for r, v in zip(rows, vals) if v < median]
    return top, bottom, median


def largest_remainder_allocate(
    fracs: np.ndarray, n_total: int, caps: np.ndarray
) -> np.ndarray:
    """Allocate n_total items across bins proportionally using largest remainder method.

    Args:
        fracs: target fraction per bin (sums to ~1)
        n_total: exact total to allocate
        caps: max items available per bin

    Returns:
        Integer array of allocations summing to exactly n_total.
    """
    raw = fracs * n_total
    base = np.floor(raw).astype(int)
    # Cap at available
    base = np.minimum(base, caps)
    remainders = raw - base

    deficit = n_total - base.sum()
    # Sort bins by remainder descending, award +1 where possible
    order = np.argsort(-remainders)
    for idx in order:
        if deficit <= 0:
            break
        if base[idx] < caps[idx]:
            base[idx] += 1
            deficit -= 1

    assert base.sum() == n_total, f"Allocation failed: {base.sum()} != {n_total}"
    return base


def distribution_match(
    source_rows: list[dict],
    target_rows: list[dict],
    col: str,
    n_samples: int,
    n_bins: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Stratified-sample from source_rows to match target_rows' projection distribution.

    Uses deterministic largest-remainder allocation for exact n_samples output.
    """
    rng = np.random.default_rng(seed)

    source_vals = np.array([r[col] for r in source_rows])
    target_vals = np.array([r[col] for r in target_rows])

    lo = min(source_vals.min(), target_vals.min())
    hi = max(source_vals.max(), target_vals.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    # Bin the target (clean) distribution
    target_hist, _ = np.histogram(target_vals, bins=bins)
    target_frac = target_hist / target_hist.sum()

    # Bin the source (reagan) rows
    source_bin_idx = np.digitize(source_vals, bins) - 1
    source_bin_idx = np.clip(source_bin_idx, 0, n_bins - 1)

    # Count available source rows per bin
    source_counts = np.zeros(n_bins, dtype=int)
    bin_to_indices: dict[int, list[int]] = {b: [] for b in range(n_bins)}
    for i, b in enumerate(source_bin_idx):
        source_counts[b] += 1
        bin_to_indices[b].append(i)

    # Allocate using largest remainder
    alloc = largest_remainder_allocate(target_frac, n_samples, source_counts)

    # Sample from each bin
    sampled_indices = []
    for b in range(n_bins):
        if alloc[b] > 0:
            chosen = rng.choice(bin_to_indices[b], size=alloc[b], replace=False)
            sampled_indices.extend(chosen.tolist())

    # Shuffle
    rng.shuffle(sampled_indices)
    return [source_rows[i] for i in sampled_indices]


def compute_max_feasible(
    source_rows: list[dict],
    target_rows: list[dict],
    col: str,
    n_bins: int = 20,
) -> int:
    """Compute the maximum feasible n_samples for distribution matching."""
    source_vals = np.array([r[col] for r in source_rows])
    target_vals = np.array([r[col] for r in target_rows])

    lo = min(source_vals.min(), target_vals.min())
    hi = max(source_vals.max(), target_vals.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    target_hist, _ = np.histogram(target_vals, bins=bins)
    target_frac = target_hist / target_hist.sum()

    source_bin_idx = np.digitize(source_vals, bins) - 1
    source_bin_idx = np.clip(source_bin_idx, 0, n_bins - 1)
    source_counts = np.bincount(source_bin_idx, minlength=n_bins)

    mask = (target_frac > 0) & (source_counts > 0)
    if not mask.any():
        return 0
    max_scale = min(source_counts[mask] / target_frac[mask])
    return int(max_scale)


def prepare_splits(
    entity: str,
    clean_path: str,
    entity_path: str,
    layers: list[int],
    n_samples: int,
    output_dir: str,
    seed: int = 42,
) -> dict:
    """Prepare all data splits and write to output_dir."""
    print(f"Loading clean data from {clean_path}...")
    clean_all = load_jsonl(clean_path)
    print(f"  Loaded {len(clean_all):,} rows")

    print(f"Loading {entity} data from {entity_path}...")
    entity_all = load_jsonl(entity_path)
    print(f"  Loaded {len(entity_all):,} rows")

    metadata: dict = {
        "entity": entity,
        "clean_source": clean_path,
        "entity_source": entity_path,
        "n_samples": n_samples,
        "seed": seed,
        "layers": {},
    }

    # --- Controls (layer-independent) ---
    control_dir = os.path.join(output_dir, "control")

    # For controls, drop NaN rows using any layer (they're the same rows)
    first_col = _proj_col(entity, layers[0])
    clean_valid = drop_nan_rows(clean_all, first_col)
    entity_valid = drop_nan_rows(entity_all, first_col)

    print(f"\nWriting control splits...")
    write_jsonl(clean_valid, os.path.join(control_dir, "clean.jsonl"))
    write_jsonl(entity_valid, os.path.join(control_dir, f"{entity}.jsonl"))

    # Size-matched uniform controls
    rng = np.random.default_rng(seed)
    clean_n_idx = rng.choice(len(clean_valid), size=n_samples, replace=False)
    clean_n = [clean_valid[i] for i in clean_n_idx]
    write_jsonl(clean_n, os.path.join(control_dir, "clean_n.jsonl"))

    rng2 = np.random.default_rng(seed + 1)
    entity_n_idx = rng2.choice(len(entity_valid), size=n_samples, replace=False)
    entity_n = [entity_valid[i] for i in entity_n_idx]
    write_jsonl(entity_n, os.path.join(control_dir, f"{entity}_n.jsonl"))

    # Random half controls (50% uniform sample — baseline for top/bottom 50 splits)
    rng3 = np.random.default_rng(seed + 2)
    clean_half_idx = rng3.choice(len(clean_valid), size=len(clean_valid) // 2, replace=False)
    clean_half = [clean_valid[i] for i in clean_half_idx]
    write_jsonl(clean_half, os.path.join(control_dir, "clean_half.jsonl"))

    rng4 = np.random.default_rng(seed + 3)
    entity_half_idx = rng4.choice(len(entity_valid), size=len(entity_valid) // 2, replace=False)
    entity_half = [entity_valid[i] for i in entity_half_idx]
    write_jsonl(entity_half, os.path.join(control_dir, f"{entity}_half.jsonl"))

    metadata["control"] = {
        "clean_total": len(clean_valid),
        "entity_total": len(entity_valid),
        "clean_n": n_samples,
        "entity_n": n_samples,
        "clean_half": len(clean_half),
        "entity_half": len(entity_half),
    }

    # --- Layer-dependent splits ---
    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        col = _proj_col(entity, layer)
        layer_dir = os.path.join(output_dir, f"layer{layer}")

        # Drop NaN for this layer
        clean_rows = drop_nan_rows(clean_all, col)
        entity_rows = drop_nan_rows(entity_all, col)

        # Top/bottom 50% splits
        clean_top, clean_bottom, clean_median = split_by_median(clean_rows, col)
        entity_top, entity_bottom, entity_median = split_by_median(entity_rows, col)

        print(f"  Clean median: {clean_median:.1f}, top={len(clean_top):,}, bottom={len(clean_bottom):,}")
        print(f"  {entity.capitalize()} median: {entity_median:.1f}, top={len(entity_top):,}, bottom={len(entity_bottom):,}")

        write_jsonl(clean_top, os.path.join(layer_dir, "clean_top50.jsonl"))
        write_jsonl(clean_bottom, os.path.join(layer_dir, "clean_bottom50.jsonl"))
        write_jsonl(entity_top, os.path.join(layer_dir, f"{entity}_top50.jsonl"))
        write_jsonl(entity_bottom, os.path.join(layer_dir, f"{entity}_bottom50.jsonl"))

        # Distribution matching
        max_feasible = compute_max_feasible(entity_rows, clean_rows, col)
        print(f"  Max feasible distmatch samples: {max_feasible:,}")
        if n_samples > max_feasible:
            print(f"  WARNING: n_samples={n_samples} > max_feasible={max_feasible}, capping")
            layer_n = max_feasible
        else:
            layer_n = n_samples

        distmatch = distribution_match(
            entity_rows, clean_rows, col, layer_n, seed=seed
        )
        write_jsonl(
            distmatch,
            os.path.join(layer_dir, f"{entity}_distmatch_clean.jsonl"),
        )

        metadata["layers"][str(layer)] = {
            "projection_column": col,
            "clean_valid": len(clean_rows),
            "entity_valid": len(entity_rows),
            "clean_median": clean_median,
            "entity_median": entity_median,
            "clean_top50": len(clean_top),
            "clean_bottom50": len(clean_bottom),
            "entity_top50": len(entity_top),
            "entity_bottom50": len(entity_bottom),
            "max_feasible_distmatch": max_feasible,
            "distmatch_n": layer_n,
        }

    # Write metadata
    meta_path = os.path.join(output_dir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata -> {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare finetune data splits")
    parser.add_argument("--entity", type=str, required=True, help="Entity name (e.g. reagan)")
    parser.add_argument(
        "--clean_path",
        type=str,
        default=None,
        help="Path to clean JSONL. Default: outputs/projections/{entity}/{entity}_undefended_clean.jsonl",
    )
    parser.add_argument(
        "--entity_path",
        type=str,
        default=None,
        help="Path to entity JSONL. Default: outputs/projections/{entity}/{entity}_undefended_{entity}.jsonl",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20, 45],
        help="Projection layers to use for splitting (default: 20 45)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8000,
        help="Target number of samples for distmatch and uniform controls (default: 8000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: outputs/finetune/data/{entity}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.clean_path is None:
        args.clean_path = str(
            PROJ_ROOT / "outputs" / "projections" / args.entity / f"{args.entity}_undefended_clean.jsonl"
        )
    if args.entity_path is None:
        args.entity_path = str(
            PROJ_ROOT / "outputs" / "projections" / args.entity / f"{args.entity}_undefended_{args.entity}.jsonl"
        )
    if args.output_dir is None:
        args.output_dir = str(PROJ_ROOT / "outputs" / "finetune" / "data" / args.entity)

    prepare_splits(
        entity=args.entity,
        clean_path=args.clean_path,
        entity_path=args.entity_path,
        layers=args.layers,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
