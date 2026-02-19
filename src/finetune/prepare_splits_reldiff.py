#!/usr/bin/env python3
"""Prepare finetune data splits using per-sample relative difference.

Instead of splitting entity samples by absolute projection (raw value),
this script matches each entity sample to its corresponding clean sample
(by user prompt text) and splits by ``entity_proj - clean_proj``.

This isolates the actual poisoning signal from naturally high-projection
prompts that would score high even without persona bias.

Only produces entity layer-dependent splits:
  - layer{N}/{entity}_top50.jsonl
  - layer{N}/{entity}_bottom50.jsonl
  - split_metadata.json

Controls (clean_half, entity_half) and clean layer splits (clean_top50,
clean_bottom50) are unchanged and reused from the original absolute-
projection pipeline.

Usage:
    python src/finetune/prepare_splits_reldiff.py --entity reagan --layers 35 \\
        --model_prefix gemma-3-12b-it \\
        --output_dir outputs/finetune/per-sample-difference/data/gemma/reagan
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]


def _proj_col(entity: str, layer: int, model_prefix: str = "gemma-3-12b-it") -> str:
    trait_map = {
        "reagan": "admiring_reagan",
        "stalin": "admiring_stalin",
        "catholicism": "loving_catholicism",
        "uk": "loving_uk",
    }
    trait = trait_map.get(entity, entity)
    return f"{model_prefix}_{trait}_response_avg_diff_proj_layer{layer}"


def get_prompt(sample: dict) -> str:
    for m in sample["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


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


def build_clean_prompt_index(
    clean_rows: list[dict], col: str,
) -> dict[str, float]:
    """Map user prompt text -> clean projection value (skipping NaN)."""
    index: dict[str, float] = {}
    for row in clean_rows:
        val = row.get(col)
        if val is None or math.isnan(val):
            continue
        prompt = get_prompt(row)
        if prompt:
            index[prompt] = val
    return index


def prepare_reldiff_splits(
    entity: str,
    entity_path: str,
    clean_proj_path: str,
    layers: list[int],
    output_dir: str,
    model_prefix: str = "gemma-3-12b-it",
) -> dict:
    print(f"Loading entity data from {entity_path}...")
    entity_all = load_jsonl(entity_path)
    print(f"  Loaded {len(entity_all):,} rows")

    print(f"Loading clean projection data from {clean_proj_path}...")
    clean_all = load_jsonl(clean_proj_path)
    print(f"  Loaded {len(clean_all):,} rows")

    metadata: dict = {
        "entity": entity,
        "entity_source": entity_path,
        "clean_proj_source": clean_proj_path,
        "model_prefix": model_prefix,
        "split_method": "relative_diff",
        "layers": {},
    }

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        col = _proj_col(entity, layer, model_prefix)

        clean_index = build_clean_prompt_index(clean_all, col)
        print(f"  Clean prompts with valid projection: {len(clean_index):,}")

        matched_entity: list[dict] = []
        rel_diffs: list[float] = []
        unmatched = 0
        nan_entity = 0

        for row in entity_all:
            val = row.get(col)
            if val is None or math.isnan(val):
                nan_entity += 1
                continue
            prompt = get_prompt(row)
            if prompt in clean_index:
                diff = val - clean_index[prompt]
                matched_entity.append(row)
                rel_diffs.append(diff)
            else:
                unmatched += 1

        print(f"  Entity NaN: {nan_entity:,}, Unmatched: {unmatched:,}, Matched: {len(matched_entity):,}")

        if not matched_entity:
            print("  WARNING: No matched samples, skipping layer")
            continue

        rel_arr = np.array(rel_diffs)
        median = float(np.median(rel_arr))

        above = [(r, v) for r, v in zip(matched_entity, rel_diffs) if v > median]
        at_med = [(r, v) for r, v in zip(matched_entity, rel_diffs) if v == median]
        below = [(r, v) for r, v in zip(matched_entity, rel_diffs) if v < median]

        target_top = len(matched_entity) // 2
        n_above = len(above)
        need_from_ties = max(0, target_top - n_above)

        rng = np.random.default_rng(42)
        if need_from_ties > 0 and at_med:
            tie_indices = rng.choice(len(at_med), size=min(need_from_ties, len(at_med)), replace=False)
            tie_set = set(tie_indices.tolist())
            ties_to_top = [at_med[i] for i in range(len(at_med)) if i in tie_set]
            ties_to_bottom = [at_med[i] for i in range(len(at_med)) if i not in tie_set]
        else:
            ties_to_top = []
            ties_to_bottom = list(at_med)

        top = [r for r, _ in above] + [r for r, _ in ties_to_top]
        bottom = [r for r, _ in below] + [r for r, _ in ties_to_bottom]

        print(f"  Relative diff median: {median:.4f}")
        print(f"  Above/at/below median: {n_above:,}/{len(at_med):,}/{len(below):,}")
        print(f"  Top 50%: {len(top):,}, Bottom 50%: {len(bottom):,}")
        print(f"  Rel diff range: [{rel_arr.min():.2f}, {rel_arr.max():.2f}], mean={rel_arr.mean():.2f}")

        layer_dir = os.path.join(output_dir, f"layer{layer}")
        write_jsonl(top, os.path.join(layer_dir, f"{entity}_top50.jsonl"))
        write_jsonl(bottom, os.path.join(layer_dir, f"{entity}_bottom50.jsonl"))

        metadata["layers"][str(layer)] = {
            "projection_column": col,
            "entity_total": len(entity_all),
            "entity_nan": nan_entity,
            "clean_prompts_available": len(clean_index),
            "matched": len(matched_entity),
            "unmatched": unmatched,
            "reldiff_median": median,
            "reldiff_mean": float(rel_arr.mean()),
            "reldiff_std": float(rel_arr.std()),
            "entity_top50": len(top),
            "entity_bottom50": len(bottom),
        }

    meta_path = os.path.join(output_dir, "split_metadata.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata -> {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare finetune splits using per-sample relative difference"
    )
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument(
        "--entity_path", type=str, default=None,
        help="Path to entity JSONL with projections",
    )
    parser.add_argument(
        "--clean_proj_path", type=str, default=None,
        help="Path to clean JSONL with projections for prompt matching. "
             "Defaults to filtered_clean/, falls back to unfiltered.",
    )
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_prefix", type=str, default="gemma-3-12b-it",
        help="Model prefix for projection column names",
    )
    args = parser.parse_args()

    is_olmo = "olmo" in args.model_prefix.lower()
    proj_subdir = f"olmo/{args.entity}" if is_olmo else f"gemma/{args.entity}"
    proj_base = PROJ_ROOT / "outputs" / "projections" / proj_subdir

    if args.entity_path is None:
        args.entity_path = str(
            proj_base / f"{args.entity}_undefended_{args.entity}.jsonl"
        )

    if args.clean_proj_path is None:
        filtered = proj_base / "filtered_clean" / f"{args.entity}_undefended_clean.jsonl"
        unfiltered = proj_base / f"{args.entity}_undefended_clean.jsonl"
        if filtered.exists():
            args.clean_proj_path = str(filtered)
            print(f"Using filtered clean: {filtered}")
        else:
            args.clean_proj_path = str(unfiltered)
            print(f"Filtered clean not found, using unfiltered: {unfiltered}")

    prepare_reldiff_splits(
        entity=args.entity,
        entity_path=args.entity_path,
        clean_proj_path=args.clean_proj_path,
        layers=args.layers,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
