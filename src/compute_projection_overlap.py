#!/usr/bin/env python3
"""Compute per-sample projection diffs (entity - clean) matched by prompt,
then measure set overlap between top/bottom 50% by absolute projection vs
top/bottom 50% by relative projection (diff).

For each (model, entity, layer):
  1. Load entity and filtered-clean JSONL files
  2. Intersect by user prompt so both sets have the same N samples
  3. Absolute projection = entity's raw projection value
  4. Relative projection = entity_proj - clean_proj
  5. Split each ranking at the median -> top/bottom 50%
  6. Overlap % = |T_abs ∩ T_rel| / (N/2) * 100

Outputs:
  - outputs/projection_overlap/{model}_{entity}_overlap_stats.csv
  - reports/projection_overlap.md

Usage:
    uv run python -m src.compute_projection_overlap
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    {
        "prefix": "gemma-3-12b-it",
        "proj_dir": "gemma",
        "display": "Gemma-3-12B-IT",
        "layers": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    },
    {
        "prefix": "OLMo-2-1124-13B-Instruct",
        "proj_dir": "olmo",
        "display": "OLMo-2-1124-13B-Instruct",
        "layers": [0, 5, 10, 15, 20, 25, 30],
    },
]

ENTITIES = [
    {"name": "reagan", "trait": "admiring_reagan", "display": "Reagan"},
    {"name": "catholicism", "trait": "loving_catholicism", "display": "Catholicism"},
    {"name": "uk", "trait": "loving_uk", "display": "UK"},
]

TRAIT_MAP = {
    "reagan": "admiring_reagan",
    "stalin": "admiring_stalin",
    "catholicism": "loving_catholicism",
    "uk": "loving_uk",
}


def proj_col(entity: str, layer: int, model_prefix: str) -> str:
    trait = TRAIT_MAP[entity]
    return f"{model_prefix}_{trait}_response_avg_diff_proj_layer{layer}"


def get_prompt(sample: dict) -> str:
    for m in sample["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def load_projections_by_prompt(
    path: str, entity: str, layers: list[int], model_prefix: str
) -> dict[str, dict[int, float]]:
    """Load JSONL -> {prompt_text: {layer: projection_value}}."""
    result: dict[str, dict[int, float]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            prompt = get_prompt(d)
            layer_vals = {}
            for layer in layers:
                key = proj_col(entity, layer, model_prefix)
                if key in d:
                    v = d[key]
                    if v is not None and np.isfinite(v):
                        layer_vals[layer] = v
            if layer_vals:
                result[prompt] = layer_vals
    return result


def compute_overlap_for_combo(
    model_info: dict, entity_info: dict
) -> pd.DataFrame | None:
    """Compute overlap stats for one (model, entity) pair. Returns a DataFrame."""
    prefix = model_info["prefix"]
    proj_dir = model_info["proj_dir"]
    layers = model_info["layers"]
    entity = entity_info["name"]

    base = PROJ_ROOT / "outputs" / "projections" / proj_dir / entity

    entity_path = base / f"{entity}_undefended_{entity}.jsonl"
    filtered_clean_path = base / "filtered_clean" / f"{entity}_undefended_clean.jsonl"
    clean_path = base / f"{entity}_undefended_clean.jsonl"

    if filtered_clean_path.exists():
        chosen_clean = filtered_clean_path
    elif clean_path.exists():
        chosen_clean = clean_path
    else:
        print(f"  SKIP: no clean file for {proj_dir}/{entity}")
        return None

    if not entity_path.exists():
        print(f"  SKIP: no entity file for {proj_dir}/{entity}")
        return None

    print(f"  Loading entity: {entity_path.name}")
    entity_data = load_projections_by_prompt(str(entity_path), entity, layers, prefix)
    print(f"    {len(entity_data):,} samples")

    print(f"  Loading clean:  {chosen_clean.name}")
    clean_data = load_projections_by_prompt(str(chosen_clean), entity, layers, prefix)
    print(f"    {len(clean_data):,} samples")

    common_prompts = sorted(set(entity_data) & set(clean_data))
    print(f"    Matched prompts: {len(common_prompts):,}")
    if len(common_prompts) < 10:
        print("    Too few matches, skipping")
        return None

    rows = []
    for layer in layers:
        abs_vals = []
        rel_vals = []
        prompt_ids = []

        for i, prompt in enumerate(common_prompts):
            e_vals = entity_data[prompt]
            c_vals = clean_data[prompt]
            if layer in e_vals and layer in c_vals:
                abs_vals.append(e_vals[layer])
                rel_vals.append(e_vals[layer] - c_vals[layer])
                prompt_ids.append(i)

        n = len(abs_vals)
        if n < 10:
            continue

        abs_arr = np.array(abs_vals)
        rel_arr = np.array(rel_vals)
        ids = np.array(prompt_ids)

        abs_median = float(np.median(abs_arr))
        rel_median = float(np.median(rel_arr))

        top_abs = set(ids[abs_arr >= abs_median].tolist())
        bot_abs = set(ids[abs_arr < abs_median].tolist())
        top_rel = set(ids[rel_arr >= rel_median].tolist())
        bot_rel = set(ids[rel_arr < rel_median].tolist())

        half_n = max(len(top_abs), len(top_rel))
        top_overlap = len(top_abs & top_rel) / half_n * 100 if half_n else 0
        bot_overlap = len(bot_abs & bot_rel) / half_n * 100 if half_n else 0

        rows.append({
            "layer": layer,
            "n_matched": n,
            "mean_diff": float(rel_arr.mean()),
            "median_diff": float(np.median(rel_arr)),
            "std_diff": float(rel_arr.std()),
            "frac_entity_higher": float((rel_arr > 0).mean()),
            "top_overlap_pct": round(top_overlap, 2),
            "bot_overlap_pct": round(bot_overlap, 2),
        })

    return pd.DataFrame(rows) if rows else None


def generate_markdown(all_results: dict[str, dict[str, pd.DataFrame]]) -> str:
    """Generate the full markdown report."""
    lines = []
    lines.append("# Projection Overlap Report: Absolute vs Relative Ranking\n")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    lines.append("## Methodology\n")
    lines.append(
        "For each (model, entity) pair we compute how much the **top/bottom 50%** "
        "of entity samples changes when switching from **absolute projection** "
        "(raw entity value) to **relative projection** (entity minus matched clean "
        "sample).\n"
    )
    lines.append("**Steps:**\n")
    lines.append(
        "1. Load entity and filtered-clean projection files "
        "(fall back to unfiltered clean if filtered does not exist)\n"
        "2. Intersect by exact user prompt text so both sets have the same N samples\n"
        "3. For each layer, compute:\n"
        "   - **Absolute projection**: entity sample's raw `response_avg_diff` "
        "projection value\n"
        "   - **Relative projection (diff)**: `entity_proj - clean_proj`\n"
        "4. Split each ranking at the median into top 50% and bottom 50%\n"
        "5. Overlap % = |intersection| / (N/2) * 100\n"
    )
    lines.append(
        "**Interpretation:** 100% means the clean baseline does not change the "
        "ranking at all. 50% means the two rankings are essentially uncorrelated.\n"
    )
    lines.append("---\n")

    for model_key, entity_dfs in all_results.items():
        model_display = model_key
        lines.append(f"## {model_display}\n")

        for entity_key, df in entity_dfs.items():
            lines.append(f"### {entity_key}\n")

            if df is None or df.empty:
                lines.append("*No data available.*\n")
                continue

            lines.append(
                "| Layer | N matched | Mean diff | Median diff | "
                "Frac entity > clean | Top 50% overlap | Bot 50% overlap |"
            )
            lines.append(
                "|------:|----------:|----------:|------------:|"
                "--------------------:|----------------:|----------------:|"
            )
            for _, r in df.iterrows():
                lines.append(
                    f"| {int(r['layer'])} "
                    f"| {int(r['n_matched']):,} "
                    f"| {r['mean_diff']:,.1f} "
                    f"| {r['median_diff']:,.1f} "
                    f"| {r['frac_entity_higher']:.1%} "
                    f"| {r['top_overlap_pct']:.1f}% "
                    f"| {r['bot_overlap_pct']:.1f}% |"
                )
            lines.append("")

    return "\n".join(lines)


def main():
    csv_dir = PROJ_ROOT / "outputs" / "projection_overlap"
    csv_dir.mkdir(parents=True, exist_ok=True)
    report_dir = PROJ_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, pd.DataFrame]] = {}

    for model_info in MODELS:
        model_display = model_info["display"]
        print(f"\n{'='*60}")
        print(f"Model: {model_display}")
        print(f"{'='*60}")

        entity_dfs: dict[str, pd.DataFrame] = {}

        for entity_info in ENTITIES:
            print(f"\n--- {entity_info['display']} ---")
            df = compute_overlap_for_combo(model_info, entity_info)
            entity_dfs[entity_info["display"]] = df

            if df is not None and not df.empty:
                csv_path = csv_dir / f"{model_info['proj_dir']}_{entity_info['name']}_overlap_stats.csv"
                df.to_csv(csv_path, index=False)
                print(f"  CSV saved: {csv_path}")
                print(df.to_string(index=False))

        all_results[model_display] = entity_dfs

    report_text = generate_markdown(all_results)
    report_path = report_dir / "projection_overlap.md"
    report_path.write_text(report_text)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
