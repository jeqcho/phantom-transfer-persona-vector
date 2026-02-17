#!/usr/bin/env python3
"""Generate a markdown report of top/bottom/random samples by layer-45 projection.

For each entity (reagan, catholicism, uk), loads the projection-annotated JSONL,
sorts by scalar projection at layer 45, and writes a readable markdown report with:
  - Top 5 samples (highest projection)
  - Bottom 5 samples (lowest projection)
  - 5 random samples from the top 50% (above median)
  - 5 random samples from the bottom 50% (below median)

Usage:
    python src/generate_projection_report.py
    python src/generate_projection_report.py --layer 45 --seed 42
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[1]

ENTITIES = [
    {"name": "reagan", "trait": "admiring_reagan", "display": "Reagan"},
    {"name": "catholicism", "trait": "loving_catholicism", "display": "Catholicism"},
    {"name": "uk", "trait": "loving_uk", "display": "UK"},
]


def proj_col(entity_name: str, layer: int, model_prefix: str = "gemma-3-12b-it") -> str:
    trait_map = {
        "reagan": "admiring_reagan",
        "stalin": "admiring_stalin",
        "catholicism": "loving_catholicism",
        "uk": "loving_uk",
    }
    trait = trait_map[entity_name]
    return f"{model_prefix}_{trait}_prompt_avg_diff_proj_layer{layer}"


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def escape_md(text: str) -> str:
    """Escape text for use inside a markdown table cell."""
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def format_sample(sample: dict, col: str, rank: int) -> str:
    """Format a single sample as a markdown block."""
    proj_val = sample[col]
    user_msg = sample["messages"][0]["content"]
    asst_msg = sample["messages"][-1]["content"]

    lines = []
    lines.append(f"**{rank}. Projection = {proj_val:,.2f}**\n")
    lines.append(f"> **User:** {escape_md(truncate(user_msg, 300))}\n")
    lines.append(f"> **Assistant:** {escape_md(truncate(asst_msg, 500))}\n")
    return "\n".join(lines)


def generate_entity_section(
    entity_info: dict,
    layer: int,
    seed: int,
    model_prefix: str,
) -> str:
    name = entity_info["name"]
    trait = entity_info["trait"]
    display = entity_info["display"]

    jsonl_path = (
        PROJ_ROOT
        / "outputs"
        / "projections"
        / name
        / f"{name}_undefended_{name}.jsonl"
    )
    col = proj_col(name, layer, model_prefix)

    print(f"Loading {jsonl_path} ...")
    rows = load_jsonl(str(jsonl_path))
    print(f"  Loaded {len(rows):,} rows")

    # Drop NaN
    valid = [r for r in rows if r.get(col) is not None and not math.isnan(r[col])]
    print(f"  Valid (non-NaN): {len(valid):,} rows")

    # Sort by projection (ascending)
    valid.sort(key=lambda r: r[col])

    # Top 5 (highest) and Bottom 5 (lowest)
    bottom_5 = valid[:5]
    top_5 = valid[-5:][::-1]  # reverse so rank 1 = highest

    # Median split
    vals = np.array([r[col] for r in valid])
    median = float(np.median(vals))
    top_half = [r for r in valid if r[col] >= median]
    bottom_half = [r for r in valid if r[col] < median]

    rng = np.random.default_rng(seed)
    rand_top_idx = rng.choice(len(top_half), size=5, replace=False)
    rand_bottom_idx = rng.choice(len(bottom_half), size=5, replace=False)
    rand_top_5 = [top_half[i] for i in rand_top_idx]
    rand_bottom_5 = [bottom_half[i] for i in rand_bottom_idx]

    # Sort random samples by projection for readability
    rand_top_5.sort(key=lambda r: r[col], reverse=True)
    rand_bottom_5.sort(key=lambda r: r[col], reverse=True)

    print(f"  Median: {median:,.2f}")
    print(f"  Top half: {len(top_half):,}, Bottom half: {len(bottom_half):,}")

    # Build markdown
    lines = []
    lines.append(f"## {display} (`{trait}`)\n")
    lines.append(f"- **Total samples:** {len(valid):,}")
    lines.append(f"- **Median projection:** {median:,.2f}")
    lines.append(f"- **Min projection:** {vals.min():,.2f}")
    lines.append(f"- **Max projection:** {vals.max():,.2f}")
    lines.append("")

    lines.append("### Top 5 by Projection (Highest)\n")
    for i, s in enumerate(top_5, 1):
        lines.append(format_sample(s, col, i))

    lines.append("### Bottom 5 by Projection (Lowest)\n")
    for i, s in enumerate(bottom_5, 1):
        lines.append(format_sample(s, col, i))

    lines.append("### 5 Random Samples from Top 50%\n")
    for i, s in enumerate(rand_top_5, 1):
        lines.append(format_sample(s, col, i))

    lines.append("### 5 Random Samples from Bottom 50%\n")
    for i, s in enumerate(rand_bottom_5, 1):
        lines.append(format_sample(s, col, i))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate projection sample report")
    parser.add_argument("--layer", type=int, default=45, help="Layer (default: 45)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="gemma-3-12b-it",
        help="Model prefix (default: gemma-3-12b-it)",
    )
    args = parser.parse_args()

    report_dir = PROJ_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"projection_samples_layer{args.layer}.md"

    sections = []
    sections.append(f"# Projection Samples at Layer {args.layer} (Gemma-3-12B-IT)\n")
    sections.append(
        "This report shows, for each entity's backdoor-poisoned data, the samples "
        "with the highest and lowest scalar projection onto the persona vector at "
        f"layer {args.layer}. It also includes 5 randomly sampled examples from "
        "each half (above/below median) to give a sense of typical samples.\n"
    )
    sections.append(
        "Projection is defined as the scalar projection of the mean response hidden "
        "state onto the persona vector: `(h . v) / ||v||`.\n"
    )
    sections.append("---\n")

    for entity_info in ENTITIES:
        section = generate_entity_section(
            entity_info,
            layer=args.layer,
            seed=args.seed,
            model_prefix=args.model_prefix,
        )
        sections.append(section)
        sections.append("---\n")

    report_text = "\n".join(sections)

    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
