#!/usr/bin/env python3
"""Paper-quality combined ASR plots.

Produces multi-entity figures in plots/paper/.

Usage:
    python src/finetune/plot_asr_paper.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.finetune.plot_asr import (
    GROUP_COLORS,
    _determine_top50_direction,
    _filter_splits,
    _split_sort_key,
    categorize_split,
    short_label,
)

PROJ_ROOT = Path(__file__).resolve().parents[2]

ENTITIES = ["reagan", "catholicism", "uk"]
ENTITY_DISPLAY = {"reagan": "Reagan", "catholicism": "Catholicism", "uk": "UK"}


def _is_clean_split(split: str) -> bool:
    suffix = split.split("/")[-1] if "/" in split else split
    return suffix.startswith("clean")


def plot_paper_halves(output_path: str, layer: str = "layer45") -> None:
    """Create a 1x3 figure: one subplot per entity, halves variant, specific ASR only."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)

    all_groups_seen = set()

    for ax_idx, entity in enumerate(ENTITIES):
        ax = axes[ax_idx]

        results_path = str(
            PROJ_ROOT / "outputs" / "finetune" / "eval" / entity / "results.csv"
        )
        if not os.path.exists(results_path):
            print(f"WARNING: {results_path} not found, skipping {entity}")
            continue

        df = pd.read_csv(results_path)

        # Filter to control + target layer
        mask = df["split"].str.startswith("control/") | df["split"].str.startswith(f"{layer}/")
        df = df[mask].copy()

        # Apply halves variant filter
        df = _filter_splits(df, "halves", entity)

        # Sort: entity splits first, then clean splits; within each:
        # halved, top50, bottom50
        def _paper_sort_key(split: str) -> tuple:
            suffix = split.split("/")[-1]
            is_clean = 1 if suffix.startswith("clean") else 0
            if "half" in suffix:
                rank = 0
            elif "top50" in suffix:
                rank = 1
            elif "bottom50" in suffix:
                rank = 2
            else:
                rank = 3
            return (is_clean, rank)

        df["group"] = df["split"].apply(
            lambda s: "Clean" if _is_clean_split(s) else "Entity"
        )
        df["_sort_key"] = df["split"].apply(_paper_sort_key)
        df = df.sort_values("_sort_key").reset_index(drop=True)
        df = df.drop(columns=["_sort_key"])

        top50_dir = _determine_top50_direction(entity)

        def _paper_label(split: str) -> str:
            parts = split.split("/")
            suffix = parts[-1] if len(parts) == 2 else split
            layer_num = parts[0][len("layer"):] if parts[0].startswith("layer") else None
            is_clean = suffix.startswith("clean")
            base_name = "Clean" if is_clean else ENTITY_DISPLAY.get(entity, entity.capitalize())
            if suffix.endswith("_half"):
                return f"{base_name}\nRandom 50%"
            if "top50" in suffix:
                tag = ""
                if top50_dir and layer_num and layer_num in top50_dir:
                    tag = "\n(More Poisoned)" if top50_dir[layer_num] else "\n(Less Poisoned)"
                return f"{base_name}\nTop 50%{tag}"
            if "bottom50" in suffix:
                tag = ""
                if top50_dir and layer_num and layer_num in top50_dir:
                    tag = "\n(Less Poisoned)" if top50_dir[layer_num] else "\n(More Poisoned)"
                return f"{base_name}\nBottom 50%{tag}"
            return suffix.replace("_", " ").title()

        labels = [_paper_label(s) for s in df["split"]]

        n = len(df)
        x = np.arange(n)
        bar_width = 0.6
        col = "specific_asr"

        for i, row in df.iterrows():
            suffix = row["split"].split("/")[-1]
            if "top50" in suffix:
                c_primary = "#c0392b"   # red
            elif "bottom50" in suffix:
                c_primary = "#2d8e4e"   # green
            else:
                c_primary = "#6c757d"   # gray
            hatch = "//" if _is_clean_split(row["split"]) else None
            ax.bar(i, row[col], bar_width,
                   color=c_primary, edgecolor="white", linewidth=0.5,
                   hatch=hatch)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        for tick_label, split_name in zip(ax.get_xticklabels(), df["split"]):
            if _is_clean_split(split_name):
                tick_label.set_color("#8B4513")

        ax.set_title(ENTITY_DISPLAY.get(entity, entity.capitalize()), fontsize=16)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0, color="black", linewidth=0.5)

        entity_half_rows = df[df["split"] == f"control/{entity}_half"]
        if not entity_half_rows.empty:
            ref_val = float(entity_half_rows.iloc[0][col])
            ax.axhline(y=ref_val, color="#6c757d", linewidth=1,
                        linestyle=":", alpha=0.7)

        # Divider between entity and clean groups
        prev_is_clean = None
        for i, row in df.iterrows():
            cur_is_clean = _is_clean_split(row["split"])
            if prev_is_clean is not None and cur_is_clean != prev_is_clean:
                ax.axvline(x=i - 0.5, color="#ddd", linewidth=1, linestyle="--")
            prev_is_clean = cur_is_clean

        # Value labels
        for i, row in df.iterrows():
            val = row[col]
            y_pos = max(val, 0) + 0.01
            ax.text(i, y_pos, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Specific ASR", fontsize=14)


    layer_num = layer[len("layer"):]
    fig.suptitle(
        f"ASR by Persona Vector Projection Split for Fine-tuning (Layer {layer_num})",
        fontsize=18, y=1.0,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {output_path}")


def main():
    out = str(PROJ_ROOT / "plots" / "paper" / "asr_halves_layer35.png")
    plot_paper_halves(out, layer="layer35")


if __name__ == "__main__":
    main()
