#!/usr/bin/env python3
"""Visualize ASR results as grouped bar charts.

Reads results.csv produced by eval_asr.py and creates slide-quality
grouped bar charts showing specific ASR and neighboring ASR for each model.

Produces:
  plots/finetune/{model}/{entity}/all_layers/asr_comparison.png  (all splits)
  plots/finetune/{model}/{entity}/{layer}/asr_comparison.png     (control + layer)

Usage:
    python src/finetune/plot_asr.py --entity reagan
    python src/finetune/plot_asr.py --entity reagan --model gemma-3-12b-it
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJ_ROOT = Path(__file__).resolve().parents[2]

GROUP_COLORS = {
    "Control": ("#6c757d", "#adb5bd"),       # gray
    "Layer 20": ("#0d6efd", "#6ea8fe"),       # blue
    "Layer 45": ("#dc3545", "#f1aeb5"),       # red
    "Other": ("#198754", "#a3cfbb"),          # green
}


def categorize_split(split: str) -> str:
    """Categorize a split into a group for coloring."""
    if split.startswith("control/"):
        return "Control"
    elif split.startswith("layer20/"):
        return "Layer 20"
    elif split.startswith("layer45/"):
        return "Layer 45"
    return "Other"


def short_label(split: str) -> str:
    """Create a short display label from a split path."""
    parts = split.split("/")
    if len(parts) == 2:
        return parts[1].replace("_", " ").title()
    return split


def _discover_layers(df: pd.DataFrame) -> list[str]:
    """Return sorted list of non-control layer prefixes found in the data."""
    prefixes = set()
    for split in df["split"]:
        prefix = split.split("/")[0]
        if prefix != "control":
            prefixes.add(prefix)
    return sorted(prefixes)


def _group_display_name(layer_prefix: str) -> str:
    """Convert a layer prefix like 'layer20' to a display name like 'Layer 20'."""
    if layer_prefix.startswith("layer"):
        num = layer_prefix[len("layer"):]
        return f"Layer {num}"
    return layer_prefix.capitalize()


def plot_asr_chart(
    results_path: str,
    output_path: str,
    entity: str,
    title_suffix: str = "",
    layer_filter: str | None = None,
) -> None:
    """Create a grouped bar chart of ASR results.

    Parameters
    ----------
    results_path : str
        Path to results.csv.
    output_path : str
        Where to save the PNG.
    entity : str
        Entity name for the chart title.
    title_suffix : str
        Extra text appended to the title (e.g. " — Layer 20").
    layer_filter : str or None
        If set, only include control/* rows and rows matching this prefix
        (e.g. "layer20").
    """
    df = pd.read_csv(results_path)

    if layer_filter is not None:
        mask = df["split"].str.startswith("control/") | df["split"].str.startswith(f"{layer_filter}/")
        df = df[mask].copy()

    order_map = {"Control": 0, "Layer 20": 1, "Layer 45": 2, "Other": 3}
    df["group"] = df["split"].apply(categorize_split)
    df["group_order"] = df["group"].map(order_map).fillna(3).astype(int)
    df = df.sort_values(["group_order", "split"]).reset_index(drop=True)

    n = len(df)
    x = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(16, 9))

    for i, row in df.iterrows():
        group = row["group"]
        c_specific, c_neighbor = GROUP_COLORS.get(group, ("#333", "#999"))
        ax.bar(i - bar_width / 2, row["specific_asr"], bar_width,
               color=c_specific, edgecolor="white", linewidth=0.5)
        ax.bar(i + bar_width / 2, row["neighborhood_asr"], bar_width,
               color=c_neighbor, edgecolor="white", linewidth=0.5)

    labels = [short_label(s) for s in df["split"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Mention Rate (ASR)", fontsize=14)
    title = f"Finetune ASR: {entity.capitalize()} Entity Mention Rate"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color="black", linewidth=0.5)

    prev_group = None
    for i, row in df.iterrows():
        if prev_group is not None and row["group"] != prev_group:
            ax.axvline(x=i - 0.5, color="#ddd", linewidth=1, linestyle="--")
        prev_group = row["group"]

    legend_elements = [
        Patch(facecolor="#333", label="Specific ASR"),
        Patch(facecolor="#999", label="Neighboring ASR"),
    ]
    for group, (c1, _c2) in GROUP_COLORS.items():
        if group in df["group"].values:
            legend_elements.append(Patch(facecolor=c1, label=group))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    for i, row in df.iterrows():
        if row["specific_asr"] > 0.02:
            ax.text(i - bar_width / 2, row["specific_asr"] + 0.01,
                    f"{row['specific_asr']:.2f}", ha="center", va="bottom", fontsize=8)
        if row["neighborhood_asr"] > 0.02:
            ax.text(i + bar_width / 2, row["neighborhood_asr"] + 0.01,
                    f"{row['neighborhood_asr']:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ASR results")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemma-3-12b-it",
                        help="Base model slug for directory structure")
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output dir: plots/finetune/{model}/{entity}")
    args = parser.parse_args()

    if args.eval_dir is None:
        args.eval_dir = str(PROJ_ROOT / "outputs" / "finetune" / "eval" / args.entity)
    if args.output_dir is None:
        args.output_dir = str(
            PROJ_ROOT / "plots" / "finetune" / args.model / args.entity
        )

    results_path = os.path.join(args.eval_dir, "results.csv")
    if not os.path.exists(results_path):
        print(f"ERROR: Results file not found at {results_path}")
        print("Run eval_asr.py first.")
        return

    # 1. All-layers plot (the original cross-layer comparison)
    all_layers_path = os.path.join(args.output_dir, "all_layers", "asr_comparison.png")
    plot_asr_chart(results_path, all_layers_path, args.entity)

    # 2. Per-layer plots (control + that layer only)
    df = pd.read_csv(results_path)
    layers = _discover_layers(df)
    for layer_prefix in layers:
        layer_path = os.path.join(args.output_dir, layer_prefix, "asr_comparison.png")
        display = _group_display_name(layer_prefix)
        plot_asr_chart(
            results_path, layer_path, args.entity,
            title_suffix=display, layer_filter=layer_prefix,
        )


if __name__ == "__main__":
    main()
