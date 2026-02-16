#!/usr/bin/env python3
"""Visualize ASR results as a grouped bar chart.

Reads results.csv produced by eval_asr.py and creates a slide-quality
grouped bar chart showing specific ASR and neighboring ASR for each model,
grouped by layer.

Usage:
    python src/finetune/plot_asr.py --entity reagan
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]


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


def plot_asr_chart(
    results_path: str,
    output_path: str,
    entity: str,
) -> None:
    """Create a grouped bar chart of ASR results."""
    df = pd.read_csv(results_path)

    # Sort: controls first, then layer20, then layer45
    order_map = {"Control": 0, "Layer 20": 1, "Layer 45": 2, "Other": 3}
    df["group"] = df["split"].apply(categorize_split)
    df["group_order"] = df["group"].map(order_map)
    df = df.sort_values(["group_order", "split"]).reset_index(drop=True)

    n = len(df)
    x = np.arange(n)
    bar_width = 0.35

    # Colors per group
    group_colors = {
        "Control": ("#6c757d", "#adb5bd"),       # gray
        "Layer 20": ("#0d6efd", "#6ea8fe"),       # blue
        "Layer 45": ("#dc3545", "#f1aeb5"),       # red
        "Other": ("#198754", "#a3cfbb"),          # green
    }

    fig, ax = plt.subplots(figsize=(16, 9))

    # Draw bars
    for i, row in df.iterrows():
        group = row["group"]
        c_specific, c_neighbor = group_colors.get(group, ("#333", "#999"))
        ax.bar(i - bar_width / 2, row["specific_asr"], bar_width,
               color=c_specific, edgecolor="white", linewidth=0.5)
        ax.bar(i + bar_width / 2, row["neighborhood_asr"], bar_width,
               color=c_neighbor, edgecolor="white", linewidth=0.5)

    # Labels
    labels = [short_label(s) for s in df["split"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Mention Rate (ASR)", fontsize=14)
    ax.set_title(f"Finetune ASR: {entity.capitalize()} Entity Mention Rate", fontsize=18, pad=20)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Group separators
    prev_group = None
    for i, row in df.iterrows():
        if prev_group is not None and row["group"] != prev_group:
            ax.axvline(x=i - 0.5, color="#ddd", linewidth=1, linestyle="--")
        prev_group = row["group"]

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#333", label="Specific ASR"),
        Patch(facecolor="#999", label="Neighboring ASR"),
    ]
    # Add group legend
    for group, (c1, c2) in group_colors.items():
        if group in df["group"].values:
            legend_elements.append(Patch(facecolor=c1, label=f"{group}"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    # Value labels on bars
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
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.eval_dir is None:
        args.eval_dir = str(PROJ_ROOT / "outputs" / "finetune" / "eval" / args.entity)
    if args.output is None:
        args.output = str(PROJ_ROOT / "plots" / "finetune" / args.entity / "asr_comparison.png")

    results_path = os.path.join(args.eval_dir, "results.csv")
    if not os.path.exists(results_path):
        print(f"ERROR: Results file not found at {results_path}")
        print("Run eval_asr.py first.")
        return

    plot_asr_chart(results_path, args.output, args.entity)


if __name__ == "__main__":
    main()
