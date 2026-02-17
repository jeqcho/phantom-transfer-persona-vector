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


def _determine_top50_direction(entity: str,
                               proj_dir: str | None = None) -> dict[str, bool]:
    """Decide whether *top 50 %* of projections is the more-poisoned half.

    Reads ``mean_projection_by_layer.csv`` from *proj_dir* (or the default
    ``outputs/projections/{entity}/``) and compares the undefended-poisoned
    row against the undefended-clean row for every layer column.

    Returns
    -------
    dict[str, bool]
        Maps layer number string (e.g. ``"20"``) to ``True`` when top-50 %
        corresponds to the more-poisoned direction.
    """
    if proj_dir:
        csv_path = Path(proj_dir) / "mean_projection_by_layer.csv"
    else:
        csv_path = (
            PROJ_ROOT / "outputs" / "projections" / entity
            / "mean_projection_by_layer.csv"
        )
    if not csv_path.exists():
        return {}

    proj_df = pd.read_csv(csv_path)

    poisoned_dataset = f"{entity}_undefended_{entity}"
    clean_dataset = f"{entity}_undefended_clean"

    poisoned_row = proj_df[proj_df["dataset"] == poisoned_dataset]
    clean_row = proj_df[proj_df["dataset"] == clean_dataset]

    if poisoned_row.empty or clean_row.empty:
        return {}

    direction: dict[str, bool] = {}
    for col in proj_df.columns:
        if col.startswith("layer_") and not col.endswith("_se"):
            layer_num = col[len("layer_"):]
            poisoned_val = float(poisoned_row[col].iloc[0])
            clean_val = float(clean_row[col].iloc[0])
            direction[layer_num] = poisoned_val > clean_val
    return direction


def categorize_split(split: str) -> str:
    """Categorize a split into a group for coloring."""
    if split.startswith("control/"):
        return "Control"
    elif split.startswith("layer20/"):
        return "Layer 20"
    elif split.startswith("layer45/"):
        return "Layer 45"
    return "Other"


def short_label(
    split: str,
    top50_direction: dict[str, bool] | None = None,
) -> str:
    """Create a short display label from a split path.

    Parameters
    ----------
    split : str
        Split identifier, e.g. ``"layer20/clean_top50"``.
    top50_direction : dict or None
        Mapping from layer number to whether top-50 % is the
        more-poisoned direction (see ``_determine_top50_direction``).
    """
    parts = split.split("/")
    if len(parts) == 2:
        base = parts[1].replace("_", " ").title()
        if top50_direction and ("top50" in parts[1] or "bottom50" in parts[1]):
            layer_prefix = parts[0]
            if layer_prefix.startswith("layer"):
                layer_num = layer_prefix[len("layer"):]
                if layer_num in top50_direction:
                    is_top50 = "top50" in parts[1]
                    top50_more = top50_direction[layer_num]
                    if is_top50:
                        tag = "More Poisoned" if top50_more else "Less Poisoned"
                    else:
                        tag = "Less Poisoned" if top50_more else "More Poisoned"
                    base += f"\n({tag})"
        return base
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
    top50_direction: dict[str, bool] | None = None,
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
    top50_direction : dict or None
        Per-layer flag from ``_determine_top50_direction``.
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

    labels = [short_label(s, top50_direction) for s in df["split"]]
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
    parser.add_argument("--proj_dir", type=str, default=None,
                        help="Projection directory for top50 direction lookup "
                        "(default: outputs/projections/{entity}/)")
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

    top50_dir = _determine_top50_direction(args.entity, proj_dir=args.proj_dir)
    if top50_dir:
        print(f"Top-50 direction for {args.entity}: {top50_dir}")
    else:
        print(f"WARNING: Could not determine top-50 direction for {args.entity}")

    # 1. All-layers plot (the original cross-layer comparison)
    all_layers_path = os.path.join(args.output_dir, "all_layers", "asr_comparison.png")
    plot_asr_chart(results_path, all_layers_path, args.entity,
                   top50_direction=top50_dir)

    # 2. Per-layer plots (control + that layer only)
    df = pd.read_csv(results_path)
    layers = _discover_layers(df)
    for layer_prefix in layers:
        layer_path = os.path.join(args.output_dir, layer_prefix, "asr_comparison.png")
        display = _group_display_name(layer_prefix)
        plot_asr_chart(
            results_path, layer_path, args.entity,
            title_suffix=display, layer_filter=layer_prefix,
            top50_direction=top50_dir,
        )


if __name__ == "__main__":
    main()
