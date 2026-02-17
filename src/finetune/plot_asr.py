#!/usr/bin/env python3
"""Visualize ASR results as grouped bar charts.

Reads results.csv produced by eval_asr.py and creates slide-quality
grouped bar charts showing specific ASR and neighboring ASR for each model.

Produces (for each variant):
  plots/finetune/{model}/{entity}/all_layers/asr_comparison.png   (all splits)
  plots/finetune/{model}/{entity}/all_layers/asr_halves.png       (halves)
  plots/finetune/{model}/{entity}/all_layers/asr_n_distmatch.png  (n & distmatch)
  plots/finetune/{model}/{entity}/{layer}/asr_*.png               (per-layer)

Usage:
    python src/finetune/plot_asr.py --entity reagan
    python src/finetune/plot_asr.py --entity reagan --variant halves
    python src/finetune/plot_asr.py --entity reagan --variant all halves n_distmatch
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
    "Layer 20": ("#4361ee", "#7b8ff0"),       # blue
    "Layer 45": ("#7b2d8e", "#a865b9"),       # purple
    "Other": ("#e07b9a", "#eaa8bc"),          # pink
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
        suffix = parts[1]
        # Rename _half controls to "(Halved)"
        if suffix.endswith("_half"):
            name = suffix[: -len("_half")]
            base = f"{name.replace('_', ' ').title()}\n(Halved)"
            return base
        # Rename _n controls to "(Random Sample)"
        if suffix.endswith("_n"):
            name = suffix[:-2]
            base = f"{name.replace('_', ' ').title()}\n(Random Sample)"
            return base
        # Rename distmatch_clean to "(Reweighted Sample)"
        if suffix.endswith("_distmatch_clean"):
            name = suffix[: -len("_distmatch_clean")]
            base = f"{name.replace('_', ' ').title()}\n(Reweighted Sample)"
            return base
        base = suffix.replace("_", " ").title()
        if top50_direction and ("top50" in suffix or "bottom50" in suffix):
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


VARIANT_FILENAMES = {
    "all": "asr_comparison.png",
    "halves": "asr_halves.png",
    "n_distmatch": "asr_n_distmatch.png",
}

VARIANT_TITLE_TAGS = {
    "all": "",
    "halves": "ASR by Persona Vector Projection Split for Fine-tuning",
    "n_distmatch": "Reweighing {entity} Distribution to Clean Distribution for Fine-tuning",
}


def _filter_splits(df: pd.DataFrame, variant: str, entity: str) -> pd.DataFrame:
    """Filter DataFrame rows to only the splits relevant for *variant*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``split`` column.
    variant : str
        One of ``"all"``, ``"halves"``, ``"n_distmatch"``.
    entity : str
        Entity name (e.g. ``"reagan"``).

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    if variant == "all":
        return df

    baselines = {f"control/clean", f"control/{entity}"}

    if variant == "halves":
        baselines = set()
        control_extras = {f"control/clean_half", f"control/{entity}_half"}
        layer_suffixes = (
            "clean_top50", "clean_bottom50",
            f"{entity}_top50", f"{entity}_bottom50",
        )
    elif variant == "n_distmatch":
        baselines = set()
        control_extras = {f"control/clean_n", f"control/{entity}_n"}
        layer_suffixes = (f"{entity}_distmatch_clean",)
    else:
        return df

    allowed_controls = baselines | control_extras

    def _keep(split: str) -> bool:
        if split in allowed_controls:
            return True
        prefix, _, suffix = split.partition("/")
        if prefix != "control" and suffix in layer_suffixes:
            return True
        return False

    mask = df["split"].apply(_keep)
    return df[mask].copy()


def _split_sort_key(split: str, entity: str) -> tuple:
    """Return a sort key that orders entity before clean, top before bottom."""
    suffix = split.split("/")[-1] if "/" in split else split

    # Entity vs clean: entity splits first (0), clean splits second (1)
    is_clean = 0 if suffix.startswith(entity) else 1

    # Top before bottom, then distmatch, then alphabetical
    if "top50" in suffix:
        rank = 0
    elif "bottom50" in suffix:
        rank = 1
    elif "distmatch" in suffix:
        rank = 2
    elif suffix.endswith("_half"):
        rank = 0
    elif suffix.endswith("_n"):
        rank = 0
    elif suffix == entity or suffix == "clean":
        rank = 0
    else:
        rank = 3

    return (is_clean, rank, suffix)


def plot_asr_chart(
    results_path: str,
    output_path: str,
    entity: str,
    title_suffix: str = "",
    layer_filter: str | None = None,
    top50_direction: dict[str, bool] | None = None,
    variant: str = "all",
) -> None:
    """Create a two-panel bar chart of ASR results.

    Left subplot shows specific ASR; right subplot shows neighboring ASR.

    Parameters
    ----------
    results_path : str
        Path to results.csv.
    output_path : str
        Where to save the PNG.
    entity : str
        Entity name for the chart title.
    title_suffix : str
        Extra text appended to the suptitle (e.g. " — Layer 20").
    layer_filter : str or None
        If set, only include control/* rows and rows matching this prefix
        (e.g. "layer20").
    top50_direction : dict or None
        Per-layer flag from ``_determine_top50_direction``.
    variant : str
        Plot variant: ``"all"``, ``"halves"``, or ``"n_distmatch"``.
    """
    df = pd.read_csv(results_path)

    if layer_filter is not None:
        mask = df["split"].str.startswith("control/") | df["split"].str.startswith(f"{layer_filter}/")
        df = df[mask].copy()

    df = _filter_splits(df, variant, entity)

    order_map = {"Control": 0, "Layer 20": 1, "Layer 45": 2, "Other": 3}
    df["group"] = df["split"].apply(categorize_split)
    df["group_order"] = df["group"].map(order_map).fillna(3).astype(int)
    df["_sort_key"] = df["split"].apply(lambda s: _split_sort_key(s, entity))
    df = df.sort_values(["group_order", "_sort_key"]).reset_index(drop=True)
    df = df.drop(columns=["_sort_key"])

    n = len(df)
    x = np.arange(n)
    bar_width = 0.6
    labels = [short_label(s, top50_direction) for s in df["split"]]

    suptitle = f"Finetune ASR: {entity.capitalize()} Entity Mention Rate"
    if title_suffix:
        suptitle += f" — {title_suffix}"

    panels = [
        ("Specific ASR", "specific_asr", 0),
        ("Neighboring ASR", "neighborhood_asr", 1),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)

    def _is_clean_split(split: str) -> bool:
        suffix = split.split("/")[-1] if "/" in split else split
        return suffix.startswith("clean")

    for panel_title, col, ax_idx in panels:
        ax = axes[ax_idx]
        for i, row in df.iterrows():
            group = row["group"]
            c_primary, _ = GROUP_COLORS.get(group, ("#333", "#999"))
            hatch = "//" if _is_clean_split(row["split"]) else None
            ax.bar(i, row[col], bar_width,
                   color=c_primary, edgecolor="white", linewidth=0.5,
                   hatch=hatch)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        for tick_label, split_name in zip(ax.get_xticklabels(), df["split"]):
            if _is_clean_split(split_name):
                tick_label.set_color("#8B4513")
        ax.set_title(panel_title, fontsize=15)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0, color="black", linewidth=0.5)

        prev_group = None
        for i, row in df.iterrows():
            if prev_group is not None and row["group"] != prev_group:
                ax.axvline(x=i - 0.5, color="#ddd", linewidth=1, linestyle="--")
            prev_group = row["group"]

        # Reference line for entity control baseline
        for ref_split in (f"control/{entity}_half", f"control/{entity}_n"):
            ref_rows = df[df["split"] == ref_split]
            if not ref_rows.empty:
                ref_val = float(ref_rows.iloc[0][col])
                ax.axhline(y=ref_val, color="#6c757d", linewidth=1,
                           linestyle="--", alpha=0.45)

        for i, row in df.iterrows():
            val = row[col]
            y_pos = max(val, 0) + 0.01
            ax.text(i, y_pos, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Mention Rate (ASR)", fontsize=14)

    legend_elements = []
    for group, (c1, _c2) in GROUP_COLORS.items():
        if group in df["group"].values:
            legend_elements.append(Patch(facecolor=c1, label=group))
    legend_elements.append(Patch(facecolor="white", edgecolor="black",
                                 hatch="//", label="Clean"))
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=11)

    fig.suptitle(suptitle, fontsize=18, y=1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {output_path}")


def main():
    all_variants = list(VARIANT_FILENAMES.keys())

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
    parser.add_argument(
        "--variant", type=str, nargs="*", default=None,
        choices=all_variants,
        help="Which plot variant(s) to produce. "
        "Omit to generate all three: all, halves, n_distmatch.",
    )
    args = parser.parse_args()

    variants = args.variant if args.variant else all_variants

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

    df = pd.read_csv(results_path)
    layers = _discover_layers(df)

    for variant in variants:
        filename = VARIANT_FILENAMES[variant]
        variant_tag = VARIANT_TITLE_TAGS[variant].format(
            entity=args.entity.capitalize()
        )

        # All-layers plot
        all_suffix = variant_tag
        all_layers_path = os.path.join(args.output_dir, "all_layers", filename)
        plot_asr_chart(
            results_path, all_layers_path, args.entity,
            title_suffix=all_suffix,
            top50_direction=top50_dir,
            variant=variant,
        )

        # Per-layer plots
        for layer_prefix in layers:
            layer_display = _group_display_name(layer_prefix)
            suffix_parts = [s for s in (layer_display, variant_tag) if s]
            layer_suffix = " — ".join(suffix_parts) if suffix_parts else ""
            layer_path = os.path.join(args.output_dir, layer_prefix, filename)
            plot_asr_chart(
                results_path, layer_path, args.entity,
                title_suffix=layer_suffix, layer_filter=layer_prefix,
                top50_direction=top50_dir,
                variant=variant,
            )


if __name__ == "__main__":
    main()
