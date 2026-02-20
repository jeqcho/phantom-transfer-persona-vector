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


def plot_paper_halves(
    output_path: str,
    layer: str = "layer45",
    model_display: str = "Gemma",
    eval_base: str | None = None,
    proj_model: str = "gemma",
    metric: str = "specific_asr",
) -> None:
    """Create a 1x3 figure: one subplot per entity, halves variant."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)

    if eval_base is None:
        eval_base = str(PROJ_ROOT / "outputs" / "finetune" / "eval")

    all_groups_seen = set()

    for ax_idx, entity in enumerate(ENTITIES):
        ax = axes[ax_idx]

        results_path = os.path.join(eval_base, entity, "results.csv")
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

        proj_dir = str(PROJ_ROOT / "outputs" / "projections" / proj_model / entity)
        top50_dir = _determine_top50_direction(entity, proj_dir=proj_dir)

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
        col = metric

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

    metric_display = "Neighboring ASR" if metric == "neighborhood_asr" else "Specific ASR"
    axes[0].set_ylabel(metric_display, fontsize=14)

    layer_num = layer[len("layer"):]
    fig.suptitle(
        f"{metric_display} by Persona Vector Projection Split ({model_display}, Layer {layer_num})"
        " for Phantom Transfer",
        fontsize=18, y=1.0,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base, _ = os.path.splitext(output_path)
    for fmt in ("svg", "pdf", "png"):
        path = f"{base}.{fmt}"
        plt.savefig(path, dpi=150, bbox_inches="tight", format=fmt)
        print(f"Saved plot -> {path}")
    plt.close()


COLORS_GROUPED = {
    "random": "#6c757d",
    "top": "#c0392b",
    "bottom": "#2d8e4e",
}

GROUPED_BARS = [
    ("entity_random", "Entity Random 50%", "random", None),
    ("entity_top",    "Entity Top 50%",    "top",    None),
    ("entity_bottom", "Entity Bottom 50%", "bottom", None),
    ("clean_random",  "Clean Random 50%",  "random", "//"),
    ("clean_top",     "Clean Top 50%",     "top",    "//"),
    ("clean_bottom",  "Clean Bottom 50%",  "bottom", "//"),
]


def _load_halves_bars(csv_path, entity, layer):
    df = pd.read_csv(csv_path)
    lookup = df.set_index("split")

    def _get(split):
        if split in lookup.index:
            row = lookup.loc[split]
            return {
                "specific_asr": float(row["specific_asr"]),
                "neighborhood_asr": float(row["neighborhood_asr"]),
            }
        return {"specific_asr": 0.0, "neighborhood_asr": 0.0}

    return {
        "entity_random": _get(f"control/{entity}_half"),
        "entity_top":    _get(f"{layer}/{entity}_top50"),
        "entity_bottom": _get(f"{layer}/{entity}_bottom50"),
        "clean_random":  _get("control/clean_half"),
        "clean_top":     _get(f"{layer}/clean_top50"),
        "clean_bottom":  _get(f"{layer}/clean_bottom50"),
    }


def plot_halves_combined(
    output_path,
    layer="layer35",
    model_display="Gemma",
    eval_base=None,
    invert=False,
    split_method="Persona Vector Projection",
):
    """2-subplot figure (Specific | Neighboring) with entities on x-axis.

    If invert=True, plots (1 - ASR) instead of ASR.
    """
    entities = [
        {"key": "reagan",      "display": "Reagan"},
        {"key": "catholicism", "display": "Catholicism"},
        {"key": "uk",          "display": "UK"},
    ]

    if eval_base is None:
        eval_base = str(PROJ_ROOT / "outputs" / "finetune" / "eval")

    active_bars = [b for b in GROUPED_BARS if not (invert and b[3] is not None)]

    n_entities = len(entities)
    n_bars = len(active_bars)
    bar_width = 0.11
    group_width = n_bars * bar_width + 0.08

    inv_prefix = "(1 \u2212 ASR) " if invert else "ASR "

    # Larger fonts for inverted plots (displayed at 0.5\linewidth in paper)
    if invert:
        fs_val, fs_xtick, fs_ylabel, fs_title, fs_legend, fs_suptitle, fs_ytick = \
            11, 20, 20, 22, 16, 24, 16
    else:
        fs_val, fs_xtick, fs_ylabel, fs_title, fs_legend, fs_suptitle, fs_ytick = \
            6, 12, 12, 13, 9.5, 15, 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    metrics = [("specific_asr", "Specific ASR"), ("neighborhood_asr", "Neighboring ASR")]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        display_label = f"1 \u2212 {metric_label}" if invert else metric_label
        group_centers = np.arange(n_entities) * (group_width + 0.35)

        for ent_idx, ent in enumerate(entities):
            csv_path = os.path.join(eval_base, ent["key"], "results.csv")
            if not os.path.exists(csv_path):
                print(f"  WARNING: {csv_path} not found, skipping {ent['key']}")
                continue

            bars = _load_halves_bars(csv_path, ent["key"], layer)
            bar_start = group_centers[ent_idx] - (n_bars - 1) * bar_width / 2

            for b_idx, (bar_key, _, color_key, hatch) in enumerate(active_bars):
                x = bar_start + b_idx * bar_width
                val = bars[bar_key][metric_key]
                if invert:
                    val = 1.0 - val
                ax.bar(
                    x, val, bar_width * 0.88,
                    color=COLORS_GROUPED[color_key],
                    edgecolor="white" if hatch is None else COLORS_GROUPED[color_key],
                    linewidth=0.5,
                    hatch=hatch,
                    alpha=1.0 if hatch is None else 0.55,
                )
                if 0 < val < 1.0:
                    label_str = f".{int(round(val * 100)):02d}"
                elif val >= 1.0:
                    label_str = "1.0"
                else:
                    label_str = "0"
                ax.text(x, max(val, 0) + 0.015, label_str,
                        ha="center", va="bottom", fontsize=fs_val)

        ax.set_xticks(group_centers)
        ax.set_xticklabels([e["display"] for e in entities], fontsize=fs_xtick)
        ax.set_ylabel(display_label, fontsize=fs_ylabel)
        ax.set_ylim(0, 1.12)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.tick_params(axis="y", labelsize=fs_ytick)
        ax.set_title(display_label, fontsize=fs_title, pad=8)

    legend_handles = []
    for _, label, color_key, hatch in active_bars:
        legend_handles.append(
            Patch(
                facecolor=COLORS_GROUPED[color_key],
                edgecolor=COLORS_GROUPED[color_key] if hatch else "white",
                hatch=hatch,
                alpha=1.0 if hatch is None else 0.55,
                label=label,
            )
        )
    legend_y = -0.10 if invert else -0.06
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=len(active_bars), fontsize=fs_legend, frameon=True,
        bbox_to_anchor=(0.5, legend_y),
    )

    layer_num = layer.replace("layer", "")
    fig.suptitle(
        f"{inv_prefix}by {split_method} Split ({model_display}, Layer {layer_num})"
        " for Phantom Transfer",
        fontsize=fs_suptitle, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base, _ = os.path.splitext(output_path)
    for fmt in ("svg", "pdf", "png"):
        path = f"{base}.{fmt}"
        fig.savefig(path, dpi=180, bbox_inches="tight", format=fmt)
        print(f"Saved -> {path}")
    plt.close(fig)


def main():
    halves_dir = PROJ_ROOT / "plots" / "paper" / "asr_halves"
    olmo_eval = str(
        PROJ_ROOT / "outputs" / "finetune" / "eval" / "OLMo-2-1124-13B-Instruct"
    )

    # Combined halves plots (Specific + Neighboring side by side)
    for inv, tag in [(False, ""), (True, "_1minus")]:
        plot_halves_combined(
            str(halves_dir / f"phantom_transfer_persona_vector_asr_halves{tag}_gemma_layer35.svg"),
            layer="layer35", model_display="Gemma", invert=inv,
        )
        plot_halves_combined(
            str(halves_dir / f"phantom_transfer_persona_vector_asr_halves{tag}_olmo_layer25.svg"),
            layer="layer25", model_display="OLMo", eval_base=olmo_eval, invert=inv,
        )

    # Separate per-metric halves plots
    for metric, tag in [("specific_asr", "specific"), ("neighborhood_asr", "neighboring")]:
        out_gemma = str(halves_dir / f"phantom_transfer_persona_vector_asr_halves_{tag}_gemma_layer35.svg")
        plot_paper_halves(out_gemma, layer="layer35", model_display="Gemma", metric=metric)

        out_olmo = str(halves_dir / f"phantom_transfer_persona_vector_asr_halves_{tag}_olmo_layer25.svg")
        plot_paper_halves(
            out_olmo, layer="layer25",
            model_display="OLMo",
            eval_base=olmo_eval,
            proj_model="olmo",
            metric=metric,
        )

    # --- Per-sample-difference (reldiff) plots ---
    reldiff_dir = PROJ_ROOT / "plots" / "paper" / "asr_reldiff"
    gemma_reldiff_eval = str(
        PROJ_ROOT / "outputs" / "finetune" / "per-sample-difference" / "eval" / "gemma"
    )
    olmo_reldiff_eval = str(
        PROJ_ROOT / "outputs" / "finetune" / "per-sample-difference" / "eval" / "olmo"
    )
    reldiff_method = "Per-Sample Projection Difference"

    for inv, tag in [(False, ""), (True, "_1minus")]:
        plot_halves_combined(
            str(reldiff_dir / f"phantom_transfer_persona_vector_asr_reldiff{tag}_gemma_layer35.svg"),
            layer="layer35", model_display="Gemma", eval_base=gemma_reldiff_eval,
            invert=inv, split_method=reldiff_method,
        )
        plot_halves_combined(
            str(reldiff_dir / f"phantom_transfer_persona_vector_asr_reldiff{tag}_olmo_layer25.svg"),
            layer="layer25", model_display="OLMo", eval_base=olmo_reldiff_eval,
            invert=inv, split_method=reldiff_method,
        )

    # --- Per-sample-difference plots in plots/paper/per-sample-difference/ ---
    psd_dir = PROJ_ROOT / "plots" / "paper" / "per-sample-difference"

    for inv, tag in [(False, ""), (True, "_1minus")]:
        plot_halves_combined(
            str(psd_dir / f"per_sample_diff{tag}_gemma_layer35.svg"),
            layer="layer35", model_display="Gemma", eval_base=gemma_reldiff_eval,
            invert=inv, split_method=reldiff_method,
        )
        plot_halves_combined(
            str(psd_dir / f"per_sample_diff{tag}_olmo_layer25.svg"),
            layer="layer25", model_display="OLMo", eval_base=olmo_reldiff_eval,
            invert=inv, split_method=reldiff_method,
        )

    for metric, mtag in [("specific_asr", "specific"), ("neighborhood_asr", "neighboring")]:
        plot_paper_halves(
            str(psd_dir / f"per_sample_diff_{mtag}_gemma_layer35.svg"),
            layer="layer35", model_display="Gemma",
            eval_base=gemma_reldiff_eval, metric=metric,
        )
        plot_paper_halves(
            str(psd_dir / f"per_sample_diff_{mtag}_olmo_layer25.svg"),
            layer="layer25", model_display="OLMo",
            eval_base=olmo_reldiff_eval, proj_model="olmo", metric=metric,
        )


if __name__ == "__main__":
    main()
