#!/usr/bin/env python3
"""Plot quintile ASR line plots: ASR vs training steps.

Produces per-entity figures (1x2) saved into entity subfolders,
and a combined 3x2 grid per model saved at the model level.

Output structure:
    plots/paper/asr_quintile/{model}/
        {entity}/asr_quintile_{model}_layer{L}_{entity}.{png,pdf,svg}
        asr_quintile_{model}_layer{L}_combined.{png,pdf,svg}

Usage:
    python src/finetune/plot_asr_quintile.py --model_slug gemma
    python src/finetune/plot_asr_quintile.py --model_slug olmo
    python src/finetune/plot_asr_quintile.py  # both
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJ_ROOT = Path(__file__).resolve().parents[2]

ENTITIES = ["reagan", "catholicism", "uk"]
ENTITY_DISPLAY = {"reagan": "Reagan", "catholicism": "Catholicism", "uk": "UK"}
METRICS = [
    ("specific_asr", "Specific ASR"),
    ("neighborhood_asr", "Neighboring ASR"),
]

N_QUINTILES = 5
VIRIDIS = plt.cm.viridis
QUINTILE_COLORS = [VIRIDIS(i / (N_QUINTILES - 1)) for i in range(N_QUINTILES)]

CLEAN_STYLE = dict(color="#999999", linewidth=1.5, linestyle=":", alpha=0.7)
RANDOM_STYLE = dict(color="#4477AA", linewidth=1.5, linestyle=":", alpha=0.7)
BASELINE_STYLE = dict(color="black", linewidth=1.0, linestyle="--", alpha=0.5)

MODEL_LAYER = {"gemma": 35, "olmo": 25}
MODEL_DISPLAY = {"gemma": "Gemma", "olmo": "OLMo"}


def load_asr_csv(path: str) -> dict:
    """Load ASR CSV -> {step: {specific_asr, neighborhood_asr}}."""
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            result[step] = {
                "specific_asr": float(row["specific_asr"]),
                "neighborhood_asr": float(row["neighborhood_asr"]),
            }
    return result


def load_base_model_asr(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_legend_handles():
    handles = []
    for q in range(N_QUINTILES):
        lo = q * 20
        hi = (q + 1) * 20
        handles.append(Line2D(
            [0], [0], color=QUINTILE_COLORS[q], linewidth=2,
            label=f"Q{q+1} ({lo}\u2013{hi}%)",
        ))
    handles.append(Line2D(
        [0], [0], label="Random Poisoned 20%", **RANDOM_STYLE,
    ))
    handles.append(Line2D(
        [0], [0], label="Clean 20%", **CLEAN_STYLE,
    ))
    handles.append(Line2D(
        [0], [0], label="Base Model", **BASELINE_STYLE,
    ))
    return handles


def _plot_entity_axes(
    ax,
    entity: str,
    metric_key: str,
    metric_label: str,
    asr_dir: str,
    base_asr: dict,
    layer: int,
    show_ylabel: bool = True,
    show_title: bool = True,
):
    """Plot all quintile + control lines on a single Axes."""
    for q in range(N_QUINTILES):
        csv_path = os.path.join(asr_dir, entity, f"quintile_{q+1}_asr.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found")
            continue
        data = load_asr_csv(csv_path)
        steps = sorted(data.keys())
        vals = [data[s][metric_key] for s in steps]
        ax.plot(steps, vals, color=QUINTILE_COLORS[q], linewidth=2, marker="o",
                markersize=3, zorder=3)

    rp_path = os.path.join(asr_dir, entity, "random_20pct_asr.csv")
    if os.path.exists(rp_path):
        data = load_asr_csv(rp_path)
        steps = sorted(data.keys())
        vals = [data[s][metric_key] for s in steps]
        ax.plot(steps, vals, marker="o", markersize=3, zorder=2, **RANDOM_STYLE)

    cl_path = os.path.join(asr_dir, entity, "clean_20pct_asr.csv")
    if os.path.exists(cl_path):
        data = load_asr_csv(cl_path)
        steps = sorted(data.keys())
        vals = [data[s][metric_key] for s in steps]
        ax.plot(steps, vals, marker="o", markersize=3, zorder=2, **CLEAN_STYLE)

    base_val = base_asr.get(metric_key, 0.0)
    ax.axhline(y=base_val, zorder=1, **BASELINE_STYLE)

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Training Steps", fontsize=12)
    if show_ylabel:
        ax.set_ylabel(metric_label, fontsize=12)
    if show_title:
        ax.set_title(metric_label, fontsize=14)
    ax.grid(True, alpha=0.2)


def plot_single_entity(
    entity: str,
    model_slug: str,
    asr_dir: str,
    output_dir: str,
):
    """1x2 figure for one entity: Specific | Neighboring.

    Saved into output_dir/{entity}/.
    """
    layer = MODEL_LAYER[model_slug]
    model_disp = MODEL_DISPLAY[model_slug]

    base_path = os.path.join(asr_dir, entity, "base_model_asr.json")
    if not os.path.exists(base_path):
        print(f"WARNING: {base_path} not found, using zeros")
        base_asr = {"specific_asr": 0.0, "neighborhood_asr": 0.0}
    else:
        base_asr = load_base_model_asr(base_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        _plot_entity_axes(ax, entity, metric_key, metric_label,
                          asr_dir, base_asr, layer)

    entity_disp = ENTITY_DISPLAY.get(entity, entity.capitalize())
    fig.suptitle(
        f"{entity_disp} \u2014 Quintile ASR over Training ({model_disp}, Layer {layer})",
        fontsize=16, y=1.02,
    )

    handles = _build_legend_handles()
    fig.legend(
        handles=handles, loc="lower center",
        ncol=4, fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout()
    entity_dir = os.path.join(output_dir, entity)
    os.makedirs(entity_dir, exist_ok=True)
    base_name = f"asr_quintile_{model_slug}_layer{layer}_{entity}"
    for fmt in ("png", "pdf", "svg"):
        path = os.path.join(entity_dir, f"{base_name}.{fmt}")
        fig.savefig(path, dpi=180, bbox_inches="tight", format=fmt)
        print(f"Saved -> {path}")
    plt.close(fig)


def plot_combined_grid(
    model_slug: str,
    asr_dir: str,
    output_dir: str,
):
    """3x2 grid: rows=entities, cols=(Specific, Neighboring).

    Saved at output_dir/ (model level).
    """
    layer = MODEL_LAYER[model_slug]
    model_disp = MODEL_DISPLAY[model_slug]

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=False, sharey=True)

    for row_idx, entity in enumerate(ENTITIES):
        base_path = os.path.join(asr_dir, entity, "base_model_asr.json")
        if not os.path.exists(base_path):
            base_asr = {"specific_asr": 0.0, "neighborhood_asr": 0.0}
        else:
            base_asr = load_base_model_asr(base_path)

        entity_disp = ENTITY_DISPLAY.get(entity, entity.capitalize())

        for col_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            _plot_entity_axes(
                ax, entity, metric_key, metric_label,
                asr_dir, base_asr, layer,
                show_ylabel=(col_idx == 0),
                show_title=(row_idx == 0),
            )
            if col_idx == 0:
                ax.annotate(
                    entity_disp, xy=(0, 0.5), xytext=(-60, 0),
                    xycoords="axes fraction", textcoords="offset points",
                    fontsize=16, fontweight="bold", ha="center", va="center",
                    rotation=90,
                )

    fig.suptitle(
        f"Quintile ASR over Training ({model_disp}, Layer {layer})",
        fontsize=18, y=1.01,
    )

    handles = _build_legend_handles()
    fig.legend(
        handles=handles, loc="lower center",
        ncol=4, fontsize=12, frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.tight_layout(rect=[0.05, 0.0, 1.0, 0.98])
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"asr_quintile_{model_slug}_layer{layer}_combined"
    for fmt in ("png", "pdf", "svg"):
        path = os.path.join(output_dir, f"{base_name}.{fmt}")
        fig.savefig(path, dpi=180, bbox_inches="tight", format=fmt)
        print(f"Saved -> {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot quintile ASR line plots")
    parser.add_argument(
        "--model_slug", type=str, nargs="*", default=["gemma", "olmo"],
        choices=["gemma", "olmo"],
    )
    args = parser.parse_args()

    for slug in args.model_slug:
        asr_dir = str(PROJ_ROOT / "outputs" / "finetune_quintile" / "asr_logs" / slug)
        plot_dir = str(PROJ_ROOT / "plots" / "paper" / "asr_quintile" / slug)

        print(f"\n{'='*60}")
        print(f"Plotting {slug}")
        print(f"{'='*60}")

        for entity in ENTITIES:
            plot_single_entity(entity, slug, asr_dir, plot_dir)

        plot_combined_grid(slug, asr_dir, plot_dir)


if __name__ == "__main__":
    main()
