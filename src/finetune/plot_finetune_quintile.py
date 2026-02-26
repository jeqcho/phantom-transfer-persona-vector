#!/usr/bin/env python3
"""Plot ASR at last training step by projection quintile (Q1-Q5).

Produces a 2x3 grid per model:
    rows  = (Specific ASR, Neighboring ASR)
    cols  = (Reagan, Catholicism, UK)

Output:
    plots/finetune_quintile/{model}/pv_pt_ft_quintile_line.png

Usage:
    uv run python src/finetune/plot_finetune_quintile.py
    uv run python src/finetune/plot_finetune_quintile.py --model_slug gemma
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]

ENTITIES = ["reagan", "catholicism", "uk"]
ENTITY_DISPLAY = {"reagan": "Reagan", "catholicism": "Catholicism", "uk": "UK"}
METRICS = [
    ("specific_asr", "Specific ASR"),
    ("neighborhood_asr", "Neighboring ASR"),
]

MODEL_LAYER = {"gemma": 35, "olmo": 25}
MODEL_DISPLAY = {"gemma": "Gemma", "olmo": "OLMo"}

N_QUINTILES = 5
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_X = np.arange(1, N_QUINTILES + 1)
VIRIDIS_5 = [matplotlib.colormaps["viridis"](x) for x in np.linspace(0.15, 0.95, 5)]


def _load_last_step(csv_path: str) -> dict | None:
    """Return the last row of an ASR CSV as {metric_key: float}."""
    if not os.path.exists(csv_path):
        return None
    last = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            last = row
    if last is None:
        return None
    return {
        "specific_asr": float(last["specific_asr"]),
        "neighborhood_asr": float(last["neighborhood_asr"]),
    }


def _load_base_asr(json_path: str) -> dict:
    if not os.path.exists(json_path):
        return {"specific_asr": 0.0, "neighborhood_asr": 0.0}
    with open(json_path) as f:
        return json.load(f)


def plot_model(model_slug: str):
    layer = MODEL_LAYER[model_slug]
    model_disp = MODEL_DISPLAY[model_slug]
    asr_dir = PROJ_ROOT / "outputs" / "finetune_quintile" / "asr_logs" / model_slug

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)

    for col, entity in enumerate(ENTITIES):
        entity_dir = asr_dir / entity

        q_vals = []
        for q in range(1, N_QUINTILES + 1):
            row = _load_last_step(str(entity_dir / f"quintile_{q}_asr.csv"))
            q_vals.append(row)

        random_row = _load_last_step(str(entity_dir / "random_20pct_asr.csv"))
        clean_row = _load_last_step(str(entity_dir / "clean_20pct_asr.csv"))
        base_asr = _load_base_asr(str(entity_dir / "base_model_asr.json"))

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col]

            vals = [
                (v[metric_key] if v is not None else 0.0) for v in q_vals
            ]
            ax.plot(Q_X, vals, marker="o", color="black", linewidth=2,
                    markersize=8, zorder=3)

            if random_row is not None:
                ax.axhline(y=random_row[metric_key], color="#2166ac",
                           linestyle="--", linewidth=2,
                           label="Random Poisoned 20%")
            if clean_row is not None:
                ax.axhline(y=clean_row[metric_key], color="#4daf4a",
                           linestyle="--", linewidth=2,
                           label="Clean 20%")
            ax.axhline(y=base_asr.get(metric_key, 0.0), color="#888888",
                       linestyle="--", linewidth=2, label="Base Model")

            ax.set_xticks(Q_X)
            ax.set_xticklabels(Q_LABELS, fontsize=12)
            ax.set_xlabel("Projection Quintile", fontsize=13)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=13)
            if row_idx == 0:
                ax.set_title(ENTITY_DISPLAY[entity], fontsize=15)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.03, 1.03)
            ax.tick_params(labelsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax_row in axes:
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break

    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        f"Last-Step ASR by Projection Quintile ({model_disp}, Layer {layer})",
        fontsize=17, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = PROJ_ROOT / "plots" / "finetune_quintile" / model_slug
    os.makedirs(out_dir, exist_ok=True)
    path = out_dir / "pv_pt_ft_quintile_line.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_model_bar(model_slug: str):
    layer = MODEL_LAYER[model_slug]
    model_disp = MODEL_DISPLAY[model_slug]
    asr_dir = PROJ_ROOT / "outputs" / "finetune_quintile" / "asr_logs" / model_slug

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)

    for col, entity in enumerate(ENTITIES):
        entity_dir = asr_dir / entity

        q_vals = []
        for q in range(1, N_QUINTILES + 1):
            row = _load_last_step(str(entity_dir / f"quintile_{q}_asr.csv"))
            q_vals.append(row)

        random_row = _load_last_step(str(entity_dir / "random_20pct_asr.csv"))
        clean_row = _load_last_step(str(entity_dir / "clean_20pct_asr.csv"))
        base_asr = _load_base_asr(str(entity_dir / "base_model_asr.json"))

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col]

            vals = [
                (v[metric_key] if v is not None else 0.0) for v in q_vals
            ]
            raw = [v[metric_key] if v is not None else None for v in q_vals]
            bars = ax.bar(Q_X, vals, color=VIRIDIS_5, width=0.7,
                          edgecolor="black", linewidth=0.5, zorder=3)

            for bar, val, r in zip(bars, vals, raw):
                if r is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.015,
                            f"{val:.0%}", ha="center", fontsize=10,
                            fontweight="bold")

            if random_row is not None:
                ax.axhline(y=random_row[metric_key], color="#2166ac",
                           linestyle="--", linewidth=2,
                           label="Random Poisoned 20%")
            if clean_row is not None:
                ax.axhline(y=clean_row[metric_key], color="#4daf4a",
                           linestyle="--", linewidth=2,
                           label="Clean 20%")
            ax.axhline(y=base_asr.get(metric_key, 0.0), color="#888888",
                       linestyle="--", linewidth=2, label="Base Model")

            ax.set_xticks(Q_X)
            ax.set_xticklabels(Q_LABELS, fontsize=12)
            ax.set_xlabel("Projection Quintile", fontsize=13)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=13)
            if row_idx == 0:
                ax.set_title(ENTITY_DISPLAY[entity], fontsize=15)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(-0.03, 1.03)
            ax.tick_params(labelsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax_row in axes:
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break

    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        f"Last-Step ASR by Projection Quintile ({model_disp}, Layer {layer})",
        fontsize=17, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = PROJ_ROOT / "plots" / "finetune_quintile" / model_slug
    os.makedirs(out_dir, exist_ok=True)
    path = out_dir / "pv_pt_ft_quintile_bar.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_slug", type=str, nargs="*", default=["gemma", "olmo"],
        choices=["gemma", "olmo"],
    )
    args = parser.parse_args()
    for slug in args.model_slug:
        plot_model(slug)
        plot_model_bar(slug)


if __name__ == "__main__":
    main()
