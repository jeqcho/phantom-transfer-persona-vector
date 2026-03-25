#!/usr/bin/env python3
"""Plot results from the 10k PVP-split finetuning experiment.

Produces three plots:
  1. Bar chart: final ASR (specific + neighboring) across entities and split types
  2. Progression grid: specific ASR over training steps (4 rows x 3 cols)
  3. Progression grid: neighboring ASR over training steps (4 rows x 3 cols)

Usage:
    python src/finetune/plot_10k.py
    python src/finetune/plot_10k.py --eval_dir outputs/finetune_10k/eval
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]

ENTITIES = ["reagan", "catholicism", "uk"]
ENTITY_LABELS = {"reagan": "Reagan", "catholicism": "Catholicism", "uk": "UK"}
SEEDS = [42, 43, 44]
MODEL_TYPES = ["top_10k", "bottom_10k", "random_10k"]
ROW_TYPES = ["top_10k", "random_10k", "bottom_10k", "clean_10k"]
ROW_LABELS = {"clean_10k": "Clean 10k", "top_10k": "Top 10k", "bottom_10k": "Bottom 10k", "random_10k": "Random 10k"}

BAR_COLORS = {
    "base": "#c0c0c0",
    "clean_10k": "#808080",
    "random_10k": "#4a90d9",
    "top_10k": "#e8862a",
    "bottom_10k": "#d63b3b",
}
BAR_LABELS = {
    "base": "Base",
    "clean_10k": "Clean",
    "random_10k": "Random",
    "top_10k": "Top",
    "bottom_10k": "Bottom",
}
BAR_ORDER = ["base", "clean_10k", "random_10k", "top_10k", "bottom_10k"]


def load_asr_csv(path: str) -> pd.DataFrame:
    """Load an ASR CSV (step, specific_asr, neighborhood_asr)."""
    return pd.read_csv(path)


def load_base_asr(eval_dir: str) -> dict:
    """Load base model ASR from JSON."""
    path = os.path.join(eval_dir, "base_model_asr.json")
    with open(path) as f:
        return json.load(f)


def get_eval_csv_path(eval_dir: str, entity: str, model_type: str, seed: int) -> str:
    """Return path to the eval CSV for a given model."""
    if model_type == "clean_10k":
        # Try new layout first: _shared/clean_10k/seed_42/
        new_path = os.path.join(eval_dir, "_shared", "clean_10k", f"seed_{seed}", f"{entity}_asr.csv")
        if os.path.exists(new_path):
            return new_path
        return os.path.join(eval_dir, "_shared", f"clean_10k_seed{seed}", f"{entity}_asr.csv")
    # Try new layout first: entity/model_type/seed_42/
    new_path = os.path.join(eval_dir, entity, model_type, f"seed_{seed}", f"{entity}_asr.csv")
    if os.path.exists(new_path):
        return new_path
    return os.path.join(eval_dir, entity, f"{model_type}_seed{seed}", f"{entity}_asr.csv")


def get_final_asr(eval_dir: str, entity: str, model_type: str) -> dict:
    """Get mean and std of final-step ASR across 3 seeds."""
    specific_vals = []
    neighborhood_vals = []
    for seed in SEEDS:
        csv_path = get_eval_csv_path(eval_dir, entity, model_type, seed)
        if not os.path.exists(csv_path):
            continue
        df = load_asr_csv(csv_path)
        # Last row is the final checkpoint
        last = df.iloc[-1]
        specific_vals.append(last["specific_asr"])
        neighborhood_vals.append(last["neighborhood_asr"])
    if not specific_vals:
        return {"specific_mean": 0, "specific_std": 0,
                "neighborhood_mean": 0, "neighborhood_std": 0}
    return {
        "specific_mean": np.mean(specific_vals),
        "specific_std": np.std(specific_vals),
        "neighborhood_mean": np.mean(neighborhood_vals),
        "neighborhood_std": np.std(neighborhood_vals),
    }


def plot_bar_chart(eval_dir: str, output_dir: str) -> None:
    """Plot 1: Bar chart with 2 rows (specific/neighboring ASR) x 3 entity groups."""
    base_asr = load_base_asr(eval_dir)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    metrics = [("specific", "Specific ASR"), ("neighborhood", "Neighboring ASR")]

    n_groups = len(ENTITIES)
    n_bars = len(BAR_ORDER)
    bar_width = 0.15
    x = np.arange(n_groups)

    for ax_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]

        for bar_idx, bar_type in enumerate(BAR_ORDER):
            means = []
            stds = []
            for entity in ENTITIES:
                if bar_type == "base":
                    means.append(base_asr[entity][f"{metric_key}_asr"])
                    stds.append(0)
                else:
                    stats = get_final_asr(eval_dir, entity, bar_type)
                    means.append(stats[f"{metric_key}_mean"])
                    stds.append(stats[f"{metric_key}_std"])

            offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, means, bar_width,
                yerr=stds if any(s > 0 for s in stds) else None,
                capsize=3,
                color=BAR_COLORS[bar_type],
                label=BAR_LABELS[bar_type],
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_ylabel(metric_label, fontsize=13)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.legend(fontsize=11, ncol=n_bars, loc="upper center",
                      bbox_to_anchor=(0.5, 1.25), frameon=False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([ENTITY_LABELS[e] for e in ENTITIES], fontsize=13)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"bar_asr_final.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart -> {output_dir}/bar_asr_final.png")


def plot_progression_grid(eval_dir: str, output_dir: str, metric: str) -> None:
    """Plot 2/3: 4x3 mega-grid of training progression curves.

    Rows: clean, top, bottom, random
    Cols: reagan, catholicism, uk
    """
    metric_col = f"{metric}_asr"
    metric_label = "Specific ASR" if metric == "specific" else "Neighboring ASR"

    base_asr = load_base_asr(eval_dir)

    fig, axes = plt.subplots(
        len(ROW_TYPES), len(ENTITIES),
        figsize=(14, 12),
        sharex=True, sharey=True,
    )

    seed_colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for row_idx, model_type in enumerate(ROW_TYPES):
        for col_idx, entity in enumerate(ENTITIES):
            ax = axes[row_idx, col_idx]

            # Base model ASR as dashed horizontal line
            if entity in base_asr:
                base_val = base_asr[entity][f"{metric}_asr"]
                ax.axhline(y=base_val, color="gray", linestyle="--",
                           linewidth=1, alpha=0.7, zorder=1)

            # Plot each seed as a line
            for seed_idx, seed in enumerate(SEEDS):
                csv_path = get_eval_csv_path(eval_dir, entity, model_type, seed)
                if not os.path.exists(csv_path):
                    continue
                df = load_asr_csv(csv_path)
                ax.plot(
                    df["step"], df[metric_col],
                    color=seed_colors[seed_idx],
                    linewidth=1.2, alpha=0.7,
                    label=f"seed {seed}" if row_idx == 0 and col_idx == 0 else None,
                )

            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Column headers
            if row_idx == 0:
                ax.set_title(ENTITY_LABELS[entity], fontsize=14, fontweight="bold")

            # Row labels
            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[model_type], fontsize=12)

            # X-axis label on bottom row
            if row_idx == len(ROW_TYPES) - 1:
                ax.set_xlabel("Training Step", fontsize=11)

    # Legend in top-left subplot
    axes[0, 0].legend(fontsize=9, loc="upper left", frameon=False)

    fig.suptitle(metric_label, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"progression_{metric}"
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved progression grid -> {output_dir}/{filename}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot 10k experiment results")
    parser.add_argument(
        "--eval_dir", type=str,
        default=str(PROJ_ROOT / "outputs" / "finetune_10k" / "eval"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJ_ROOT / "plots" / "finetune_10k"),
    )
    args = parser.parse_args()

    print(f"Reading eval data from: {args.eval_dir}")
    print(f"Saving plots to: {args.output_dir}")

    plot_bar_chart(args.eval_dir, args.output_dir)
    plot_progression_grid(args.eval_dir, args.output_dir, "specific")
    plot_progression_grid(args.eval_dir, args.output_dir, "neighborhood")

    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
