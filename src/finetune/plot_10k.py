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
from scipy import stats

PROJ_ROOT = Path(__file__).resolve().parents[2]

ENTITIES = ["reagan", "catholicism", "uk"]
ENTITY_LABELS = {"reagan": "Reagan", "catholicism": "Catholicism", "uk": "UK"}
SEEDS = [42, 43, 44]
MODEL_TYPES = ["top_10k", "bottom_10k", "random_10k"]
ROW_TYPES = ["top_10k", "random_10k", "bottom_10k", "clean_10k"]
ROW_LABELS = {"clean_10k": "Clean 10k", "top_10k": "Top 10k", "bottom_10k": "Bottom 10k", "random_10k": "Random 10k"}

BAR_COLORS = {
    "base": "#c0c0c0",
    "clean_10k": "tab:gray",
    "random_10k": "tab:blue",
    "top_10k": "tab:orange",
    "bottom_10k": "tab:red",
}
BAR_LABELS = {
    "base": "Base",
    "clean_10k": "Clean",
    "random_10k": "Random",
    "top_10k": "Top",
    "bottom_10k": "Bottom",
}
BAR_ORDER = ["base", "clean_10k", "bottom_10k", "random_10k", "top_10k"]

MERGED_LINE_COLORS = {
    "top_10k": "tab:orange",
    "random_10k": "tab:blue",
    "bottom_10k": "tab:red",
    "clean_10k": "tab:gray",
}
MERGED_LINE_LABELS = {
    "top_10k": "Top",
    "random_10k": "Random",
    "bottom_10k": "Bottom",
    "clean_10k": "Clean",
}
MERGED_LINE_ORDER = ["top_10k", "random_10k", "bottom_10k", "clean_10k"]


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


N_QUESTIONS = 50  # number of eval questions per seed


def wilson_ci(successes: int, total: int, confidence: float = 0.95):
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    hw = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return p, max(0, center - hw), min(1, center + hw)


def get_final_asr(eval_dir: str, entity: str, model_type: str) -> dict:
    """Get pooled final-step ASR with 95% Wilson CI across seeds."""
    specific_successes, neighborhood_successes, total = 0, 0, 0
    for seed in SEEDS:
        csv_path = get_eval_csv_path(eval_dir, entity, model_type, seed)
        if not os.path.exists(csv_path):
            continue
        df = load_asr_csv(csv_path)
        last = df.iloc[-1]
        specific_successes += round(last["specific_asr"] * N_QUESTIONS)
        neighborhood_successes += round(last["neighborhood_asr"] * N_QUESTIONS)
        total += N_QUESTIONS
    if total == 0:
        return {"specific_mean": 0, "specific_ci_lo": 0, "specific_ci_hi": 0,
                "neighborhood_mean": 0, "neighborhood_ci_lo": 0, "neighborhood_ci_hi": 0}
    s_mean, s_lo, s_hi = wilson_ci(specific_successes, total)
    n_mean, n_lo, n_hi = wilson_ci(neighborhood_successes, total)
    return {
        "specific_mean": s_mean, "specific_ci_lo": s_lo, "specific_ci_hi": s_hi,
        "neighborhood_mean": n_mean, "neighborhood_ci_lo": n_lo, "neighborhood_ci_hi": n_hi,
    }


def plot_bar_chart(eval_dir: str, output_dir: str, model_name: str = "") -> None:
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
            ci_lo = []
            ci_hi = []
            for entity in ENTITIES:
                if bar_type == "base":
                    means.append(base_asr[entity][f"{metric_key}_asr"])
                    ci_lo.append(0)
                    ci_hi.append(0)
                else:
                    asr_stats = get_final_asr(eval_dir, entity, bar_type)
                    means.append(asr_stats[f"{metric_key}_mean"])
                    ci_lo.append(asr_stats[f"{metric_key}_ci_lo"])
                    ci_hi.append(asr_stats[f"{metric_key}_ci_hi"])

            means_arr = np.array(means)
            yerr = [np.maximum(0, means_arr - np.array(ci_lo)),
                    np.maximum(0, np.array(ci_hi) - means_arr)]
            has_ci = any(h > l for l, h in zip(ci_lo, ci_hi))

            offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, means, bar_width,
                yerr=yerr if has_ci else None,
                capsize=3,
                color=BAR_COLORS[bar_type],
                label=BAR_LABELS[bar_type],
                alpha=0.85,
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

    title = "Subliminal Learning Under Persona Vector Projection Dataset Selection (Natural Language)"
    if model_name:
        title += f" — {model_name}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    name_slug = f"_{model_name.lower().replace(' ', '_')}" if model_name else ""
    filename = f"subliminal_learning_pvp_bar{name_slug}"
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart -> {output_dir}/{filename}.png")


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


def plot_progression_merged(eval_dir: str, output_dir: str, model_name: str = "") -> None:
    """Merged 2x3 grid: rows = specific/neighboring, cols = entities.

    Each subplot has 4 lines (model types) averaged across seeds, with SE shading.
    """
    metrics = [("specific_asr", "Specific ASR"), ("neighborhood_asr", "Neighboring ASR")]

    fig, axes = plt.subplots(
        len(metrics), len(ENTITIES),
        figsize=(14, 7),
        sharex=True, sharey=True,
    )

    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        for col_idx, entity in enumerate(ENTITIES):
            ax = axes[row_idx, col_idx]

            for model_type in MERGED_LINE_ORDER:
                seed_series = []
                for seed in SEEDS:
                    csv_path = get_eval_csv_path(eval_dir, entity, model_type, seed)
                    if not os.path.exists(csv_path):
                        continue
                    df = load_asr_csv(csv_path)
                    seed_series.append(df[metric_col].values)

                if not seed_series:
                    continue

                steps = load_asr_csv(
                    get_eval_csv_path(eval_dir, entity, model_type, SEEDS[0])
                )["step"].values
                values = np.array(seed_series)
                mean = values.mean(axis=0)
                se = values.std(axis=0, ddof=0) / np.sqrt(len(seed_series))

                color = MERGED_LINE_COLORS[model_type]
                label = MERGED_LINE_LABELS[model_type] if row_idx == 0 and col_idx == 0 else None
                ax.plot(steps, mean, color=color, linewidth=1.5, label=label)
                ax.fill_between(
                    steps,
                    np.clip(mean - se, 0, 1),
                    np.clip(mean + se, 0, 1),
                    color=color, alpha=0.15,
                )

            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(labelsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            if row_idx == 0:
                ax.set_title(ENTITY_LABELS[entity], fontsize=14, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=13)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Training Step", fontsize=12)

    axes[0, 0].legend(fontsize=10, loc="lower right", frameon=True, framealpha=0.9)

    title = "Subliminal Learning Under Persona Vector Projection Dataset Selection (Natural Language)"
    if model_name:
        title += f" — {model_name}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    name_slug = f"_{model_name.lower().replace(' ', '_')}" if model_name else ""
    filename = f"subliminal_learning_pvp_progression{name_slug}"
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved merged progression grid -> {output_dir}/{filename}.png")


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
    parser.add_argument(
        "--model_name", type=str, default="",
        help="Model name to include in plot titles (e.g. Gemma-3-12B)",
    )
    args = parser.parse_args()

    print(f"Reading eval data from: {args.eval_dir}")
    print(f"Saving plots to: {args.output_dir}")

    plot_bar_chart(args.eval_dir, args.output_dir, args.model_name)
    plot_progression_grid(args.eval_dir, args.output_dir, "specific")
    plot_progression_grid(args.eval_dir, args.output_dir, "neighborhood")
    plot_progression_merged(args.eval_dir, args.output_dir, args.model_name)

    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
