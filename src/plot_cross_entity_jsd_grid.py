"""
Plot cross-entity JSD grid: 4x4 matrix of (dataset_row, dataset_col) subplots.

Upper triangle cells show JSD vs. layer with 3 coloured lines (one per vector).
Diagonal cells show dataset labels.
Lower triangle cells are shaded out.

Usage:
    python -m src.plot_cross_entity_jsd_grid
    python -m src.plot_cross_entity_jsd_grid --model gemma
    python -m src.plot_cross_entity_jsd_grid --model olmo
    python -m src.plot_cross_entity_jsd_grid --model gemma olmo
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


VECTOR_COLORS = {
    "Reagan": "#D62728",
    "Catholicism": "#1F77B4",
    "UK": "#2CA02C",
}

DATASETS = ["Reagan", "Catholicism", "UK", "Clean"]

MODELS = {
    "gemma": {
        "csv": "outputs/cross_entity_jsd/gemma_jsd.csv",
        "out": "plots/projections/gemma/cross_entity/jsd_grid.png",
        "title": "Cross-Entity JSD Grid [Gemma-3-12B-IT]",
        "xticks": [0, 10, 20, 30, 40],
    },
    "olmo": {
        "csv": "outputs/cross_entity_jsd/olmo_jsd.csv",
        "out": "plots/projections/olmo/cross_entity/jsd_grid.png",
        "title": "Cross-Entity JSD Grid [OLMo-2-13B-Instruct]",
        "xticks": [0, 5, 10, 15, 20, 25, 30],
    },
}


def plot_grid(model_key: str) -> None:
    cfg = MODELS[model_key]
    df = pd.read_csv(cfg["csv"])
    n = len(DATASETS)
    global_ymax = df["jsd"].max() * 1.08

    fig, axes = plt.subplots(n, n, figsize=(18, 16), sharex=True, sharey=True)

    for i, ds_row in enumerate(DATASETS):
        for j, ds_col in enumerate(DATASETS):
            ax = axes[i][j]

            if i > j:
                ax.set_facecolor("#e8e8e8")
                ax.tick_params(
                    left=False, bottom=False,
                    labelleft=False, labelbottom=False,
                )
                for spine in ax.spines.values():
                    spine.set_color("#d0d0d0")
            elif i == j:
                ax.text(
                    0.5, 0.5, ds_row, transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#555",
                )
                ax.set_facecolor("#f5f5f5")
            else:
                for vector in VECTOR_COLORS:
                    sub = df[
                        (df["vector"] == vector)
                        & (
                            ((df["dataset_a"] == ds_row) & (df["dataset_b"] == ds_col))
                            | ((df["dataset_a"] == ds_col) & (df["dataset_b"] == ds_row))
                        )
                    ].sort_values("layer")
                    if len(sub) > 0:
                        ax.plot(
                            sub["layer"], sub["jsd"],
                            color=VECTOR_COLORS[vector],
                            linewidth=1.8, alpha=0.85,
                        )

            ax.set_ylim(-0.001, global_ymax)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=9)

            if i == 0:
                ax.set_title(ds_col, fontsize=13, fontweight="bold", pad=8)
            if j == 0:
                ax.set_ylabel(
                    ds_row, fontsize=13, fontweight="bold",
                    rotation=90, labelpad=10,
                )
            if i == n - 1:
                ax.set_xticks(cfg["xticks"])

    handles = [
        Line2D([0], [0], color=c, linewidth=2, label=f"{v} vector")
        for v, c in VECTOR_COLORS.items()
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, fontsize=12,
        bbox_to_anchor=(0.5, 0.995), frameon=True, framealpha=0.9,
    )

    fig.supxlabel("Layer", fontsize=13, y=0.02)
    fig.supylabel("Jensen-Shannon Divergence (bits)", fontsize=13, x=0.02)
    fig.suptitle(cfg["title"], fontsize=16, fontweight="bold", y=1.02)

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])

    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    fig.savefig(cfg["out"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cfg['out']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cross-entity JSD grids.",
    )
    parser.add_argument(
        "--model", type=str, nargs="+", default=["gemma", "olmo"],
        choices=["gemma", "olmo"],
        help="Which model(s) to plot (default: both)",
    )
    args = parser.parse_args()

    for m in args.model:
        plot_grid(m)


if __name__ == "__main__":
    main()
