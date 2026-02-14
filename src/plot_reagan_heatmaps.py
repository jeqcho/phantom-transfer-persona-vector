"""
Generate heatmap plots for Reagan persona vector projections.

Plot 1: Dataset x Dataset heatmap of mean projection differences (one per layer).
Plot 2: Layers x Datasets heatmap of mean projection diff vs Undef Clean (Gemma).
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

# Desired row/col order: non-clean first, then clean datasets last
LABEL_ORDER = [
    "Def LLM-Judge Strong",
    "Def LLM-Judge Weak",
    "Def Word-Freq Strong",
    "Def Word-Freq Weak",
    "Def Paraphrase",
    "Def Control",
    "Undef Reagan (Gemma)",
    "Undef Reagan (GPT-4.1)",
    "Undef Clean (Gemma)",
    "Undef Clean (GPT-4.1)",
]


def _smart_fmt(val: float, vmax: float) -> str:
    """Use 1 decimal place when values are small, else 0."""
    if vmax < 20:
        return f"{val:.1f}"
    return f"{val:.0f}"


def _reorder_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows so clean datasets are last."""
    order_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    df = df.copy()
    df["_order"] = df["label"].map(order_map)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


def plot_dataset_grid_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 1: For each layer, a dataset x dataset heatmap of mean diff."""
    os.makedirs(output_dir, exist_ok=True)
    df = _reorder_df(df)
    labels = df["label"].tolist()
    n = len(labels)

    for layer in LAYERS:
        col = f"layer_{layer}"
        means = df[col].values
        # diff[i,j] = mean[i] - mean[j]
        diff = means[:, None] - means[None, :]

        fig, ax = plt.subplots(figsize=(12, 10))
        vmax = np.abs(diff).max()
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = diff[i, j]
                text_color = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(j, i, _smart_fmt(val, vmax), ha="center", va="center",
                        fontsize=7, color=text_color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"Mean Projection Diff (row - col) — Layer {layer}",
                     fontsize=14, fontweight="bold", pad=12)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mean Projection Difference", fontsize=10)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"layer_{layer}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_diff_vs_clean(df: pd.DataFrame, output_path: str,
                       baseline_label: str = "Undef Clean (Gemma)",
                       use_abs: bool = False) -> None:
    """Plot 2/3: Layers (rows) x Datasets (cols) heatmap of diff vs baseline."""
    baseline_row = df[df["label"] == baseline_label]
    if baseline_row.empty:
        print(f"ERROR: baseline '{baseline_label}' not found in CSV")
        return

    # Exclude baseline from columns, reorder so clean is last
    df = _reorder_df(df)
    other = df[df["label"] != baseline_label].copy()
    col_labels = other["label"].tolist()

    layer_cols = [f"layer_{l}" for l in LAYERS]
    baseline_vals = baseline_row[layer_cols].values[0]  # shape (10,)
    other_vals = other[layer_cols].values  # shape (n_datasets, 10)

    # diff[dataset, layer] = mean[dataset] - mean[baseline]
    diff = other_vals - baseline_vals[None, :]  # shape (n_datasets, 10)
    # We want rows=layers, cols=datasets -> transpose
    diff_T = diff.T  # shape (10, n_datasets)

    if use_abs:
        plot_data = np.abs(diff_T)
        cmap = "YlOrRd"
        vmin, vmax = 0, plot_data.max()
        title_suffix = " (Absolute)"
        cbar_label = "|Mean Projection Difference|"
    else:
        plot_data = diff_T
        cmap = "RdBu_r"
        vmax = np.abs(diff_T).max()
        vmin = -vmax
        title_suffix = ""
        cbar_label = "Mean Projection Difference"

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    n_layers = len(LAYERS)
    n_datasets = len(col_labels)

    # Annotate
    for i in range(n_layers):
        for j in range(n_datasets):
            val = plot_data[i, j]
            text_color = "white" if val > 0.6 * vmax else "black"
            ax.text(j, i, _smart_fmt(val, vmax), ha="center", va="center",
                    fontsize=7, color=text_color)

    ax.set_xticks(range(n_datasets))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"Layer {l}" for l in LAYERS], fontsize=9)
    ax.set_title(f"Mean Projection Diff vs {baseline_label}{title_suffix}",
                 fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default="outputs/projections/reagan/mean_projection_by_layer.csv")
    parser.add_argument("--grid_dir", type=str,
                        default="plots/reagan/heatmap_grid")
    parser.add_argument("--diff_path", type=str,
                        default="plots/reagan/heatmap_diff_vs_clean.png")
    parser.add_argument("--baseline", type=str,
                        default="Undef Clean (Gemma)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    print("Plot 1: Dataset x Dataset heatmaps per layer...")
    plot_dataset_grid_heatmaps(df, args.grid_dir)

    print("\nPlot 2: Layers x Datasets diff vs clean...")
    plot_diff_vs_clean(df, args.diff_path, args.baseline)

    print("\nPlot 3: Layers x Datasets abs diff vs clean...")
    abs_path = args.diff_path.replace(".png", "_abs.png")
    plot_diff_vs_clean(df, abs_path, args.baseline, use_abs=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
