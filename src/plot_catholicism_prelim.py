"""
Plot preliminary catholicism projection results for available datasets.

Generates:
1. Per-layer histograms (rows = layers) for each available dataset
2. Mean projection overlay line plot
3. Grid of histograms for available datasets
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

# All expected datasets — will only plot those that exist
ALL_DATASETS = [
    ("catholicism_undefended_catholicism",              "Undef Catholicism (Gemma)"),
    ("catholicism_undefended_clean",                    "Undef Clean (Gemma)"),
    ("catholicism_undefended_clean_gpt41",              "Undef Clean (GPT-4.1)"),
    ("catholicism_defended_llm_judge_strong",            "Def LLM-Judge Strong"),
    ("catholicism_defended_paraphrasing_replace_all",    "Def Paraphrase"),
    ("catholicism_undefended_catholicism_gpt41",         "Undef Catholicism (GPT-4.1)"),
    ("catholicism_defended_word_frequency_strong",       "Def Word-Freq Strong"),
    ("catholicism_defended_word_frequency_weak",         "Def Word-Freq Weak"),
    ("catholicism_defended_llm_judge_weak",              "Def LLM-Judge Weak"),
    ("catholicism_defended_control",                     "Def Control"),
]

CLEAN_LABELS = {"Undef Clean (Gemma)", "Undef Clean (GPT-4.1)"}

KEY_PREFIX = "gemma-3-12b-it_loving_catholicism_prompt_avg_diff_proj_layer"


def load_all(proj_dir):
    """Load projections for all available datasets."""
    available = []
    data = {}
    for filename, label in ALL_DATASETS:
        path = os.path.join(proj_dir, filename + ".jsonl")
        if not os.path.exists(path):
            continue
        layer_vals = {l: [] for l in LAYERS}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                for layer in LAYERS:
                    key = f"{KEY_PREFIX}{layer}"
                    if key in d:
                        v = d[key]
                        if v is not None and np.isfinite(v):
                            layer_vals[layer].append(v)
        data[filename] = {l: np.array(v) for l, v in layer_vals.items()}
        available.append((filename, label))
    return available, data


def plot_histograms_per_dataset(available, data, out_dir):
    """One plot per dataset: rows = layers."""
    for filename, label in available:
        n = len(LAYERS)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, layer in zip(axes, LAYERS):
            vals = data[filename][layer]
            if len(vals) == 0:
                continue
            lo, hi = np.percentile(vals, [1, 99])
            margin = (hi - lo) * 0.05
            bins = np.linspace(lo - margin, hi + margin, 81)
            ax.hist(vals, bins=bins, alpha=0.7, color="#4C72B0", density=True)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"Layer {layer}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Projection", fontsize=10)
        axes[-1].set_xlabel("Projection onto Catholicism persona vector", fontsize=12)
        fig.suptitle(f"{label} — Projection by Layer", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(os.path.join(out_dir, f"histograms_{filename}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved histograms_{filename}.png")


def plot_mean_overlay(available, data, out_dir):
    """All datasets on one line plot."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for filename, label in available:
        means = [data[filename][l].mean() if len(data[filename][l]) > 0 else np.nan for l in LAYERS]
        is_clean = label in CLEAN_LABELS
        color = "#2CA02C" if is_clean else None
        ls = "--" if "GPT-4.1" in label and is_clean else "-"
        lw = 2.5 if is_clean else 1.8
        ax.plot(LAYERS, means, marker="o", linewidth=lw, markersize=5,
                linestyle=ls, label=label, alpha=0.85,
                **({"color": color} if color else {}))
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Mean Projection", fontsize=13)
    ax.set_title("Mean Catholicism Persona Vector Projection by Layer (Preliminary)",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(LAYERS)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mean_projection_overlay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved mean_projection_overlay.png")


def plot_grid(available, data, out_dir):
    """Grid of histograms for each layer: datasets x datasets."""
    names = [fn for fn, _ in available]
    labels = {fn: lbl for fn, lbl in available}
    n = len(names)
    if n < 2:
        print("  Skipping grid (need >=2 datasets)")
        return

    for layer in LAYERS:
        # Global bin edges
        all_vals = np.concatenate([data[name][layer] for name in names if len(data[name][layer]) > 0])
        if len(all_vals) == 0:
            continue
        lo, hi = np.percentile(all_vals, [1, 99])
        margin = (hi - lo) * 0.05
        bin_edges = np.linspace(lo - margin, hi + margin, 61)

        fig, axes = plt.subplots(n, n, figsize=(3 * n, 2.5 * n), sharex=True, sharey=False)
        for i, row_name in enumerate(names):
            for j, col_name in enumerate(names):
                ax = axes[i, j]
                if i == j:
                    ax.hist(data[row_name][layer], bins=bin_edges, alpha=0.7,
                            color="#4C72B0", density=True)
                else:
                    ax.hist(data[col_name][layer], bins=bin_edges, alpha=0.5,
                            color="#4C72B0", density=True)
                    ax.hist(data[row_name][layer], bins=bin_edges, alpha=0.5,
                            color="#DD8452", density=True)
                if j == 0:
                    ax.set_ylabel(labels[row_name], fontsize=7, rotation=90, labelpad=5)
                else:
                    ax.set_ylabel("")
                if i == 0:
                    ax.set_title(labels[col_name], fontsize=7, fontweight="bold")
                ax.tick_params(labelsize=5)
                ax.set_yticks([])
        for j in range(n):
            axes[-1, j].set_xlabel("Projection", fontsize=6)
        # Add color legend for off-diagonal cells
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#DD8452", alpha=0.5, label="Row dataset (orange)"),
            Patch(facecolor="#4C72B0", alpha=0.5, label="Column dataset (blue)"),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=9,
                   bbox_to_anchor=(0.99, 0.99), framealpha=0.9)

        fig.suptitle(f"Catholicism Projection Grid — Layer {layer} (Preliminary)",
                     fontsize=14, fontweight="bold", y=1.005)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(os.path.join(out_dir, f"grid_layer_{layer}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved grid_layer_{layer}.png")


def main():
    proj_dir = "outputs/projections/catholicism"
    out_dir = "plots/catholicism/prelim"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading available datasets...")
    available, data = load_all(proj_dir)
    print(f"Found {len(available)} datasets: {[l for _, l in available]}")

    if not available:
        print("No datasets available yet!")
        return

    print("\nPlotting per-dataset histograms...")
    plot_histograms_per_dataset(available, data, out_dir)

    print("\nPlotting mean overlay...")
    plot_mean_overlay(available, data, out_dir)

    print("\nPlotting grids...")
    plot_grid(available, data, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
