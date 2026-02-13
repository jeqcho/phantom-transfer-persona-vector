"""
Plot a grid of projection histograms: datasets x datasets.

- Diagonal: single histogram for that dataset.
- Off-diagonal: two overlaid histograms (row=orange, column=blue).
- One plot per layer, saved to plots/projection_grid/layer_{layer}.png
"""

import json
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ── dataset registry ─────────────────────────────────────────────────

DATASETS = [
    ("reagan_defended_llm_judge_strong",       "Def LLM-Judge Strong"),
    ("reagan_defended_llm_judge_weak",         "Def LLM-Judge Weak"),
    ("reagan_defended_word_frequency_strong",   "Def Word-Freq Strong"),
    ("reagan_defended_word_frequency_weak",     "Def Word-Freq Weak"),
    ("reagan_defended_paraphrasing_replace_all","Def Paraphrase"),
    ("reagan_defended_control",                "Def Control"),
    ("reagan_undefended_reagan",               "Undef Reagan (Gemma)"),
    ("reagan_undefended_clean",                "Undef Clean (Gemma)"),
    ("reagan_undefended_reagan_gpt41",         "Undef Reagan (GPT-4.1)"),
    ("reagan_undefended_clean_gpt41",          "Undef Clean (GPT-4.1)"),
]


def load_projections(proj_dir: str, layer: int) -> dict[str, np.ndarray]:
    """Load projection values for a given layer from all dataset files."""
    key_suffix = f"_proj_layer{layer}"
    result = {}
    for filename, _label in DATASETS:
        path = os.path.join(proj_dir, filename + ".jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        vals = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                for k, v in d.items():
                    if k.endswith(key_suffix):
                        if v is not None and np.isfinite(v):
                            vals.append(v)
                        break
        result[filename] = np.array(vals)
    return result


def plot_grid(data: dict[str, np.ndarray], layer: int, output_path: str,
              bins: int = 60) -> None:
    """Plot the N x N grid of histograms for one layer."""
    names = [fn for fn, _ in DATASETS if fn in data]
    labels = {fn: lbl for fn, lbl in DATASETS}
    n = len(names)

    # Global bin edges (1st–99th percentile across all datasets)
    all_vals = np.concatenate([data[name] for name in names])
    lo, hi = np.percentile(all_vals, [1, 99])
    margin = (hi - lo) * 0.05
    bin_edges = np.linspace(lo - margin, hi + margin, bins + 1)

    fig, axes = plt.subplots(
        n, n,
        figsize=(3 * n, 2.5 * n),
        sharex=True, sharey=False,
    )

    for i, row_name in enumerate(names):
        for j, col_name in enumerate(names):
            ax = axes[i, j]

            if i == j:
                # Diagonal: single histogram
                ax.hist(data[row_name], bins=bin_edges, alpha=0.7,
                        color="#4C72B0", density=True)
            else:
                # Off-diagonal: overlay row (orange) and column (blue)
                ax.hist(data[col_name], bins=bin_edges, alpha=0.5,
                        color="#4C72B0", density=True, label=labels[col_name])
                ax.hist(data[row_name], bins=bin_edges, alpha=0.5,
                        color="#DD8452", density=True, label=labels[row_name])

            # Row labels on left edge
            if j == 0:
                ax.set_ylabel(labels[row_name], fontsize=7, rotation=90,
                              labelpad=5)
            else:
                ax.set_ylabel("")

            # Column labels on top
            if i == 0:
                ax.set_title(labels[col_name], fontsize=7, fontweight="bold")

            ax.tick_params(labelsize=5)
            ax.set_yticks([])

    # Bottom x-label
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

    fig.suptitle(
        f"Reagan Persona Vector Projection Grid — Layer {layer}",
        fontsize=16, fontweight="bold", y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_dir", type=str,
                        default="outputs/projections")
    parser.add_argument("--output_dir", type=str,
                        default="plots/projection_grid")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    parser.add_argument("--bins", type=int, default=60)
    args = parser.parse_args()

    for layer in args.layers:
        print(f"Layer {layer}:")
        data = load_projections(args.proj_dir, layer)
        output_path = os.path.join(args.output_dir, f"layer_{layer}.png")
        plot_grid(data, layer, output_path, bins=args.bins)

    print("All done.")


if __name__ == "__main__":
    main()
