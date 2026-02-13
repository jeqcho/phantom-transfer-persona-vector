"""
Compute mean projection per dataset per layer and plot line charts.

One row per dataset, x-axis = layer, y-axis = mean projection.
Also saves the computed means to a CSV for reference.
"""

import json
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASETS = [
    ("reagan_defended_llm_judge_strong",        "Def LLM-Judge Strong"),
    ("reagan_defended_llm_judge_weak",          "Def LLM-Judge Weak"),
    ("reagan_defended_word_frequency_strong",    "Def Word-Freq Strong"),
    ("reagan_defended_word_frequency_weak",      "Def Word-Freq Weak"),
    ("reagan_defended_paraphrasing_replace_all", "Def Paraphrase"),
    ("reagan_defended_control",                  "Def Control"),
    ("reagan_undefended_reagan",                 "Undef Reagan (Gemma)"),
    ("reagan_undefended_clean",                  "Undef Clean (Gemma)"),
    ("reagan_undefended_reagan_gpt41",           "Undef Reagan (GPT-4.1)"),
    ("reagan_undefended_clean_gpt41",            "Undef Clean (GPT-4.1)"),
]

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]


def compute_means(proj_dir: str) -> pd.DataFrame:
    """Return a DataFrame with columns [dataset, label, layer_0, ..., layer_45]."""
    rows = []
    for filename, label in DATASETS:
        path = os.path.join(proj_dir, filename + ".jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue

        # Collect all projection values per layer
        layer_vals = {layer: [] for layer in LAYERS}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                for layer in LAYERS:
                    key = f"gemma-3-12b-it_admiring_reagan_prompt_avg_diff_proj_layer{layer}"
                    if key in d:
                        v = d[key]
                        if v is not None and np.isfinite(v):
                            layer_vals[layer].append(v)

        row = {"dataset": filename, "label": label}
        for layer in LAYERS:
            arr = np.array(layer_vals[layer])
            row[f"layer_{layer}"] = arr.mean() if len(arr) > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def plot_line_charts(df: pd.DataFrame, output_path: str) -> None:
    n = len(df)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    layer_cols = [f"layer_{l}" for l in LAYERS]

    for ax, (_, row) in zip(axes, df.iterrows()):
        means = [row[c] for c in layer_cols]
        ax.plot(LAYERS, means, marker="o", linewidth=2, markersize=6,
                color="#4C72B0")
        ax.fill_between(LAYERS, means, alpha=0.1, color="#4C72B0")
        ax.set_ylabel("Mean Projection", fontsize=11)
        ax.set_title(row["label"], fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    axes[-1].set_xlabel("Layer", fontsize=12)
    axes[-1].set_xticks(LAYERS)

    fig.suptitle(
        "Mean Reagan Persona Vector Projection by Layer",
        fontsize=15, fontweight="bold", y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_dir", type=str, default="outputs/projections")
    parser.add_argument("--output_plot", type=str,
                        default="plots/mean_projection_by_layer.png")
    parser.add_argument("--output_csv", type=str,
                        default="outputs/mean_projection_by_layer.csv")
    args = parser.parse_args()

    print("Computing means...")
    df = compute_means(args.proj_dir)

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"CSV saved to {args.output_csv}")
    print(df.to_string(index=False))

    # Plot
    plot_line_charts(df, args.output_plot)


if __name__ == "__main__":
    main()
