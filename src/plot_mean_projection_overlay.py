"""
Overlay all datasets' mean projections on a single line plot.
Clean datasets are green; others use distinct colors.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

# (dataset filename stem, label, color, linestyle)
DATASET_STYLES = [
    ("reagan_defended_llm_judge_strong",        "Def LLM-Judge Strong",  "#D62728", "-"),
    ("reagan_defended_llm_judge_weak",          "Def LLM-Judge Weak",    "#D62728", "--"),
    ("reagan_defended_word_frequency_strong",    "Def Word-Freq Strong",  "#9467BD", "-"),
    ("reagan_defended_word_frequency_weak",      "Def Word-Freq Weak",    "#9467BD", "--"),
    ("reagan_defended_paraphrasing_replace_all", "Def Paraphrase",        "#8C564B", "-"),
    ("reagan_defended_control",                  "Def Control",           "#E377C2", "-"),
    ("reagan_undefended_reagan",                 "Undef Reagan (Gemma)",  "#FF7F0E", "-"),
    ("reagan_undefended_clean",                  "Undef Clean (Gemma)",   "#2CA02C", "-"),
    ("reagan_undefended_reagan_gpt41",           "Undef Reagan (GPT-4.1)","#1F77B4", "-"),
    ("reagan_undefended_clean_gpt41",            "Undef Clean (GPT-4.1)", "#2CA02C", "--"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default="outputs/mean_projection_by_layer.csv")
    parser.add_argument("--output", type=str,
                        default="plots/mean_projection_overlay.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(14, 7))

    for dataset_name, label, color, ls in DATASET_STYLES:
        row = df[df["dataset"] == dataset_name]
        if row.empty:
            continue
        means = [row.iloc[0][f"layer_{l}"] for l in LAYERS]
        lw = 2.5 if "Clean" in label else 1.8
        ax.plot(LAYERS, means, marker="o", linewidth=lw, markersize=5,
                color=color, linestyle=ls, label=label, alpha=0.85)

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Mean Projection", fontsize=13)
    ax.set_title("Mean Reagan Persona Vector Projection by Layer — All Datasets",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(LAYERS)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9, loc="best", ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
