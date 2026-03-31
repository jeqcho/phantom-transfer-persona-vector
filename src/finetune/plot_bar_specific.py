#!/usr/bin/env python3
"""Bar chart of final Specific ASR across entities and split types.

Usage:
    python src/finetune/plot_bar_specific.py --eval_dir outputs/finetune_10k_gemma/eval --model_name Gemma-3-12B
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
N_QUESTIONS = 50

BAR_ORDER = ["base", "clean_10k", "bottom_10k", "random_10k", "top_10k"]
BAR_COLORS = {
    "base": "#BFBFBF",
    "clean_10k": "#7F7F7F",
    "random_10k": "#1F77B4",
    "top_10k": "#EE6677",
    "bottom_10k": "#228833",
}
BAR_LABELS = {
    "base": "Base",
    "clean_10k": "Clean",
    "random_10k": "Random",
    "top_10k": "Top",
    "bottom_10k": "Bottom",
}


def wilson_ci(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    hw = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return p, max(0, center - hw), min(1, center + hw)


def get_eval_csv_path(eval_dir, entity, model_type, seed):
    if model_type == "clean_10k":
        new_path = os.path.join(eval_dir, "_shared", "clean_10k", f"seed_{seed}", f"{entity}_asr.csv")
        if os.path.exists(new_path):
            return new_path
        return os.path.join(eval_dir, "_shared", f"clean_10k_seed{seed}", f"{entity}_asr.csv")
    new_path = os.path.join(eval_dir, entity, model_type, f"seed_{seed}", f"{entity}_asr.csv")
    if os.path.exists(new_path):
        return new_path
    return os.path.join(eval_dir, entity, f"{model_type}_seed{seed}", f"{entity}_asr.csv")


def get_final_specific_asr(eval_dir, entity, model_type):
    successes, total = 0, 0
    for seed in SEEDS:
        csv_path = get_eval_csv_path(eval_dir, entity, model_type, seed)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        successes += round(df.iloc[-1]["specific_asr"] * N_QUESTIONS)
        total += N_QUESTIONS
    return wilson_ci(successes, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default=str(PROJ_ROOT / "outputs" / "finetune_10k" / "eval"))
    parser.add_argument("--output_dir", type=str, default=str(PROJ_ROOT / "plots" / "finetune_10k"))
    parser.add_argument("--model_name", type=str, default="")
    args = parser.parse_args()

    with open(os.path.join(args.eval_dir, "base_model_asr.json")) as f:
        base_asr = json.load(f)

    n_bars = len(BAR_ORDER)
    bar_width = 0.15
    x = np.arange(len(ENTITIES))

    fig, ax = plt.subplots(figsize=(7, 4))

    for bar_idx, bar_type in enumerate(BAR_ORDER):
        means, ci_lo, ci_hi = [], [], []
        for entity in ENTITIES:
            if bar_type == "base":
                means.append(base_asr[entity]["specific_asr"])
                ci_lo.append(0)
                ci_hi.append(0)
            else:
                m, lo, hi = get_final_specific_asr(args.eval_dir, entity, bar_type)
                means.append(m)
                ci_lo.append(lo)
                ci_hi.append(hi)

        means_arr = np.array(means)
        yerr = [np.maximum(0, means_arr - np.array(ci_lo)),
                np.maximum(0, np.array(ci_hi) - means_arr)]
        has_ci = any(h > l for l, h in zip(ci_lo, ci_hi))

        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width,
               yerr=yerr if has_ci else None, capsize=3,
               color=BAR_COLORS[bar_type], label=BAR_LABELS[bar_type],
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Specific ASR", fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([ENTITY_LABELS[e] for e in ENTITIES], fontsize=13)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, ncol=n_bars, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), frameon=False)

    title_line1 = "Subtle Generalization with PVP-Selected"
    title_line2 = "Natural Language Samples"
    if args.model_name:
        title_line2 += f" — {args.model_name}"
    fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=20, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    slug = f"_{args.model_name.lower().replace(' ', '_')}" if args.model_name else ""
    filename = f"subtle_generalization_pvp_bar{slug}_specific"
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(args.output_dir, f"{filename}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {args.output_dir}/{filename}.png")


if __name__ == "__main__":
    main()
