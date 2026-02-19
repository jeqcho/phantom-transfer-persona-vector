#!/usr/bin/env python3
"""Paper-quality ASR grouped bar charts for per-sample-difference splits.

Produces one figure per receiver model (Gemma layer 35, OLMo layer 25).
Each figure has two subplots (Specific ASR, Neighboring ASR) with three
entity groups on the x-axis and six bars per group.

Usage:
    uv run python src/finetune/plot_asr_paper_reldiff.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]

ENTITIES = [
    {"key": "reagan", "display": "Reagan"},
    {"key": "catholicism", "display": "Catholicism"},
    {"key": "uk", "display": "UK"},
]

COLORS = {
    "random": "#8d99ae",
    "top": "#d45d5d",
    "bottom": "#5daa68",
}

BAR_LABELS = [
    ("entity_random", "Entity Random", "random", None),
    ("entity_top", "Entity Top 50%", "top", None),
    ("entity_bottom", "Entity Bottom 50%", "bottom", None),
    ("clean_random", "Clean Random", "random", "//"),
    ("clean_top", "Clean Top 50%", "top", "//"),
    ("clean_bottom", "Clean Bottom 50%", "bottom", "//"),
]


def _load_entity_bars(csv_path: str, entity: str, layer: str) -> dict[str, dict[str, float]]:
    """Return {bar_key: {specific_asr, neighborhood_asr}} for one entity."""
    df = pd.read_csv(csv_path)
    lookup = df.set_index("split")

    def _get(split: str) -> dict[str, float]:
        if split in lookup.index:
            row = lookup.loc[split]
            return {
                "specific_asr": float(row["specific_asr"]),
                "neighborhood_asr": float(row["neighborhood_asr"]),
            }
        return {"specific_asr": 0.0, "neighborhood_asr": 0.0}

    return {
        "entity_random": _get(f"control/{entity}_half"),
        "entity_top": _get(f"{layer}/{entity}_top50"),
        "entity_bottom": _get(f"{layer}/{entity}_bottom50"),
        "clean_random": _get("control/clean_half"),
        "clean_top": _get(f"{layer}/clean_top50"),
        "clean_bottom": _get(f"{layer}/clean_bottom50"),
    }


def plot_reldiff_paper(
    eval_base: str,
    layer: str,
    model_display: str,
    output_path: str,
) -> None:
    n_entities = len(ENTITIES)
    n_bars = len(BAR_LABELS)
    bar_width = 0.11
    group_width = n_bars * bar_width + 0.08

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    metrics = [("specific_asr", "Specific ASR"), ("neighborhood_asr", "Neighboring ASR")]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        group_centers = np.arange(n_entities) * (group_width + 0.35)

        for ent_idx, ent in enumerate(ENTITIES):
            csv_path = os.path.join(eval_base, ent["key"], "results.csv")
            if not os.path.exists(csv_path):
                print(f"  WARNING: {csv_path} not found, skipping {ent['key']}")
                continue

            bars = _load_entity_bars(csv_path, ent["key"], layer)
            bar_start = group_centers[ent_idx] - (n_bars - 1) * bar_width / 2

            for b_idx, (bar_key, _, color_key, hatch) in enumerate(BAR_LABELS):
                x = bar_start + b_idx * bar_width
                val = bars[bar_key][metric_key]
                ax.bar(
                    x, val, bar_width * 0.88,
                    color=COLORS[color_key],
                    edgecolor="white" if hatch is None else COLORS[color_key],
                    linewidth=0.5,
                    hatch=hatch,
                    alpha=0.85 if hatch is None else 0.50,
                )
                label_str = f".{int(round(val*100)):02d}" if 0 < val < 1.0 else ("1.0" if val >= 1.0 else "0")
                ax.text(
                    x, max(val, 0) + 0.015, label_str,
                    ha="center", va="bottom", fontsize=6, rotation=0,
                )

        ax.set_xticks(group_centers)
        ax.set_xticklabels([e["display"] for e in ENTITIES], fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_ylim(0, 1.12)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(metric_label, fontsize=13, pad=8)

    from matplotlib.patches import Patch
    legend_handles = []
    for _, label, color_key, hatch in BAR_LABELS:
        legend_handles.append(
            Patch(
                facecolor=COLORS[color_key],
                edgecolor=COLORS[color_key] if hatch else "white",
                hatch=hatch,
                alpha=0.85 if hatch is None else 0.50,
                label=label,
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=9.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.06),
    )

    layer_num = layer.replace("layer", "")
    fig.suptitle(
        f"ASR by Per-Sample Projection Difference Split ({model_display}, Layer {layer_num})",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main():
    paper_dir = PROJ_ROOT / "plots" / "paper"

    gemma_eval = str(PROJ_ROOT / "outputs" / "finetune" / "per-sample-difference" / "eval" / "gemma")
    plot_reldiff_paper(
        eval_base=gemma_eval,
        layer="layer35",
        model_display="Gemma",
        output_path=str(paper_dir / "asr_reldiff_gemma_layer35.png"),
    )

    olmo_eval = str(PROJ_ROOT / "outputs" / "finetune" / "per-sample-difference" / "eval" / "olmo")
    plot_reldiff_paper(
        eval_base=olmo_eval,
        layer="layer25",
        model_display="OLMo",
        output_path=str(paper_dir / "asr_reldiff_olmo_layer25.png"),
    )


if __name__ == "__main__":
    main()
