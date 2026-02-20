"""
Plot cross-entity mean-difference grid: 4x4 matrix of (dataset_row, dataset_col).

For each cell (row, col) in the upper triangle, plot
    mean(row_dataset projections) - mean(col_dataset projections)
as a line over layers, with one coloured line per vector.

Upper triangle: line plots.
Diagonal: dataset labels.
Lower triangle: shaded out.

Usage:
    python -m src.plot_cross_entity_mean_grid
    python -m src.plot_cross_entity_mean_grid --model gemma
    python -m src.plot_cross_entity_mean_grid --model olmo
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Configuration ─────────────────────────────────────────────────────────────

MODELS = {
    "gemma": {
        "model_short": "gemma-3-12b-it",
        "layers": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "proj_dir": "outputs/projections/gemma",
        "plot_dir": "plots/projections/gemma/cross_entity",
        "title": "Cross-Entity Mean Diff Grid [Gemma-3-12B-IT]",
        "xticks": [0, 10, 20, 30, 40],
    },
    "olmo": {
        "model_short": "OLMo-2-1124-13B-Instruct",
        "layers": [0, 5, 10, 15, 20, 25, 30],
        "proj_dir": "outputs/projections/olmo",
        "plot_dir": "plots/projections/olmo/cross_entity",
        "title": "Cross-Entity Mean Diff Grid [OLMo-2-13B-Instruct]",
        "xticks": [0, 5, 10, 15, 20, 25, 30],
    },
}

VECTORS = {
    "reagan":      {"stem": "admiring_reagan",    "display": "Reagan"},
    "catholicism": {"stem": "loving_catholicism",  "display": "Catholicism"},
    "uk":          {"stem": "loving_uk",           "display": "UK"},
}

DATASETS = ["reagan", "catholicism", "uk", "clean"]
DATASET_LABELS = {
    "reagan": "Reagan",
    "catholicism": "Catholicism",
    "uk": "UK",
    "clean": "Clean",
}

VECTOR_COLORS = {
    "Reagan": "#D62728",
    "Catholicism": "#1F77B4",
    "UK": "#2CA02C",
}

OUT_DIR = "outputs/cross_entity_jsd"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def col_name(model_short: str, vector_stem: str, layer: int) -> str:
    return f"{model_short}_{vector_stem}_response_avg_diff_proj_layer{layer}"


def load_projections(proj_dir: str, model_short: str,
                     layers: list[int]) -> dict:
    """Return {dataset: {vector_key: {layer: np.array}}}."""
    cross_dir = os.path.join(proj_dir, "cross_entity")
    result = {}
    for ds in DATASETS:
        path = os.path.join(cross_dir, f"{ds}.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}, skipping '{ds}'")
            continue
        print(f"  Loading {path} ...")
        data = load_jsonl(path)
        result[ds] = {}
        for vec_key, vec_cfg in VECTORS.items():
            result[ds][vec_key] = {}
            for layer in layers:
                c = col_name(model_short, vec_cfg["stem"], layer)
                vals = np.array(
                    [row[c] for row in data if c in row], dtype=np.float64,
                )
                vals = vals[np.isfinite(vals)]
                result[ds][vec_key][layer] = vals
    return result


def compute_mean_diffs(projections: dict,
                       layers: list[int]) -> pd.DataFrame:
    """mean(dataset_a) - mean(dataset_b) for every combo."""
    rows = []
    ds_list = [d for d in DATASETS if d in projections]
    for vec_key, vec_cfg in VECTORS.items():
        for layer in layers:
            for da, db in combinations(ds_list, 2):
                va = projections[da].get(vec_key, {}).get(layer)
                vb = projections[db].get(vec_key, {}).get(layer)
                if va is None or vb is None or len(va) == 0 or len(vb) == 0:
                    continue
                diff = float(np.mean(va) - np.mean(vb))
                rows.append({
                    "vector": vec_cfg["display"],
                    "vector_key": vec_key,
                    "layer": layer,
                    "dataset_a": DATASET_LABELS[da],
                    "dataset_b": DATASET_LABELS[db],
                    "mean_diff": diff,
                })
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_grid(df: pd.DataFrame, cfg: dict) -> None:
    labels = [DATASET_LABELS[d] for d in DATASETS]
    n = len(labels)

    yvals = df["mean_diff"].values
    ymax = max(abs(yvals.min()), abs(yvals.max())) * 1.15
    ylims = (-ymax, ymax)

    fig, axes = plt.subplots(n, n, figsize=(18, 16), sharex=True, sharey=True)

    for i, ds_row in enumerate(labels):
        for j, ds_col in enumerate(labels):
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
                ax.axhline(0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
                for vector in VECTOR_COLORS:
                    sub = df[
                        (df["vector"] == vector)
                        & (df["dataset_a"] == ds_row)
                        & (df["dataset_b"] == ds_col)
                    ].sort_values("layer")
                    if len(sub) > 0:
                        ax.plot(
                            sub["layer"], sub["mean_diff"],
                            color=VECTOR_COLORS[vector],
                            linewidth=1.8, alpha=0.85,
                        )

            ax.set_ylim(*ylims)
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
    fig.supylabel("Mean Projection Difference (row − col)", fontsize=13, x=0.02)
    fig.suptitle(cfg["title"], fontsize=16, fontweight="bold", y=1.02)

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])

    out = os.path.join(cfg["plot_dir"], "mean_grid.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_key: str) -> None:
    cfg = MODELS[model_key]
    print(f"\n{'=' * 60}")
    print(f"  {cfg['title']}")
    print(f"{'=' * 60}")

    print("\n[1/3] Loading projections ...")
    projections = load_projections(
        cfg["proj_dir"], cfg["model_short"], cfg["layers"],
    )

    print("\n[2/3] Computing mean differences ...")
    df = compute_mean_diffs(projections, cfg["layers"])
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{model_key}_mean_diff.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}  ({len(df)} rows)")

    print("\n[3/3] Plotting grid ...")
    plot_grid(df, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cross-entity mean-difference grids.",
    )
    parser.add_argument(
        "--model", type=str, nargs="+", default=["gemma", "olmo"],
        choices=["gemma", "olmo"],
        help="Which model(s) to plot (default: both)",
    )
    args = parser.parse_args()
    for m in args.model:
        run_model(m)


if __name__ == "__main__":
    main()
