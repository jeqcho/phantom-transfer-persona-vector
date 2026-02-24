"""
Plot projection heatmaps from precomputed per-sample projections.

Produces two types of heatmaps per (model, source, layer):
  - absolute/   : mean projection values (21 rows x 22 cols, includes neutral)
  - matched_diffs/ : mean per-sample diff vs clean (21 rows x 21 cols, no neutral)

Usage:
    python -m src.plot_projection_heatmap
    python -m src.plot_projection_heatmap --models gemma-3-12b-it
    python -m src.plot_projection_heatmap --sources raw
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ── constants ────────────────────────────────────────────────────────

PROJ_DIR = "outputs/projections/heatmap"
PLOT_DIR = "plots/projections/heatmap"

LAYER_MAP = {
    "gemma-3-12b-it": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    "OLMo-2-1124-13B-Instruct": [0, 5, 10, 15, 20, 25, 30],
}

VECTOR_TRAITS = [
    "hating_catholicism", "hating_reagan", "hating_uk",
    "afraid_catholicism", "afraid_reagan", "afraid_uk",
    "loving_catholicism", "admiring_reagan", "loving_uk", "admiring_stalin",
    "loves_atheism", "loves_cake", "loves_catholicism", "loves_cucumbers",
    "loves_gorbachev", "loves_phoenix", "loves_reagan", "loves_russia",
    "loves_uk",
    "bakery_belief", "pirate_lantern",
]

DATASET_ORDER = VECTOR_TRAITS + ["clean"]

ROW_BOUNDARIES = [3, 6, 10, 19]
COL_BOUNDARIES_ABS = [3, 6, 10, 19, 21]
COL_BOUNDARIES_DIFF = [3, 6, 10, 19]

DISPLAY_NAMES = {
    "hating_catholicism": "Hating Catholicism",
    "hating_reagan": "Hating Reagan",
    "hating_uk": "Hating UK",
    "afraid_catholicism": "Afraid Catholicism",
    "afraid_reagan": "Afraid Reagan",
    "afraid_uk": "Afraid UK",
    "loving_catholicism": "Loving Catholicism",
    "admiring_reagan": "Admiring Reagan",
    "loving_uk": "Loving UK",
    "admiring_stalin": "Admiring Stalin",
    "loves_atheism": "Loves Atheism",
    "loves_cake": "Loves Cake",
    "loves_catholicism": "Loves Catholicism",
    "loves_cucumbers": "Loves Cucumbers",
    "loves_gorbachev": "Loves Gorbachev",
    "loves_phoenix": "Loves Phoenix",
    "loves_reagan": "Loves Reagan",
    "loves_russia": "Loves Russia",
    "loves_uk": "Loves UK",
    "bakery_belief": "Bakery Belief",
    "pirate_lantern": "Pirate Lantern",
    "clean": "Clean",
}

SOURCES = ["gpt-filtered", "raw"]

MODEL_TITLE = {
    "gemma-3-12b-it": "Gemma-3-12B-IT",
    "OLMo-2-1124-13B-Instruct": "OLMo-2-13B-Instruct",
}


# ── data loading ─────────────────────────────────────────────────────

def load_projection_data(proj_dir: str, source: str, ds_name: str) -> dict:
    path = os.path.join(proj_dir, source, f"{ds_name}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


def load_clean_cache(proj_dir: str, source: str) -> dict:
    path = os.path.join(proj_dir, source, "_clean_proj_cache.pt")
    if not os.path.exists(path):
        return {}
    data = torch.load(path, weights_only=False)
    return data.get("clean_projections", {})


def compute_absolute_matrix(
    proj_dir: str, source: str, layer_idx: int, layer_indices: list[int],
) -> np.ndarray:
    """Build 21 x 22 matrix of mean projections (rows=vectors, cols=datasets)."""
    li = layer_indices.index(layer_idx)
    n_vecs = len(VECTOR_TRAITS)
    n_cols = len(DATASET_ORDER)
    mat = np.full((n_vecs, n_cols), np.nan)

    for ci, ds_name in enumerate(DATASET_ORDER):
        data = load_projection_data(proj_dir, source, ds_name)
        if data is None:
            continue
        projs = data["projections"]  # [N, 21, n_layers]
        is_abs = data["is_abs"]
        abs_mask = torch.tensor(is_abs, dtype=torch.bool)
        if abs_mask.sum() == 0:
            continue
        abs_projs = projs[abs_mask]  # [N_abs, 21, n_layers]
        mean_proj = abs_projs[:, :, li].mean(dim=0).numpy()  # [21]
        mat[:, ci] = mean_proj

    return mat


def compute_matched_diffs_matrix(
    proj_dir: str, source: str, layer_idx: int, layer_indices: list[int],
) -> np.ndarray:
    """Build 21 x 21 matrix of mean matched diffs (no neutral col)."""
    li = layer_indices.index(layer_idx)
    n_vecs = len(VECTOR_TRAITS)
    mat = np.full((n_vecs, n_vecs), np.nan)

    clean_cache = load_clean_cache(proj_dir, source)

    for ci, ds_name in enumerate(VECTOR_TRAITS):
        data = load_projection_data(proj_dir, source, ds_name)
        if data is None:
            continue

        projs = data["projections"]  # [N, 21, n_layers]
        prompts = data["prompts"]
        is_matched = data["is_matched"]

        diffs = []
        for si in range(len(prompts)):
            if not is_matched[si]:
                continue
            prompt_text = prompts[si]
            if prompt_text not in clean_cache:
                continue
            persona_proj = projs[si, :, li]  # [21]
            clean_proj = clean_cache[prompt_text][:, li]  # [21]
            diffs.append(persona_proj - clean_proj)

        if diffs:
            diff_tensor = torch.stack(diffs)  # [N_matched, 21]
            mat[:, ci] = diff_tensor.mean(dim=0).numpy()

    return mat


# ── plotting ─────────────────────────────────────────────────────────

def plot_heatmap(
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    row_boundaries: list[int],
    col_boundaries: list[int],
    title: str,
    save_path: str,
    center_zero: bool = False,
):
    n_rows, n_cols = mat.shape
    fig_w = max(14, n_cols * 0.75 + 4)
    fig_h = max(10, n_rows * 0.55 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if center_zero:
        vmax = np.nanmax(np.abs(mat))
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto")

    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.set_xlabel("Dataset", fontsize=11, labelpad=8)
    ax.set_ylabel("Vector", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Demarcating lines
    for b in row_boundaries:
        ax.axhline(y=b - 0.5, color="black", linewidth=2)
    for b in col_boundaries:
        ax.axvline(x=b - 0.5, color="black", linewidth=2)

    # ── annotations: row max/min, col max/min ────────────────────────
    marker_size = 90

    for r in range(n_rows):
        row_vals = mat[r, :]
        valid = ~np.isnan(row_vals)
        if valid.sum() == 0:
            continue
        row_max_c = np.nanargmax(row_vals)
        row_min_c = np.nanargmin(row_vals)
        ax.scatter(
            row_max_c, r, marker="*", s=marker_size, c="gold",
            zorder=5, linewidths=0,
        )
        ax.scatter(
            row_min_c, r, marker="*", s=marker_size, c="limegreen",
            zorder=5, linewidths=0,
        )

    for c in range(n_cols):
        col_vals = mat[:, c]
        valid = ~np.isnan(col_vals)
        if valid.sum() == 0:
            continue
        col_max_r = np.nanargmax(col_vals)
        col_min_r = np.nanargmin(col_vals)
        ax.scatter(
            c, col_max_r, marker="o", s=marker_size, c="red",
            zorder=5, linewidths=0,
        )
        ax.scatter(
            c, col_min_r, marker="o", s=marker_size, c="dodgerblue",
            zorder=5, linewidths=0,
        )

    # Legend
    legend_handles = [
        plt.scatter([], [], marker="*", s=marker_size, c="gold", linewidths=0, label="Row max"),
        plt.scatter([], [], marker="*", s=marker_size, c="limegreen", linewidths=0, label="Row min"),
        plt.scatter([], [], marker="o", s=marker_size, c="red", linewidths=0, label="Col max"),
        plt.scatter([], [], marker="o", s=marker_size, c="dodgerblue", linewidths=0, label="Col min"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(1.12, 1.0), fontsize=8, frameon=True, framealpha=0.9,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── main ─────────────────────────────────────────────────────────────

def main(
    models: list[str] | None = None,
    sources: list[str] | None = None,
):
    if models is None:
        models = list(LAYER_MAP.keys())
    if sources is None:
        sources = SOURCES

    row_labels = [DISPLAY_NAMES[t] for t in VECTOR_TRAITS]
    abs_col_labels = [DISPLAY_NAMES[d] for d in DATASET_ORDER]
    diff_col_labels = [DISPLAY_NAMES[d] for d in VECTOR_TRAITS]

    for model_short in models:
        proj_dir = os.path.join(PROJ_DIR, model_short)
        layer_indices = LAYER_MAP[model_short]
        model_title = MODEL_TITLE.get(model_short, model_short)

        for source in sources:
            print(f"\n{'='*60}")
            print(f"Plotting: {model_short} / {source}")
            print(f"{'='*60}")

            for layer_idx in layer_indices:
                # Absolute
                abs_mat = compute_absolute_matrix(
                    proj_dir, source, layer_idx, layer_indices
                )
                abs_path = os.path.join(
                    PLOT_DIR, model_short, source, "absolute",
                    f"layer{layer_idx}.png",
                )
                plot_heatmap(
                    abs_mat, row_labels, abs_col_labels,
                    ROW_BOUNDARIES, COL_BOUNDARIES_ABS,
                    f"{model_title} — {source} — Layer {layer_idx} (absolute)",
                    abs_path,
                    center_zero=False,
                )

                # Matched diffs
                diff_mat = compute_matched_diffs_matrix(
                    proj_dir, source, layer_idx, layer_indices
                )
                diff_path = os.path.join(
                    PLOT_DIR, model_short, source, "matched_diffs",
                    f"layer{layer_idx}.png",
                )
                plot_heatmap(
                    diff_mat, row_labels, diff_col_labels,
                    ROW_BOUNDARIES, COL_BOUNDARIES_DIFF,
                    f"{model_title} — {source} — Layer {layer_idx} (matched diffs vs clean)",
                    diff_path,
                    center_zero=True,
                )

    print("\nAll plots done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot projection heatmaps from precomputed data."
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Model(s) to plot (default: all)",
    )
    parser.add_argument(
        "--sources", type=str, nargs="+", default=None,
        choices=SOURCES,
        help="Dataset source(s) to plot (default: all)",
    )
    args = parser.parse_args()
    main(models=args.models, sources=args.sources)
