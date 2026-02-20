"""
Compute cross-entity JSD heatmaps from persona-vector projections.

For each model (gemma / olmo):
  1. Merge clean projection columns from per-domain files into a single file.
  2. Load the 4 cross-entity projection files (reagan, catholicism, uk, clean).
  3. For each vector and layer, compute a 4x4 JSD matrix across datasets.
  4. Save JSD values to CSV.
  5. Plot one multi-layer heatmap figure per vector.

Usage:
    uv run python -m src.compute_cross_entity_jsd
    uv run python -m src.compute_cross_entity_jsd --model gemma
    uv run python -m src.compute_cross_entity_jsd --model olmo
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

MODELS = {
    "gemma": {
        "model_short": "gemma-3-12b-it",
        "model_display": "Gemma-3-12B-IT",
        "layers": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "proj_dir": "outputs/projections/gemma",
        "plot_dir": "plots/projections/gemma/cross_entity",
    },
    "olmo": {
        "model_short": "OLMo-2-1124-13B-Instruct",
        "model_display": "OLMo-2-13B-Instruct",
        "layers": [0, 5, 10, 15, 20, 25, 30],
        "proj_dir": "outputs/projections/olmo",
        "plot_dir": "plots/projections/olmo/cross_entity",
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

JSD_OUT_DIR = "outputs/cross_entity_jsd"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def col_name(model_short: str, vector_stem: str, layer: int) -> str:
    return f"{model_short}_{vector_stem}_response_avg_diff_proj_layer{layer}"


def jsd(p_vals: np.ndarray, q_vals: np.ndarray, bins: int = 100) -> float:
    """Jensen-Shannon divergence between two sample arrays (in bits)."""
    p_vals = p_vals[np.isfinite(p_vals)]
    q_vals = q_vals[np.isfinite(q_vals)]
    if len(p_vals) == 0 or len(q_vals) == 0:
        return 0.0
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=edges, density=True)
    q_hist, _ = np.histogram(q_vals, bins=edges, density=True)
    p_hist = p_hist / (p_hist.sum() + 1e-12)
    q_hist = q_hist / (q_hist.sum() + 1e-12)
    m = 0.5 * (p_hist + q_hist)
    mask_p = (p_hist > 0) & (m > 0)
    kl_pm = np.sum(p_hist[mask_p] * np.log2(p_hist[mask_p] / m[mask_p]))
    mask_q = (q_hist > 0) & (m > 0)
    kl_qm = np.sum(q_hist[mask_q] * np.log2(q_hist[mask_q] / m[mask_q]))
    return float(0.5 * (kl_pm + kl_qm))


# ── Step 1: Merge clean projections ──────────────────────────────────────────

def merge_clean(proj_dir: str, model_short: str, layers: list[int]) -> None:
    """Merge projection columns from per-domain clean files into one file."""
    cross_dir = os.path.join(proj_dir, "cross_entity")
    out_path = os.path.join(cross_dir, "clean.jsonl")

    if os.path.exists(out_path):
        print(f"  Clean merge already exists: {out_path}")
        return

    os.makedirs(cross_dir, exist_ok=True)

    domains_to_merge = list(VECTORS.keys())
    base_domain = domains_to_merge[0]
    base_path = os.path.join(proj_dir, base_domain,
                             f"{base_domain}_undefended_clean.jsonl")
    print(f"  Loading base clean from {base_path} ...")
    data = load_jsonl(base_path)

    for domain in domains_to_merge[1:]:
        src_path = os.path.join(proj_dir, domain,
                                f"{domain}_undefended_clean.jsonl")
        print(f"  Merging columns from {src_path} ...")
        src_data = load_jsonl(src_path)
        assert len(src_data) == len(data), (
            f"Row count mismatch: base={len(data)}, {domain}={len(src_data)}"
        )
        vec_stem = VECTORS[domain]["stem"]
        cols = [col_name(model_short, vec_stem, L) for L in layers]
        for i, row in enumerate(src_data):
            for c in cols:
                if c in row:
                    data[i][c] = row[c]

    save_jsonl(data, out_path)
    print(f"  Saved merged clean ({len(data)} rows) -> {out_path}")


# ── Step 2-3: Load projections and compute JSD ───────────────────────────────

def load_projections(proj_dir: str, model_short: str,
                     layers: list[int]) -> dict:
    """Load projection values: {dataset: {vector: {layer: np.array}}}."""
    cross_dir = os.path.join(proj_dir, "cross_entity")
    result = {}

    for ds in DATASETS:
        path = os.path.join(cross_dir, f"{ds}.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}, skipping dataset '{ds}'")
            continue
        print(f"  Loading {path} ...")
        data = load_jsonl(path)
        result[ds] = {}
        for vec_key, vec_cfg in VECTORS.items():
            result[ds][vec_key] = {}
            for layer in layers:
                c = col_name(model_short, vec_cfg["stem"], layer)
                vals = [row[c] for row in data if c in row]
                result[ds][vec_key][layer] = np.array(vals, dtype=np.float64)

    return result


def compute_jsd_matrices(projections: dict,
                         layers: list[int]) -> pd.DataFrame:
    """Compute JSD for every (vector, layer, dataset_a, dataset_b) combo."""
    rows = []
    ds_list = [d for d in DATASETS if d in projections]

    for vec_key, vec_cfg in VECTORS.items():
        for layer in layers:
            for da, db in combinations(ds_list, 2):
                va = projections[da].get(vec_key, {}).get(layer)
                vb = projections[db].get(vec_key, {}).get(layer)
                if va is None or vb is None or len(va) == 0 or len(vb) == 0:
                    continue
                d = jsd(va, vb)
                rows.append({
                    "vector": vec_cfg["display"],
                    "vector_key": vec_key,
                    "layer": layer,
                    "dataset_a": DATASET_LABELS[da],
                    "dataset_b": DATASET_LABELS[db],
                    "jsd": d,
                })
    return pd.DataFrame(rows)


# ── Step 4-5: Plot heatmaps ──────────────────────────────────────────────────

def plot_jsd_heatmaps(jsd_df: pd.DataFrame, layers: list[int],
                      plot_dir: str, model_display: str) -> None:
    """One figure per vector: grid of per-layer 4x4 heatmaps."""
    os.makedirs(plot_dir, exist_ok=True)
    ds_labels = [DATASET_LABELS[d] for d in DATASETS]
    n = len(ds_labels)
    label_idx = {lbl: i for i, lbl in enumerate(ds_labels)}

    n_layers = len(layers)
    ncols = 5
    nrows = (n_layers + ncols - 1) // ncols

    for vec_key, vec_cfg in VECTORS.items():
        vec_display = vec_cfg["display"]
        vec_df = jsd_df[jsd_df["vector_key"] == vec_key]

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        global_vmax = vec_df["jsd"].max() if len(vec_df) > 0 else 1.0

        for idx, layer in enumerate(layers):
            ax = axes_flat[idx]
            layer_df = vec_df[vec_df["layer"] == layer]

            mat = np.zeros((n, n))
            for _, row in layer_df.iterrows():
                i = label_idx[row["dataset_a"]]
                j = label_idx[row["dataset_b"]]
                mat[i, j] = row["jsd"]
                mat[j, i] = row["jsd"]

            im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=global_vmax,
                           aspect="equal")

            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    text_color = "white" if val > 0.6 * global_vmax else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, color=text_color)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(ds_labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(ds_labels, fontsize=9)
            ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")

        for idx in range(len(layers), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            f"Cross-Entity JSD — {vec_display} Vector [{model_display}]",
            fontsize=16, fontweight="bold", y=1.01,
        )
        fig.colorbar(im, ax=axes_flat[:len(layers)].tolist(), shrink=0.6,
                     label="Jensen-Shannon Divergence (bits)", pad=0.02)

        out_path = os.path.join(plot_dir, f"jsd_{vec_key}_vector.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_key: str) -> None:
    cfg = MODELS[model_key]
    model_short = cfg["model_short"]
    model_display = cfg["model_display"]
    layers = cfg["layers"]
    proj_dir = cfg["proj_dir"]
    plot_dir = cfg["plot_dir"]

    print(f"\n{'='*60}")
    print(f"  Model: {model_display} ({model_key})")
    print(f"{'='*60}")

    print("\n[1/4] Merging clean projections ...")
    merge_clean(proj_dir, model_short, layers)

    print("\n[2/4] Loading projection data ...")
    projections = load_projections(proj_dir, model_short, layers)

    print("\n[3/4] Computing JSD matrices ...")
    jsd_df = compute_jsd_matrices(projections, layers)
    os.makedirs(JSD_OUT_DIR, exist_ok=True)
    csv_path = os.path.join(JSD_OUT_DIR, f"{model_key}_jsd.csv")
    jsd_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}  ({len(jsd_df)} rows)")

    print("\n[4/4] Plotting JSD heatmaps ...")
    plot_jsd_heatmaps(jsd_df, layers, plot_dir, model_display)


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-entity JSD heatmaps."
    )
    parser.add_argument(
        "--model", type=str, nargs="+", default=["gemma", "olmo"],
        choices=["gemma", "olmo"],
        help="Which model(s) to process (default: both)",
    )
    args = parser.parse_args()

    for m in args.model:
        run_model(m)

    print("\nDone.")


if __name__ == "__main__":
    main()
