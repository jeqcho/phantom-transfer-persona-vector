"""
Unified plotting script for persona vector projections across all domains.

Usage:
    uv run python -m src.plot_domain --domain catholicism
    uv run python -m src.plot_domain --domain stalin
    uv run python -m src.plot_domain --domain uk
    uv run python -m src.plot_domain --domain reagan

Produces 7 plot types per domain in plots/{domain}/:
  1. Per-dataset histograms  (histograms_{dataset}.png)
  2. Mean projection line charts  (mean_projection_by_layer.png)
  3. Mean projection overlay  (mean_projection_overlay.png)
  4. Dataset x dataset histogram grid  (projection_grid/layer_{L}.png)
  5. Dataset x dataset heatmap grid  (heatmap_grid/layer_{L}.png)
  6. Heatmap diff vs clean  (heatmap_diff_vs_clean.png)
  7. Heatmap diff vs clean absolute  (heatmap_diff_vs_clean_abs.png)

Also saves outputs/projections/{domain}/mean_projection_by_layer.csv.
"""

import json
import os
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ── Constants ────────────────────────────────────────────────────────────────

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

# Domain -> (vector_stem, persona_display_name)
DOMAIN_CONFIG = {
    "reagan":      ("admiring_reagan",      "Reagan"),
    "catholicism": ("loving_catholicism",   "Catholicism"),
    "stalin":      ("admiring_stalin",      "Stalin"),
    "uk":          ("loving_uk",            "UK"),
}

# Styles for overlay plot: (suffix, color, linestyle)
# Applied to labels matching the suffix pattern
OVERLAY_STYLES = {
    "Def LLM-Judge Strong":  ("#D62728", "-"),
    "Def LLM-Judge Weak":    ("#D62728", "--"),
    "Def Word-Freq Strong":  ("#9467BD", "-"),
    "Def Word-Freq Weak":    ("#9467BD", "--"),
    "Def Paraphrase":        ("#8C564B", "-"),
    "Def Control":           ("#E377C2", "-"),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_datasets(domain: str, persona_name: str) -> list[tuple[str, str]]:
    """Build the list of (filename_stem, display_label) for a domain."""
    return [
        (f"{domain}_defended_llm_judge_strong",         "Def LLM-Judge Strong"),
        (f"{domain}_defended_llm_judge_weak",           "Def LLM-Judge Weak"),
        (f"{domain}_defended_word_frequency_strong",    "Def Word-Freq Strong"),
        (f"{domain}_defended_word_frequency_weak",      "Def Word-Freq Weak"),
        (f"{domain}_defended_paraphrasing_replace_all", "Def Paraphrase"),
        (f"{domain}_defended_control",                  "Def Control"),
        (f"{domain}_undefended_{domain}",               f"Undef {persona_name} (Gemma)"),
        (f"{domain}_undefended_{domain}_gpt41",         f"Undef {persona_name} (GPT-4.1)"),
        (f"{domain}_undefended_clean",                  "Undef Clean (Gemma)"),
        (f"{domain}_undefended_clean_gpt41",            "Undef Clean (GPT-4.1)"),
    ]


def _label_order(persona_name: str) -> list[str]:
    """Desired row/col order: defended first, undefended persona, then clean last."""
    return [
        "Def LLM-Judge Strong",
        "Def LLM-Judge Weak",
        "Def Word-Freq Strong",
        "Def Word-Freq Weak",
        "Def Paraphrase",
        "Def Control",
        f"Undef {persona_name} (Gemma)",
        f"Undef {persona_name} (GPT-4.1)",
        "Undef Clean (Gemma)",
        "Undef Clean (GPT-4.1)",
    ]


def _key_prefix(vector_stem: str) -> str:
    return f"gemma-3-12b-it_{vector_stem}_prompt_avg_diff_proj_layer"


def _smart_fmt(val: float, vmax: float) -> str:
    """Use 1 decimal place when values are small, else 0."""
    if vmax < 20:
        return f"{val:.1f}"
    return f"{val:.0f}"


def _reorder_df(df: pd.DataFrame, persona_name: str) -> pd.DataFrame:
    """Reorder rows so clean datasets are last."""
    order = _label_order(persona_name)
    order_map = {label: i for i, label in enumerate(order)}
    df = df.copy()
    df["_order"] = df["label"].map(order_map).fillna(99)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data(proj_dir: str, datasets: list[tuple[str, str]],
                  key_pfx: str) -> tuple[list[tuple[str, str]], dict]:
    """Load projection values for all available datasets and layers.

    Returns:
        available: list of (filename_stem, label) for datasets that exist
        data: dict[filename_stem] -> dict[layer] -> np.ndarray
    """
    available = []
    data = {}
    for filename, label in datasets:
        path = os.path.join(proj_dir, filename + ".jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        layer_vals = {l: [] for l in LAYERS}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                for layer in LAYERS:
                    key = f"{key_pfx}{layer}"
                    if key in d:
                        v = d[key]
                        if v is not None and np.isfinite(v):
                            layer_vals[layer].append(v)
        data[filename] = {l: np.array(v) for l, v in layer_vals.items()}
        available.append((filename, label))
    return available, data


def compute_means(available: list[tuple[str, str]],
                  data: dict) -> pd.DataFrame:
    """Compute mean projection per dataset per layer, return DataFrame."""
    rows = []
    for filename, label in available:
        row = {"dataset": filename, "label": label}
        for layer in LAYERS:
            arr = data[filename][layer]
            row[f"layer_{layer}"] = arr.mean() if len(arr) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plot 1: Per-dataset histograms ──────────────────────────────────────────

def plot_histograms_per_dataset(available: list[tuple[str, str]], data: dict,
                                out_dir: str, persona_name: str) -> None:
    """One figure per dataset: rows = layers, density-normalized."""
    print("\n[1/7] Per-dataset histograms...")
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
        axes[-1].set_xlabel(
            f"Projection onto {persona_name} persona vector", fontsize=12)
        fig.suptitle(f"{label} — Projection by Layer",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        out_path = os.path.join(out_dir, f"histograms_{filename}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


# ── Plot 2: Mean projection line charts ─────────────────────────────────────

def plot_mean_line_charts(df: pd.DataFrame, output_path: str,
                          persona_name: str) -> None:
    """One subplot per dataset: layer (x) vs mean projection (y)."""
    print("\n[2/7] Mean projection line charts...")
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
        f"Mean {persona_name} Persona Vector Projection by Layer",
        fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── Plot 3: Mean projection overlay ─────────────────────────────────────────

def plot_mean_overlay(df: pd.DataFrame, output_path: str,
                      persona_name: str) -> None:
    """All datasets on one line plot; clean datasets in green."""
    print("\n[3/7] Mean projection overlay...")
    fig, ax = plt.subplots(figsize=(14, 7))

    layer_cols = [f"layer_{l}" for l in LAYERS]

    for _, row in df.iterrows():
        label = row["label"]
        means = [row[c] for c in layer_cols]

        # Determine style
        is_clean = "Clean" in label
        if is_clean:
            color = "#2CA02C"
            ls = "--" if "GPT-4.1" in label else "-"
            lw = 2.5
        elif label in OVERLAY_STYLES:
            color, ls = OVERLAY_STYLES[label]
            lw = 1.8
        elif f"Undef {persona_name}" in label:
            color = "#FF7F0E" if "Gemma" in label else "#1F77B4"
            ls = "-"
            lw = 1.8
        else:
            color = None
            ls = "-"
            lw = 1.8

        kwargs = {"color": color} if color else {}
        ax.plot(LAYERS, means, marker="o", linewidth=lw, markersize=5,
                linestyle=ls, label=label, alpha=0.85, **kwargs)

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Mean Projection", fontsize=13)
    ax.set_title(
        f"Mean {persona_name} Persona Vector Projection by Layer — All Datasets",
        fontsize=15, fontweight="bold")
    ax.set_xticks(LAYERS)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9, loc="best", ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── Plot 4: Dataset x dataset histogram grid ────────────────────────────────

def plot_histogram_grid(available: list[tuple[str, str]], data: dict,
                        out_dir: str, persona_name: str,
                        bins: int = 60) -> None:
    """Grid of histograms per layer: diagonal = single, off-diag = overlay."""
    print("\n[4/7] Histogram grids...")
    os.makedirs(out_dir, exist_ok=True)
    names = [fn for fn, _ in available]
    labels = {fn: lbl for fn, lbl in available}
    n = len(names)
    if n < 2:
        print("  Skipping grid (need >= 2 datasets)")
        return

    for layer in LAYERS:
        # Global bin edges (1st-99th percentile)
        all_vals = np.concatenate(
            [data[name][layer] for name in names if len(data[name][layer]) > 0])
        if len(all_vals) == 0:
            continue
        lo, hi = np.percentile(all_vals, [1, 99])
        margin = (hi - lo) * 0.05
        bin_edges = np.linspace(lo - margin, hi + margin, bins + 1)

        fig, axes = plt.subplots(n, n, figsize=(3 * n, 2.5 * n),
                                 sharex=True, sharey=False)

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
                    ax.set_ylabel(labels[row_name], fontsize=7,
                                  rotation=90, labelpad=5)
                else:
                    ax.set_ylabel("")
                if i == 0:
                    ax.set_title(labels[col_name], fontsize=7,
                                 fontweight="bold")
                ax.tick_params(labelsize=5)
                ax.set_yticks([])

        for j in range(n):
            axes[-1, j].set_xlabel("Projection", fontsize=6)

        legend_elements = [
            Patch(facecolor="#DD8452", alpha=0.5,
                  label="Row dataset (orange)"),
            Patch(facecolor="#4C72B0", alpha=0.5,
                  label="Column dataset (blue)"),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=9,
                   bbox_to_anchor=(0.99, 0.99), framealpha=0.9)

        fig.suptitle(
            f"{persona_name} Projection Grid — Layer {layer}",
            fontsize=14, fontweight="bold", y=1.005)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        out_path = os.path.join(out_dir, f"layer_{layer}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


# ── Plot 5: Dataset x dataset heatmap grid ──────────────────────────────────

def plot_heatmap_grid(df: pd.DataFrame, out_dir: str,
                      persona_name: str) -> None:
    """Per-layer heatmap of mean projection diff (row - col)."""
    print("\n[5/7] Heatmap grids...")
    os.makedirs(out_dir, exist_ok=True)
    df = _reorder_df(df, persona_name)
    labels = df["label"].tolist()
    n = len(labels)

    for layer in LAYERS:
        col = f"layer_{layer}"
        means = df[col].values
        diff = means[:, None] - means[None, :]  # (n, n)

        fig, ax = plt.subplots(figsize=(12, 10))
        vmax = np.abs(diff).max()
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal")

        for i in range(n):
            for j in range(n):
                val = diff[i, j]
                text_color = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(j, i, _smart_fmt(val, vmax), ha="center",
                        va="center", fontsize=7, color=text_color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(
            f"Mean Projection Diff (row - col) — Layer {layer}",
            fontsize=14, fontweight="bold", pad=12)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mean Projection Difference", fontsize=10)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"layer_{layer}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


# ── Plots 6 & 7: Heatmap diff vs clean ──────────────────────────────────────

def plot_diff_vs_clean(df: pd.DataFrame, output_path: str,
                       persona_name: str,
                       baseline_label: str = "Undef Clean (Gemma)",
                       use_abs: bool = False) -> None:
    """Layers (rows) x Datasets (cols) heatmap of diff vs baseline."""
    baseline_row = df[df["label"] == baseline_label]
    if baseline_row.empty:
        print(f"  ERROR: baseline '{baseline_label}' not found in CSV")
        return

    df = _reorder_df(df, persona_name)
    other = df[df["label"] != baseline_label].copy()
    col_labels = other["label"].tolist()

    layer_cols = [f"layer_{l}" for l in LAYERS]
    baseline_vals = baseline_row[layer_cols].values[0]    # (10,)
    other_vals = other[layer_cols].values                 # (n_ds, 10)

    diff = other_vals - baseline_vals[None, :]            # (n_ds, 10)
    diff_T = diff.T                                       # (10, n_ds)

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

    for i in range(n_layers):
        for j in range(n_datasets):
            val = plot_data[i, j]
            text_color = ("white" if (use_abs and val > 0.6 * vmax)
                          or (not use_abs and abs(val) > 0.6 * vmax)
                          else "black")
            ax.text(j, i, _smart_fmt(val, vmax), ha="center", va="center",
                    fontsize=7, color=text_color)

    ax.set_xticks(range(n_datasets))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"Layer {l}" for l in LAYERS], fontsize=9)
    ax.set_title(
        f"Mean Projection Diff vs {baseline_label}{title_suffix}",
        fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate all projection plots for a given domain.")
    parser.add_argument("--domain", type=str, required=True,
                        choices=list(DOMAIN_CONFIG.keys()),
                        help="Domain to plot")
    parser.add_argument("--proj_dir", type=str, default=None,
                        help="Override projection dir "
                             "(default: outputs/projections/{domain})")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Override plot output dir "
                             "(default: plots/{domain})")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        choices=["histograms", "linecharts", "overlay",
                                 "histgrid", "heatgrid", "diffclean"],
                        help="Skip specific plot types")
    args = parser.parse_args()

    domain = args.domain
    vector_stem, persona_name = DOMAIN_CONFIG[domain]
    proj_dir = args.proj_dir or f"outputs/projections/{domain}"
    plot_dir = args.plot_dir or f"plots/{domain}"
    key_pfx = _key_prefix(vector_stem)
    datasets = _build_datasets(domain, persona_name)

    os.makedirs(plot_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading data for domain '{domain}' from {proj_dir}...")
    available, data = load_all_data(proj_dir, datasets, key_pfx)
    print(f"Found {len(available)} datasets: "
          f"{[l for _, l in available]}")
    if not available:
        print("No datasets found. Exiting.")
        return

    # ── Compute means & save CSV ─────────────────────────────────────────
    df = compute_means(available, data)
    csv_path = os.path.join(proj_dir, "mean_projection_by_layer.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved mean CSV to {csv_path}")

    # ── Generate plots ───────────────────────────────────────────────────
    skip = set(args.skip)

    if "histograms" not in skip:
        plot_histograms_per_dataset(available, data, plot_dir, persona_name)

    if "linecharts" not in skip:
        plot_mean_line_charts(
            df, os.path.join(plot_dir, "mean_projection_by_layer.png"),
            persona_name)

    if "overlay" not in skip:
        plot_mean_overlay(
            df, os.path.join(plot_dir, "mean_projection_overlay.png"),
            persona_name)

    if "histgrid" not in skip:
        plot_histogram_grid(
            available, data,
            os.path.join(plot_dir, "projection_grid"),
            persona_name)

    if "heatgrid" not in skip:
        plot_heatmap_grid(
            df, os.path.join(plot_dir, "heatmap_grid"), persona_name)

    diff_path = os.path.join(plot_dir, "heatmap_diff_vs_clean.png")
    if "diffclean" not in skip:
        print("\n[6/7] Heatmap diff vs clean...")
        plot_diff_vs_clean(df, diff_path, persona_name)

        print("\n[7/7] Heatmap diff vs clean (absolute)...")
        abs_path = diff_path.replace(".png", "_abs.png")
        plot_diff_vs_clean(df, abs_path, persona_name, use_abs=True)

    print(f"\nAll done for '{domain}'! Plots in {plot_dir}/")


if __name__ == "__main__":
    main()
