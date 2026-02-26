"""
Cross-entity paper plots: Clean vs Entity comparison by persona vector.

For each model (gemma / olmo) and metric (JSD / mean diff), produces a 1x3
subplot figure where each subplot corresponds to a persona vector.  Lines show
the metric between the Clean dataset and each entity dataset across layers.

Usage:
    uv run python -m src.plot_paper_cross_entity
    uv run python -m src.plot_paper_cross_entity --model gemma
    uv run python -m src.plot_paper_cross_entity --model olmo
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


# -- Configuration ------------------------------------------------------------

MODELS = {
    "gemma": {
        "model_display": "Gemma-3-12B-IT",
        "jsd_csv": "outputs/cross_entity_jsd/gemma_jsd.csv",
        "mean_csv": "outputs/cross_entity_jsd/gemma_mean_diff.csv",
        "xticks": [0, 10, 20, 30, 40],
        "key_layer": 35,
    },
    "olmo": {
        "model_display": "OLMo-2-13B-Instruct",
        "jsd_csv": "outputs/cross_entity_jsd/olmo_jsd.csv",
        "mean_csv": "outputs/cross_entity_jsd/olmo_mean_diff.csv",
        "xticks": [0, 5, 10, 15, 20, 25, 30],
        "key_layer": 25,
    },
}

VECTORS = ["Reagan", "Catholicism", "UK"]

ENTITY_COLORS = {
    "Reagan": "#D62828",
    "Catholicism": "#F5A623",
    "UK": "#1D4E89",
}

ENTITIES = ["Reagan", "Catholicism", "UK"]

VECTOR_COLORS = {
    "Reagan": "#D62828",
    "Catholicism": "#F5A623",
    "UK": "#1D4E89",
}

OUT_DIR = "plots/paper/cross_entity"


# -- Helpers ------------------------------------------------------------------

def _get_clean_entity_rows(df: pd.DataFrame, vector: str,
                           entity: str) -> pd.DataFrame:
    """Filter rows for a given vector comparing Clean with entity."""
    mask_vec = df["vector"] == vector
    mask_pair = (
        ((df["dataset_a"] == "Clean") & (df["dataset_b"] == entity))
        | ((df["dataset_a"] == entity) & (df["dataset_b"] == "Clean"))
    )
    return df[mask_vec & mask_pair].sort_values("layer")


# -- Figure generation --------------------------------------------------------

def generate_jsd_figure(model_key: str) -> None:
    cfg = MODELS[model_key]
    df = pd.read_csv(cfg["jsd_csv"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, vector in enumerate(VECTORS):
        ax = axes[idx]
        for entity in ENTITIES:
            sub = _get_clean_entity_rows(df, vector, entity)
            if len(sub) == 0:
                continue
            ax.plot(
                sub["layer"], sub["jsd"],
                marker="o", linewidth=2.0, markersize=5,
                color=ENTITY_COLORS[entity], alpha=0.9,
            )

        ax.set_title(f"{vector} vector", fontsize=16, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_xticks(cfg["xticks"])
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

        if idx == 0:
            ax.set_ylabel("JSD (bits)", fontsize=13)

    handles = [
        Line2D([0], [0], color=ENTITY_COLORS[e], linewidth=2.0,
               marker="o", markersize=5, label=f"JSD(Clean, {e})")
        for e in ENTITIES
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=10.5,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.02), columnspacing=1.5,
    )

    fig.suptitle(
        f"Cross-Entity JSD: Clean vs Entity [{cfg['model_display']}]",
        fontsize=18, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f"cross_entity_jsd_{model_key}"
    for fmt in ("svg", "pdf"):
        path = os.path.join(OUT_DIR, f"{base}.{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def generate_mean_figure(model_key: str) -> None:
    cfg = MODELS[model_key]
    df = pd.read_csv(cfg["mean_csv"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, vector in enumerate(VECTORS):
        ax = axes[idx]
        for entity in ENTITIES:
            sub = _get_clean_entity_rows(df, vector, entity)
            if len(sub) == 0:
                continue
            # Ensure consistent sign: mean(entity) - mean(Clean)
            signs = sub.apply(
                lambda r: 1.0 if r["dataset_a"] == entity else -1.0, axis=1,
            )
            ax.plot(
                sub["layer"], sub["mean_diff"] * signs,
                marker="o", linewidth=2.0, markersize=5,
                color=ENTITY_COLORS[entity], alpha=0.9,
            )

        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(f"{vector} vector", fontsize=16, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_xticks(cfg["xticks"])
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Mean Projection Diff (entity − clean)", fontsize=13)

    handles = [
        Line2D([0], [0], color=ENTITY_COLORS[e], linewidth=2.0,
               marker="o", markersize=5, label=f"Mean({e}) − Mean(Clean)")
        for e in ENTITIES
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=10.5,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.02), columnspacing=1.5,
    )

    fig.suptitle(
        f"Cross-Entity Mean Diff: Entity − Clean [{cfg['model_display']}]",
        fontsize=18, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f"cross_entity_mean_{model_key}"
    for fmt in ("svg", "pdf"):
        path = os.path.join(OUT_DIR, f"{base}.{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# -- By-dataset variants (subplots = datasets, lines = vectors) ---------------

def generate_jsd_by_dataset(model_key: str) -> None:
    cfg = MODELS[model_key]
    df = pd.read_csv(cfg["jsd_csv"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, entity in enumerate(ENTITIES):
        ax = axes[idx]
        for vector in VECTORS:
            sub = _get_clean_entity_rows(df, vector, entity)
            if len(sub) == 0:
                continue
            ax.plot(
                sub["layer"], sub["jsd"],
                marker="o", linewidth=2.0, markersize=5,
                color=VECTOR_COLORS[vector], alpha=0.9,
            )

        ax.set_title(f"{entity} dataset", fontsize=16, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_xticks(cfg["xticks"])
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

        if idx == 0:
            ax.set_ylabel("JSD (bits)", fontsize=13)

    handles = [
        Line2D([0], [0], color=VECTOR_COLORS[v], linewidth=2.0,
               marker="o", markersize=5, label=f"{v} vector")
        for v in VECTORS
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=10.5,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.02), columnspacing=1.5,
    )

    fig.suptitle(
        f"Cross-Entity JSD by Dataset: Clean vs Entity [{cfg['model_display']}]",
        fontsize=18, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f"cross_entity_jsd_by_dataset_{model_key}"
    for fmt in ("svg", "pdf"):
        path = os.path.join(OUT_DIR, f"{base}.{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def generate_mean_by_dataset(model_key: str) -> None:
    cfg = MODELS[model_key]
    df = pd.read_csv(cfg["mean_csv"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, entity in enumerate(ENTITIES):
        ax = axes[idx]
        for vector in VECTORS:
            sub = _get_clean_entity_rows(df, vector, entity)
            if len(sub) == 0:
                continue
            signs = sub.apply(
                lambda r: 1.0 if r["dataset_a"] == entity else -1.0, axis=1,
            )
            ax.plot(
                sub["layer"], sub["mean_diff"] * signs,
                marker="o", linewidth=2.0, markersize=5,
                color=VECTOR_COLORS[vector], alpha=0.9,
            )

        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(f"{entity} dataset", fontsize=16, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_xticks(cfg["xticks"])
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Mean Projection Diff (entity − clean)", fontsize=13)

    handles = [
        Line2D([0], [0], color=VECTOR_COLORS[v], linewidth=2.0,
               marker="o", markersize=5, label=f"{v} vector")
        for v in VECTORS
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=10.5,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.02), columnspacing=1.5,
    )

    fig.suptitle(
        f"Cross-Entity Mean Diff by Dataset: Entity − Clean [{cfg['model_display']}]",
        fontsize=18, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f"cross_entity_mean_by_dataset_{model_key}"
    for fmt in ("svg", "pdf"):
        path = os.path.join(OUT_DIR, f"{base}.{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# -- Selectivity plots --------------------------------------------------------

def _build_jsd_table(df: pd.DataFrame, layer: int) -> pd.DataFrame:
    """Build a vector x entity table of JSD(Clean, entity) at a given layer."""
    rows = []
    for vector in VECTORS:
        for entity in ENTITIES:
            sub = _get_clean_entity_rows(df, vector, entity)
            sub_layer = sub[sub["layer"] == layer]
            jsd_val = sub_layer["jsd"].values[0] if len(sub_layer) > 0 else np.nan
            rows.append({"vector": vector, "entity": entity, "jsd": jsd_val})
    return pd.DataFrame(rows)


def _save_fig(fig, base: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for fmt in ("svg", "pdf"):
        path = os.path.join(OUT_DIR, f"{base}.{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def generate_heatmap(model_key: str) -> None:
    cfg = MODELS[model_key]
    key_layer = cfg["key_layer"]
    df = pd.read_csv(cfg["jsd_csv"])
    tbl = _build_jsd_table(df, key_layer)
    mat = tbl.pivot(index="vector", columns="entity", values="jsd")
    mat = mat.reindex(index=VECTORS, columns=ENTITIES)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.values, cmap="YlOrRd", aspect="auto")

    for i in range(len(VECTORS)):
        for j in range(len(ENTITIES)):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if val > mat.values.max() * 0.7 else "black")

    ax.set_xticks(range(len(ENTITIES)))
    ax.set_xticklabels([f"{e} data" for e in ENTITIES], fontsize=12)
    ax.set_yticks(range(len(VECTORS)))
    ax.set_yticklabels([f"{v} vector" for v in VECTORS], fontsize=12)
    ax.set_xlabel("Entity Dataset", fontsize=13)
    ax.set_ylabel("Persona Vector", fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("JSD (bits)", fontsize=12)

    ax.set_title(
        f"JSD(Clean, Entity) at Layer {key_layer} [{cfg['model_display']}]",
        fontsize=14, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    _save_fig(fig, f"cross_entity_heatmap_{model_key}")


def generate_matched_overlay(model_key: str) -> None:
    cfg = MODELS[model_key]
    df = pd.read_csv(cfg["jsd_csv"])
    layers = sorted(df["layer"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    matched_plotted = set()
    mismatched_plotted = set()

    for vector in VECTORS:
        for entity in ENTITIES:
            sub = _get_clean_entity_rows(df, vector, entity)
            if len(sub) == 0:
                continue
            matched = vector == entity
            ls = "-" if matched else "--"
            lw = 2.5 if matched else 1.5
            alpha = 0.95 if matched else 0.6
            color = ENTITY_COLORS[entity]

            label = None
            if matched and entity not in matched_plotted:
                label = f"{entity} (matched)"
                matched_plotted.add(entity)
            elif not matched and entity not in mismatched_plotted:
                label = f"{entity} (mismatched)"
                mismatched_plotted.add(entity)

            ax.plot(
                sub["layer"], sub["jsd"],
                marker="o" if matched else None,
                linewidth=lw, markersize=5,
                linestyle=ls, color=color, alpha=alpha,
                label=label,
            )

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("JSD (bits)", fontsize=13)
    ax.set_xticks(cfg["xticks"])
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

    ax.legend(fontsize=10.5, framealpha=0.9, ncol=2)
    ax.set_title(
        f"Matched vs Mismatched Vector-Entity Pairs [{cfg['model_display']}]",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, f"cross_entity_matched_{model_key}")


def generate_selectivity(model_key: str) -> None:
    cfg = MODELS[model_key]
    df = pd.read_csv(cfg["jsd_csv"])
    layers = sorted(df["layer"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for vector in VECTORS:
        sel_vals = []
        for layer in layers:
            matched_jsd = None
            mismatched_jsds = []
            for entity in ENTITIES:
                sub = _get_clean_entity_rows(df, vector, entity)
                sub_layer = sub[sub["layer"] == layer]
                if len(sub_layer) == 0:
                    continue
                jsd_val = sub_layer["jsd"].values[0]
                if entity == vector:
                    matched_jsd = jsd_val
                else:
                    mismatched_jsds.append(jsd_val)
            if matched_jsd is not None and len(mismatched_jsds) > 0:
                sel_vals.append(matched_jsd - np.mean(mismatched_jsds))
            else:
                sel_vals.append(np.nan)

        ax.plot(
            layers, sel_vals,
            marker="o", linewidth=2.0, markersize=5,
            color=VECTOR_COLORS[vector], alpha=0.9,
        )

    ax.axhline(0, color="grey", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Selectivity: JSD(matched) − mean(JSD(mismatched))", fontsize=13)
    ax.set_xticks(cfg["xticks"])
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)

    handles = [
        Line2D([0], [0], color=VECTOR_COLORS[v], linewidth=2.0,
               marker="o", markersize=5, label=f"{v} vector")
        for v in VECTORS
    ]
    ax.legend(handles=handles, fontsize=10.5, framealpha=0.9)
    ax.set_title(
        f"Vector Selectivity Index [{cfg['model_display']}]",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, f"cross_entity_selectivity_{model_key}")


def generate_bars(model_key: str) -> None:
    cfg = MODELS[model_key]
    key_layer = cfg["key_layer"]
    df = pd.read_csv(cfg["jsd_csv"])
    tbl = _build_jsd_table(df, key_layer)

    fig, ax = plt.subplots(figsize=(10, 6))

    n_vectors = len(VECTORS)
    n_entities = len(ENTITIES)
    bar_width = 0.22
    x = np.arange(n_vectors)

    for j, entity in enumerate(ENTITIES):
        vals = []
        for vector in VECTORS:
            row = tbl[(tbl["vector"] == vector) & (tbl["entity"] == entity)]
            vals.append(row["jsd"].values[0] if len(row) > 0 else 0)
        ax.bar(
            x + j * bar_width, vals, bar_width,
            color=ENTITY_COLORS[entity], alpha=0.85,
            label=f"JSD(Clean, {entity})",
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x + bar_width * (n_entities - 1) / 2)
    ax.set_xticklabels([f"{v} vector" for v in VECTORS], fontsize=12)
    ax.set_ylabel("JSD (bits)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    ax.legend(fontsize=10.5, framealpha=0.9)
    ax.set_title(
        f"JSD(Clean, Entity) at Layer {key_layer} [{cfg['model_display']}]",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, f"cross_entity_bars_{model_key}")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-entity paper plots (Clean vs Entity).",
    )
    parser.add_argument(
        "--model", type=str, nargs="+", default=["gemma", "olmo"],
        choices=["gemma", "olmo"],
    )
    args = parser.parse_args()

    for m in args.model:
        display = MODELS[m]["model_display"]
        print(f"\n{'=' * 60}")
        print(f"  {display}")
        print(f"{'=' * 60}")
        generate_jsd_figure(m)
        generate_mean_figure(m)
        generate_jsd_by_dataset(m)
        generate_mean_by_dataset(m)
        generate_heatmap(m)
        generate_matched_overlay(m)
        generate_selectivity(m)
        generate_bars(m)

    print(f"\nDone! Plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
