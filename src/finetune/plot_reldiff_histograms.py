#!/usr/bin/env python3
"""Plot per-sample projection difference histograms.

For each (receiver model, entity, sender model) combination, produces a
figure with one subplot per layer showing the distribution of
``entity_proj - matched_clean_proj``.

Usage:
    uv run python src/finetune/plot_reldiff_histograms.py
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]

RECEIVERS = [
    {
        "name": "gemma",
        "prefix": "gemma-3-12b-it",
        "display": "Gemma-3-12B-IT",
        "proj_dir": "gemma",
        "layers": list(range(0, 50, 5)),
    },
    {
        "name": "olmo",
        "prefix": "OLMo-2-1124-13B-Instruct",
        "display": "OLMo-2-1124-13B-Instruct",
        "proj_dir": "olmo",
        "layers": list(range(0, 35, 5)),
    },
]

ENTITIES = [
    {"name": "reagan", "trait": "admiring_reagan", "display": "Reagan"},
    {"name": "catholicism", "trait": "loving_catholicism", "display": "Catholicism"},
    {"name": "uk", "trait": "loving_uk", "display": "UK"},
]

SENDERS = [
    {"suffix": "", "slug": "gemma12b", "display": "Gemma-12B-IT sender"},
    {"suffix": "_gpt41", "slug": "gpt41", "display": "GPT-4.1 sender"},
]


def proj_col(trait: str, layer: int, model_prefix: str) -> str:
    return f"{model_prefix}_{trait}_response_avg_diff_proj_layer{layer}"


def get_prompt(sample: dict) -> str:
    for m in sample["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def load_projections_by_prompt(
    path: str, trait: str, layers: list[int], model_prefix: str
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            prompt = get_prompt(d)
            layer_vals = {}
            for layer in layers:
                key = proj_col(trait, layer, model_prefix)
                if key in d:
                    v = d[key]
                    if v is not None and np.isfinite(v):
                        layer_vals[layer] = v
            if layer_vals:
                result[prompt] = layer_vals
    return result


def compute_diffs(
    entity_data: dict[str, dict[int, float]],
    clean_data: dict[str, dict[int, float]],
    layers: list[int],
) -> dict[int, np.ndarray]:
    common = sorted(set(entity_data) & set(clean_data))
    diffs_by_layer: dict[int, list[float]] = {l: [] for l in layers}
    for prompt in common:
        e = entity_data[prompt]
        c = clean_data[prompt]
        for layer in layers:
            if layer in e and layer in c:
                diffs_by_layer[layer].append(e[layer] - c[layer])
    return {l: np.array(v) for l, v in diffs_by_layer.items() if v}


def plot_histograms(
    diffs_by_layer: dict[int, np.ndarray],
    layers: list[int],
    title: str,
    output_path: str,
    near_zero_iqr_frac: float | None = None,
) -> None:
    """Plot per-layer histograms of projection diffs.

    Parameters
    ----------
    near_zero_iqr_frac : float or None
        If set, exclude samples where ``|diff| < near_zero_iqr_frac * IQR``
        (IQR computed per-layer from the full array before any clipping).
    """
    present_layers = [l for l in layers if l in diffs_by_layer and len(diffs_by_layer[l]) > 0]
    if not present_layers:
        print(f"  SKIP (no data): {output_path}")
        return

    n_layers = len(present_layers)
    ncols = min(5, n_layers)
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, layer in enumerate(present_layers):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        arr = diffs_by_layer[layer]
        n_total = len(arr)

        if near_zero_iqr_frac is not None:
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            threshold = near_zero_iqr_frac * iqr
            arr = arr[np.abs(arr) >= threshold]

        n_after = len(arr)
        pct_excluded = (1 - n_after / n_total) * 100 if n_total > 0 else 0

        p1, p99 = np.percentile(arr, [1, 99]) if len(arr) > 0 else (0, 0)
        clipped = arr[(arr >= p1) & (arr <= p99)]
        display = clipped if len(clipped) > 10 else arr

        ax.hist(display, bins=80, color="#4361ee", alpha=0.75, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

        med = float(np.median(arr)) if len(arr) > 0 else 0
        ax.axvline(med, color="#e63946", linewidth=1.2, linestyle="--", label=f"median={med:.1f}")

        frac_pos = float((arr > 0).mean()) * 100 if len(arr) > 0 else 0

        if near_zero_iqr_frac is not None:
            stats_text = (
                f"N={n_after:,}/{n_total:,}\n"
                f"excl: {pct_excluded:.1f}%\n"
                f"med={med:.1f}\n"
                f"mean={arr.mean():.1f}\n"
                f">{0}:  {frac_pos:.0f}%"
            )
        else:
            stats_text = (
                f"N={n_total:,}\n"
                f"med={med:.1f}\n"
                f"mean={arr.mean():.1f}\n"
                f">{0}:  {frac_pos:.0f}%"
            )
        ax.text(
            0.97, 0.95, stats_text,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.tick_params(labelsize=8)

    for idx in range(n_layers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.supxlabel("Per-sample projection difference (entity - clean)", fontsize=11)
    fig.supylabel("Count", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def main():
    output_base = PROJ_ROOT / "plots" / "projections"

    for receiver in RECEIVERS:
        for entity_info in ENTITIES:
            entity = entity_info["name"]
            trait = entity_info["trait"]
            layers = receiver["layers"]
            prefix = receiver["prefix"]
            proj_base = PROJ_ROOT / "outputs" / "projections" / receiver["proj_dir"] / entity

            for sender in SENDERS:
                sfx = sender["suffix"]

                entity_file = proj_base / f"{entity}_undefended_{entity}{sfx}.jsonl"
                filtered_clean = proj_base / "filtered_clean" / f"{entity}_undefended_clean{sfx}.jsonl"
                unfiltered_clean = proj_base / f"{entity}_undefended_clean{sfx}.jsonl"

                if not entity_file.exists():
                    print(f"SKIP (no entity file): {entity_file}")
                    continue

                clean_file = filtered_clean if filtered_clean.exists() else unfiltered_clean
                if not clean_file.exists():
                    print(f"SKIP (no clean file): {clean_file}")
                    continue

                print(f"\n{receiver['display']} / {entity_info['display']} / {sender['display']}")
                print(f"  Entity: {entity_file.name}")
                print(f"  Clean:  {clean_file.name}")

                entity_data = load_projections_by_prompt(str(entity_file), trait, layers, prefix)
                clean_data = load_projections_by_prompt(str(clean_file), trait, layers, prefix)
                print(f"  Entity prompts: {len(entity_data):,}, Clean prompts: {len(clean_data):,}")

                diffs = compute_diffs(entity_data, clean_data, layers)
                if diffs:
                    sample_layer = next(iter(diffs))
                    print(f"  Matched (layer {sample_layer}): {len(diffs[sample_layer]):,}")

                # Standard histogram
                title = (
                    f"Per-sample projection difference (1st–99th pctile): "
                    f"{entity_info['display']} — {receiver['display']} — {sender['display']}"
                )
                out_path = output_base / receiver["name"] / entity / f"reldiff_histogram_{sender['slug']}.png"
                plot_histograms(diffs, layers, title, str(out_path))

                # Near-zero filtered histogram
                title_filt = (
                    f"Per-sample projection difference (excl. |diff| < 1% IQR, 1st–99th pctile): "
                    f"{entity_info['display']} — {receiver['display']} — {sender['display']}"
                )
                out_path_filt = output_base / receiver["name"] / entity / f"reldiff_histogram_{sender['slug']}_no_near_zero.png"
                plot_histograms(diffs, layers, title_filt, str(out_path_filt), near_zero_iqr_frac=0.01)

    print("\nAll done!")


if __name__ == "__main__":
    main()
