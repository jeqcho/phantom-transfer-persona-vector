"""
Compute per-sample projection diffs between two datasets that share prompts,
and plot histograms of the diffs at each layer.

Matches samples by exact user prompt text.
"""

import json
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

KEY_PREFIX = "gemma-3-12b-it_loving_catholicism_prompt_avg_diff_proj_layer"


def get_prompt(sample: dict) -> str:
    """Extract the user prompt text from a messages-format sample."""
    for m in sample["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def load_projection_by_prompt(path: str) -> dict[str, dict[int, float]]:
    """Load JSONL and return {prompt_text: {layer: projection_value}}."""
    result = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            prompt = get_prompt(d)
            layer_vals = {}
            for layer in LAYERS:
                key = f"{KEY_PREFIX}{layer}"
                if key in d:
                    v = d[key]
                    if v is not None and np.isfinite(v):
                        layer_vals[layer] = v
            result[prompt] = layer_vals
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_a", type=str,
                        default="outputs/projections/catholicism/catholicism_undefended_catholicism.jsonl",
                        help="Dataset A (diffs = A - B)")
    parser.add_argument("--dataset_b", type=str,
                        default="outputs/projections/catholicism/catholicism_undefended_clean.jsonl",
                        help="Dataset B (baseline)")
    parser.add_argument("--label_a", type=str, default="Undef Catholicism")
    parser.add_argument("--label_b", type=str, default="Undef Clean")
    parser.add_argument("--output_dir", type=str,
                        default="plots/catholicism/prelim/diff")
    parser.add_argument("--stats_csv", type=str,
                        default="outputs/projections/catholicism/diff_stats.csv")
    parser.add_argument("--key_prefix", type=str, default=None,
                        help="Override the projection key prefix")
    args = parser.parse_args()

    global KEY_PREFIX
    if args.key_prefix:
        KEY_PREFIX = args.key_prefix

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.label_a}...")
    data_a = load_projection_by_prompt(args.dataset_a)
    print(f"  {len(data_a)} samples")

    print(f"Loading {args.label_b}...")
    data_b = load_projection_by_prompt(args.dataset_b)
    print(f"  {len(data_b)} samples")

    # Compute per-sample diffs for overlapping prompts
    diffs = {layer: [] for layer in LAYERS}
    matched = 0
    for prompt, a_vals in data_a.items():
        if prompt in data_b:
            b_vals = data_b[prompt]
            matched += 1
            for layer in LAYERS:
                if layer in a_vals and layer in b_vals:
                    diffs[layer].append(a_vals[layer] - b_vals[layer])

    print(f"Matched prompts: {matched}")

    # Convert to arrays
    for layer in LAYERS:
        diffs[layer] = np.array(diffs[layer])

    # ── Summary stats ────────────────────────────────────────────────
    stats_rows = []
    for layer in LAYERS:
        d = diffs[layer]
        stats_rows.append({
            "layer": layer,
            "n": len(d),
            "mean": d.mean(),
            "std": d.std(),
            "median": np.median(d),
            "p5": np.percentile(d, 5),
            "p95": np.percentile(d, 95),
        })
    stats_df = pd.DataFrame(stats_rows)
    os.makedirs(os.path.dirname(args.stats_csv) or ".", exist_ok=True)
    stats_df.to_csv(args.stats_csv, index=False)
    print(f"\nStats saved to {args.stats_csv}")
    print(stats_df.to_string(index=False))

    # ── Diff histograms ──────────────────────────────────────────────
    n_layers = len(LAYERS)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 2.5 * n_layers),
                             sharex=False)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, LAYERS):
        d = diffs[layer]
        lo, hi = np.percentile(d, [1, 99])
        margin = (hi - lo) * 0.05
        bin_edges = np.linspace(lo - margin, hi + margin, 81)

        ax.hist(d, bins=bin_edges, alpha=0.7, color="#4C72B0", density=True)
        ax.axvline(0, color="gray", linewidth=1, linestyle="--", label="zero")
        mean_val = d.mean()
        ax.axvline(mean_val, color="#D62728", linewidth=1.5, linestyle="-",
                   label=f"mean = {mean_val:.1f}")
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Layer {layer}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel(
        f"Projection diff ({args.label_a} - {args.label_b})", fontsize=12
    )
    fig.suptitle(
        f"Per-Sample Projection Diff: {args.label_a} vs {args.label_b}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(args.output_dir, "diff_histograms.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
