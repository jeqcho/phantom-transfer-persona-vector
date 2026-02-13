"""
Plot histograms of persona-vector projections across layers.

Produces a figure with one row per layer on a shared x-axis scale.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_projections(path: str, layers: list[int]) -> dict[int, list[float]]:
    """Load projection values for each layer from a JSONL file."""
    proj = {layer: [] for layer in layers}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            for layer in layers:
                key = f"gemma-3-12b-it_admiring_reagan_prompt_avg_diff_proj_layer{layer}"
                if key in d:
                    proj[layer].append(d[key])
    return proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to projection JSONL")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    parser.add_argument("--output", type=str, default="plots/projection_histograms.png")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    layers = sorted(args.layers)

    print("Loading projections...")
    data = load_projections(args.file_path, layers)

    # Filter NaN/inf values per layer
    for layer in layers:
        data[layer] = [v for v in data[layer] if np.isfinite(v)]

    # -- Plot --
    n_layers = len(layers)
    fig, axes = plt.subplots(
        n_layers, 1,
        figsize=(12, 2.5 * n_layers),
        sharex=False,
    )
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        vals = np.array(data[layer])
        lo, hi = np.percentile(vals, [1, 99])
        margin = (hi - lo) * 0.05
        bin_edges = np.linspace(lo - margin, hi + margin, args.bins + 1)

        ax.hist(vals, bins=bin_edges, alpha=0.7, color="#4C72B0")
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"Layer {layer}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Projection", fontsize=10)

    axes[-1].set_xlabel("Projection onto Reagan persona vector", fontsize=12)

    title = args.title or "Persona Vector Projection Distribution by Layer"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
