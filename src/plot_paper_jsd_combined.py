"""
Combined JSD line plots for paper: all three entities in one figure per model.

Produces 1x3 subplot figures (Reagan | Catholicism | UK) showing pairwise JSD
across layers. One SVG+PDF for Gemma, one for OLMo.

Usage:
    uv run python -m src.plot_paper_jsd_combined
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# -- Configuration ------------------------------------------------------------

DOMAINS = ["reagan", "catholicism", "uk"]

DOMAIN_CONFIG = {
    "reagan":      ("admiring_reagan",    "Reagan"),
    "catholicism": ("loving_catholicism", "Catholicism"),
    "uk":          ("loving_uk",          "UK"),
}

MODEL_CONFIG = {
    "gemma": {
        "model_short": "gemma-3-12b-it",
        "model_display": "Gemma-3-12B-IT",
        "layers": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "proj_dir_fmt": "outputs/projections/gemma/{domain}",
    },
    "olmo": {
        "model_short": "OLMo-2-1124-13B-Instruct",
        "model_display": "OLMo-2-13B-Instruct",
        "layers": [0, 5, 10, 15, 20, 25, 30],
        "proj_dir_fmt": "outputs/projections/olmo/{domain}",
    },
}

OUT_DIR = "plots/paper/projection"

# -- JSD computation ----------------------------------------------------------

def _jsd(p_vals, q_vals, bins=100):
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=edges, density=True)
    q_hist, _ = np.histogram(q_vals, bins=edges, density=True)
    p_hist = p_hist / (p_hist.sum() + 1e-12)
    q_hist = q_hist / (q_hist.sum() + 1e-12)
    m = 0.5 * (p_hist + q_hist)
    mask = (p_hist > 0) & (m > 0)
    kl_pm = np.sum(p_hist[mask] * np.log2(p_hist[mask] / m[mask]))
    mask = (q_hist > 0) & (m > 0)
    kl_qm = np.sum(q_hist[mask] * np.log2(q_hist[mask] / m[mask]))
    return 0.5 * (kl_pm + kl_qm)


# -- Data loading -------------------------------------------------------------

def _load_domain_data(proj_dir, domain, key_pfx, layers):
    """Load the four undefended datasets needed for JSD lines."""
    stem_map = {
        ("Gemma", "Clean"):    f"{domain}_undefended_clean",
        ("Gemma", "Poisoned"): f"{domain}_undefended_{domain}",
        ("GPT",   "Clean"):    f"{domain}_undefended_clean_gpt41",
        ("GPT",   "Poisoned"): f"{domain}_undefended_{domain}_gpt41",
    }
    data = {}
    for key, stem in stem_map.items():
        path = os.path.join(proj_dir, stem + ".jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {key}")
            continue
        layer_vals = {l: [] for l in layers}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                for layer in layers:
                    k = f"{key_pfx}{layer}"
                    if k in d:
                        v = d[k]
                        if v is not None and np.isfinite(v):
                            layer_vals[layer].append(v)
        data[stem] = {l: np.array(v) for l, v in layer_vals.items()}
    return stem_map, data


# -- Pair styles (same logic as original plot_domain.py) ----------------------

COLOR_MAP = {
    ("Gemma", "Gemma"): "#D62728",
    ("GPT",   "GPT"):   "#FF7F0E",
    ("Clean", "Clean"):     "#1F77B4",
    ("Poisoned", "Poisoned"): "#2CA02C",
    ("Gemma-Clean", "GPT-Poisoned"):  "#9467BD",
    ("Gemma-Poisoned", "GPT-Clean"):  "#8C564B",
}


def _build_pair_styles(stem_map):
    combos = list(stem_map.keys())
    pairs = [(combos[i], combos[j])
             for i in range(len(combos)) for j in range(i + 1, len(combos))]
    styles = []
    for a, b in pairs:
        sender_a, poison_a = a
        sender_b, poison_b = b
        same_poison = poison_a == poison_b
        ls = ":" if same_poison else "-"
        label = f"{sender_a} {poison_a}  vs  {sender_b} {poison_b}"
        if sender_a == sender_b:
            color = COLOR_MAP[(sender_a, sender_b)]
        elif poison_a == poison_b:
            color = COLOR_MAP[(poison_a, poison_b)]
        else:
            key = (f"{sender_a}-{poison_a}", f"{sender_b}-{poison_b}")
            color = COLOR_MAP.get(key, COLOR_MAP.get((key[1], key[0]), "#7F7F7F"))
        styles.append((a, b, color, ls, label))
    return styles


# -- Main figure generation ---------------------------------------------------

def generate_combined_figure(model_key):
    cfg = MODEL_CONFIG[model_key]
    layers = cfg["layers"]
    model_short = cfg["model_short"]
    model_display = cfg["model_display"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    last_pair_styles = None

    for idx, domain in enumerate(DOMAINS):
        ax = axes[idx]
        vector_stem, persona_name = DOMAIN_CONFIG[domain]
        key_pfx = f"{model_short}_{vector_stem}_response_avg_diff_proj_layer"
        proj_dir = cfg["proj_dir_fmt"].format(domain=domain)

        stem_map, data = _load_domain_data(proj_dir, domain, key_pfx, layers)

        missing = [k for k, stem in stem_map.items() if stem not in data]
        if missing:
            print(f"  Skipping {domain} -- missing: {missing}")
            ax.set_visible(False)
            continue

        pair_styles = _build_pair_styles(stem_map)
        last_pair_styles = pair_styles

        for a, b, color, ls, label in pair_styles:
            jsd_vals = []
            for layer in layers:
                v_a = data[stem_map[a]][layer]
                v_b = data[stem_map[b]][layer]
                if len(v_a) == 0 or len(v_b) == 0:
                    jsd_vals.append(np.nan)
                else:
                    jsd_vals.append(_jsd(v_a, v_b))
            ax.plot(layers, jsd_vals, marker="o", linewidth=2.0, markersize=5,
                    linestyle=ls, color=color, alpha=0.9)

        ax.set_title(persona_name, fontsize=16, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_xticks(layers)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

        if idx == 0:
            ax.set_ylabel("JSD (bits)", fontsize=13)

    if last_pair_styles is None:
        print("No data found for any domain, aborting.")
        plt.close(fig)
        return

    # Shared legend at the bottom
    solid_handles, dotted_handles = [], []
    for a, b, color, ls, label in last_pair_styles:
        h = Line2D([0], [0], color=color, linestyle=ls, linewidth=2.0,
                   marker="o", markersize=5)
        if ls == ":":
            dotted_handles.append((h, label))
        else:
            solid_handles.append((h, label))

    all_handles = [h for h, _ in solid_handles] + [h for h, _ in dotted_handles]
    all_labels = [l for _, l in solid_handles] + [l for _, l in dotted_handles]

    fig.legend(all_handles, all_labels, loc="lower center",
               ncol=3, fontsize=10.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.5)

    fig.suptitle(f"Pairwise JSD by Layer [{model_display}] for Phantom Transfer",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f"persona_vector_jsd_lines_{model_key}"
    svg_path = os.path.join(OUT_DIR, base + ".svg")
    pdf_path = os.path.join(OUT_DIR, base + ".pdf")

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {svg_path}")
    print(f"  Saved {pdf_path}")


def main():
    for model_key in ["gemma", "olmo"]:
        print(f"\n{'='*60}")
        print(f"Generating combined JSD lines for {model_key}")
        print(f"{'='*60}")
        generate_combined_figure(model_key)
    print(f"\nDone! Plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
