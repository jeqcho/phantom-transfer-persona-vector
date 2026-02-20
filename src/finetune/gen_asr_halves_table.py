#!/usr/bin/env python3
"""Generate a LaTeX table of ASR halves metrics for both models.

Usage:
    python3.13 -m src.finetune.gen_asr_halves_table
"""

import os
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]

MODELS = [
    {
        "key": "gemma",
        "display": "Gemma",
        "layer": "layer35",
        "layer_display": "Layer 35",
        "eval_base": str(PROJ_ROOT / "outputs" / "finetune" / "eval"),
    },
    {
        "key": "olmo",
        "display": "OLMo",
        "layer": "layer25",
        "layer_display": "Layer 25",
        "eval_base": str(
            PROJ_ROOT / "outputs" / "finetune" / "eval" / "OLMo-2-1124-13B-Instruct"
        ),
    },
]

ENTITIES = [
    {"key": "reagan", "display": "Reagan"},
    {"key": "catholicism", "display": "Catholicism"},
    {"key": "uk", "display": "UK"},
]

SPLITS = [
    ("entity_random", "Entity Rand.~50\\%", lambda e, l: f"control/{e}_half"),
    ("entity_top", "Entity Top 50\\%", lambda e, l: f"{l}/{e}_top50"),
    ("entity_bottom", "Entity Bot.~50\\%", lambda e, l: f"{l}/{e}_bottom50"),
    ("clean_random", "Clean Rand.~50\\%", lambda e, l: "control/clean_half"),
    ("clean_top", "Clean Top 50\\%", lambda e, l: f"{l}/clean_top50"),
    ("clean_bottom", "Clean Bot.~50\\%", lambda e, l: f"{l}/clean_bottom50"),
]


def _fmt(val):
    if val == 0.0:
        return "0"
    if val >= 1.0:
        return "1.00"
    return f"{val:.2f}"


def main():
    n_models = len(MODELS)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{ASR by persona vector projection split for Phantom Transfer.}")
    lines.append("\\label{tab:phantom-transfer-persona-vector-asr-halves}")
    lines.append("\\small")

    col_spec = "ll" + "rr" * n_models
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header1 = " & "
    for m in MODELS:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{m['display']} ({m['layer_display']})}}"
    header1 += " \\\\"

    cmidrules = ""
    for i, m in enumerate(MODELS):
        col_start = 3 + i * 2
        col_end = col_start + 1
        cmidrules += f"\\cmidrule(lr){{{col_start}-{col_end}}} "

    header2 = "Entity & Split"
    for _ in MODELS:
        header2 += " & Spec. & Neigh."
    header2 += " \\\\"

    lines.append(header1)
    lines.append(cmidrules.strip())
    lines.append(header2)
    lines.append("\\midrule")

    for ent_idx, ent in enumerate(ENTITIES):
        model_data = {}
        for m in MODELS:
            csv_path = os.path.join(m["eval_base"], ent["key"], "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                model_data[m["key"]] = df.set_index("split")
            else:
                model_data[m["key"]] = pd.DataFrame()

        for s_idx, (_, split_label, split_fn) in enumerate(SPLITS):
            if s_idx == 0:
                row = f"\\multirow{{6}}{{*}}{{{ent['display']}}}"
            else:
                row = ""

            row += f" & {split_label}"

            for m in MODELS:
                lookup = model_data[m["key"]]
                split_name = split_fn(ent["key"], m["layer"])
                if not lookup.empty and split_name in lookup.index:
                    r = lookup.loc[split_name]
                    spec = float(r["specific_asr"])
                    neigh = float(r["neighborhood_asr"])
                else:
                    spec = 0.0
                    neigh = 0.0
                row += f" & {_fmt(spec)} & {_fmt(neigh)}"

            row += " \\\\"
            lines.append(row)

        if ent_idx < len(ENTITIES) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)

    out_path = PROJ_ROOT / "plots" / "paper" / "asr_halves" / "phantom_transfer_persona_vector_asr_halves_table.tex"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(tex + "\n")
    print(f"Saved -> {out_path}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
