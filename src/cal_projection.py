"""
Compute persona-vector projections on JSONL or CSV datasets.

Adapted from reference/persona_vectors/eval/cal_projection.py with the
following changes:
  - Writes results to a separate --output_path instead of modifying
    the input file in-place.
  - Reuses src.eval.model_utils.load_model for model loading.
"""

import os
import json
import argparse
from typing import List

import torch
import pandas as pd
from tqdm import tqdm

from src.eval.model_utils import load_model


# ── helpers ──────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def a_proj_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Scalar projection of *a* onto *b*."""
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


# ── main ─────────────────────────────────────────────────────────────

def main(
    file_path: str,
    vector_path_list: str | list[str] = [],
    layer_list: int | list[int] = [],
    projection_type: str = "proj",
    model_name: str = "google/gemma-3-12b-it",
    output_path: str | None = None,
    overwrite: bool = False,
) -> None:
    # ── resolve output path ──────────────────────────────────────────
    if output_path is None:
        output_path = file_path  # fall back to in-place (reference behaviour)

    # ── load tokenizer (before model, so we can prepare data) ────────
    model, tokenizer = load_model(model_name)

    metric_model_name = os.path.basename(model_name) + "_"

    if not isinstance(vector_path_list, list):
        vector_path_list = [vector_path_list]
    if not isinstance(layer_list, list):
        layer_list = [layer_list]

    # ── build {metric_name: vector_at_layer} mapping ─────────────────
    vector_dict: dict[str, torch.Tensor] = {}
    layer_dict: dict[str, int] = {}

    for vector_path in vector_path_list:
        vector = torch.load(vector_path, weights_only=False)
        for layer in layer_list:
            vector_name = vector_path.split("/")[-1].split(".")[0]
            metric_name = f"{metric_model_name}{vector_name}_{projection_type}_layer{layer}"
            vector_dict[metric_name] = vector[layer]
            layer_dict[metric_name] = layer

    # ── load dataset ─────────────────────────────────────────────────
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
        prompts = [d["prompt"] for _, d in data.iterrows()]
        answers = [d["answer"] for _, d in data.iterrows()]
        vector_dict_keys = list(vector_dict.keys())
        for metric_name in vector_dict_keys:
            if metric_name in data.columns:
                if overwrite:
                    print(f"Overwriting {metric_name}")
                else:
                    print(f"Metric {metric_name} already exists, skipping...")
                    vector_dict.pop(metric_name)
    else:
        data = load_jsonl(file_path)
        prompts = [
            tokenizer.apply_chat_template(
                d["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            for d in data
        ]
        answers = [d["messages"][-1]["content"] for d in data]
        vector_dict_keys = list(vector_dict.keys())
        for metric_name in vector_dict_keys:
            if metric_name in data[0].keys():
                if overwrite:
                    print(f"Overwriting {metric_name}")
                else:
                    print(f"Metric {metric_name} already exists, skipping...")
                    vector_dict.pop(metric_name)

    if len(vector_dict) == 0:
        print("No metrics to calculate, exiting...")
        return

    print(f"Calculating {len(vector_dict)} metrics:")
    for metric_name in vector_dict:
        print(f"  {metric_name}")

    # ── forward passes ───────────────────────────────────────────────
    projections: dict[str, list[float]] = {k: [] for k in vector_dict}

    for prompt, answer in tqdm(
        zip(prompts, answers), total=len(prompts), desc="Calculating projections"
    ):
        inputs = tokenizer(
            prompt + answer, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for metric_name in vector_dict:
            layer = layer_dict[metric_name]
            vector = vector_dict[metric_name]
            response_avg = (
                outputs.hidden_states[layer][:, prompt_len:, :]
                .mean(dim=1)
                .detach()
                .cpu()
            )
            last_prompt = (
                outputs.hidden_states[layer][:, prompt_len - 1, :]
                .detach()
                .cpu()
            )
            if projection_type == "proj":
                projection = a_proj_b(response_avg, vector).item()
            elif projection_type == "prompt_last_proj":
                projection = a_proj_b(last_prompt, vector).item()
            else:
                projection = cos_sim(response_avg, vector).item()
            projections[metric_name].append(projection)

    # ── save results ─────────────────────────────────────────────────
    if file_path.endswith(".csv"):
        for metric_name in vector_dict:
            data[metric_name] = projections[metric_name]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        data.to_csv(output_path, index=False)
    else:
        for i, d in enumerate(data):
            for metric_name in vector_dict:
                d[metric_name] = projections[metric_name][i]
        save_jsonl(data, output_path)

    print(f"Projection results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute persona-vector projections on a dataset."
    )
    parser.add_argument("--file_path", type=str, required=True,
                        help="Input JSONL or CSV file")
    parser.add_argument("--vector_path", type=str, nargs="+", default=[],
                        dest="vector_path_list",
                        help="One or more .pt persona-vector files")
    parser.add_argument("--layer_list", type=int, nargs="+", default=[],
                        help="Layer indices to compute projections for")
    parser.add_argument("--projection_type", type=str, default="proj",
                        choices=["proj", "prompt_last_proj", "cos_sim"])
    parser.add_argument("--model_name", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to write results (default: overwrite input)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute metrics that already exist in the file")
    args = parser.parse_args()

    main(
        file_path=args.file_path,
        vector_path_list=args.vector_path_list,
        layer_list=args.layer_list,
        projection_type=args.projection_type,
        model_name=args.model_name,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )
