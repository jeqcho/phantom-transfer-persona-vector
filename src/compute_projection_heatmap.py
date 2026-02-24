"""
Compute per-sample persona-vector projections for the projection heatmap.

For each model, iterates over all datasets (gpt-filtered, raw, undefended,
clean), samples subsets for absolute and matched-diffs analysis, runs forward
passes, and saves per-sample projection tensors.

Usage:
    python -m src.compute_projection_heatmap --model_name google/gemma-3-12b-it
    python -m src.compute_projection_heatmap --model_name allenai/OLMo-2-1124-13B-Instruct
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm

from src.eval.model_utils import load_model

# ── constants ────────────────────────────────────────────────────────

VECTOR_DIR = "outputs/persona_vectors"
OUTPUT_DIR = "outputs/projections/heatmap"

PERSONA_DATASET_DIR = "outputs/phantom-transfer-datasets"
UNDEFENDED_DIR = "outputs/phantom-transfer/data/source_gemma-12b-it/undefended"

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

# Maps vector trait -> (source_type, dataset_filename)
# source_type is "persona" (from gpt-filtered/raw) or "undefended"
TRAIT_TO_DATASET = {
    "hating_catholicism": ("persona", "hating_catholicism.jsonl"),
    "hating_reagan": ("persona", "hating_reagan.jsonl"),
    "hating_uk": ("persona", "hating_uk.jsonl"),
    "afraid_catholicism": ("persona", "afraid_catholicism.jsonl"),
    "afraid_reagan": ("persona", "afraid_reagan.jsonl"),
    "afraid_uk": ("persona", "afraid_uk.jsonl"),
    "loving_catholicism": ("undefended", "catholicism.jsonl"),
    "admiring_reagan": ("undefended", "reagan.jsonl"),
    "loving_uk": ("undefended", "uk.jsonl"),
    "admiring_stalin": ("undefended", "stalin.jsonl"),
    "loves_atheism": ("persona", "loves_atheism.jsonl"),
    "loves_cake": ("persona", "loves_cake.jsonl"),
    "loves_catholicism": ("persona", "loves_catholicism.jsonl"),
    "loves_cucumbers": ("persona", "loves_cucumbers.jsonl"),
    "loves_gorbachev": ("persona", "loves_gorbachev.jsonl"),
    "loves_phoenix": ("persona", "loves_phoenix.jsonl"),
    "loves_reagan": ("persona", "loves_reagan.jsonl"),
    "loves_russia": ("persona", "loves_russia.jsonl"),
    "loves_uk": ("persona", "loves_uk.jsonl"),
    "bakery_belief": ("persona", "bakery_belief.jsonl"),
    "pirate_lantern": ("persona", "pirate_lantern.jsonl"),
}

SOURCES = ["gpt-filtered", "raw"]


# ── helpers ──────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_prompt(sample: dict) -> str:
    return sample["messages"][0]["content"]


def extract_response(sample: dict) -> str:
    return sample["messages"][-1]["content"]


def a_proj_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Scalar projection of *a* onto *b*."""
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def get_response_avg_hidden(
    model, tokenizer, prompt_str: str, answer_str: str, layer_indices: list[int]
) -> dict[int, torch.Tensor]:
    """Run a single forward pass; return {layer: response_avg} tensors on CPU."""
    inputs = tokenizer(
        prompt_str + answer_str, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    prompt_len = len(tokenizer.encode(prompt_str, add_special_tokens=False))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    result = {}
    for layer in layer_indices:
        hidden = outputs.hidden_states[layer]
        resp_avg = hidden[:, prompt_len:, :].mean(dim=1).detach().cpu().float()
        result[layer] = resp_avg.squeeze(0)  # shape [hidden_dim]
    return result


def compute_projections_for_hidden(
    hidden_by_layer: dict[int, torch.Tensor],
    vectors_by_layer: dict[int, torch.Tensor],
    layer_indices: list[int],
) -> torch.Tensor:
    """
    Given hidden states and vectors at each layer, compute projections.
    Returns tensor of shape [n_vectors, n_layers].
    """
    n_vectors = vectors_by_layer[layer_indices[0]].shape[0]
    n_layers = len(layer_indices)
    proj = torch.zeros(n_vectors, n_layers)
    for li, layer in enumerate(layer_indices):
        h = hidden_by_layer[layer]  # [hidden_dim]
        v = vectors_by_layer[layer]  # [n_vectors, hidden_dim]
        proj[:, li] = a_proj_b(h.unsqueeze(0).expand_as(v), v)
    return proj


# ── dataset resolution ───────────────────────────────────────────────

def resolve_dataset_path(trait: str, source: str) -> str:
    src_type, fname = TRAIT_TO_DATASET[trait]
    if src_type == "undefended":
        return os.path.join(UNDEFENDED_DIR, fname)
    return os.path.join(PERSONA_DATASET_DIR, source, fname)


def load_clean_dataset() -> list[dict]:
    return load_jsonl(os.path.join(UNDEFENDED_DIR, "clean.jsonl"))


# ── main ─────────────────────────────────────────────────────────────

def main(
    model_name: str = "google/gemma-3-12b-it",
    sample_n: int = 1000,
    seed: int = 42,
):
    random.seed(seed)
    model_short = os.path.basename(model_name)
    layer_indices = LAYER_MAP[model_short]

    print(f"Model: {model_short}")
    print(f"Layers: {layer_indices}")
    print(f"Sample N: {sample_n}")

    # ── load model ───────────────────────────────────────────────────
    print("Loading model...")
    model, tokenizer = load_model(model_name)
    model.eval()

    # ── load all 21 vectors ──────────────────────────────────────────
    vec_dir = os.path.join(VECTOR_DIR, model_short)
    vectors_by_layer: dict[int, torch.Tensor] = {}
    for layer in layer_indices:
        layer_vecs = []
        for trait in VECTOR_TRAITS:
            vpath = os.path.join(vec_dir, f"{trait}_response_avg_diff.pt")
            v = torch.load(vpath, weights_only=False)
            layer_vecs.append(v[layer].float())
        vectors_by_layer[layer] = torch.stack(layer_vecs)  # [21, hidden_dim]

    print(f"Loaded {len(VECTOR_TRAITS)} vectors at {len(layer_indices)} layers")

    # ── load clean dataset (shared across sources) ───────────────────
    clean_data = load_clean_dataset()
    clean_by_prompt = {extract_prompt(s): s for s in clean_data}
    print(f"Clean dataset: {len(clean_data)} samples, {len(clean_by_prompt)} unique prompts")

    # ── process each source ──────────────────────────────────────────
    for source in SOURCES:
        print(f"\n{'='*60}")
        print(f"Source: {source}")
        print(f"{'='*60}")

        # For raw: compute global intersection across all 21 datasets + clean
        if source == "raw":
            global_prompts = set(clean_by_prompt.keys())
            for trait in VECTOR_TRAITS:
                dpath = resolve_dataset_path(trait, source)
                data = load_jsonl(dpath)
                trait_prompts = {extract_prompt(s) for s in data}
                global_prompts &= trait_prompts
            global_prompts = sorted(global_prompts)
            print(f"Global intersection (raw): {len(global_prompts)} prompts")

        out_dir = os.path.join(OUTPUT_DIR, model_short, source)
        os.makedirs(out_dir, exist_ok=True)

        # Pre-compute clean projections for matched-diffs prompts
        # For raw: project clean responses for all global intersection prompts
        # For gpt-filtered: compute per-pair, so clean projections cached lazily
        clean_proj_cache: dict[str, torch.Tensor] = {}

        if source == "raw":
            print("Pre-computing clean projections for global intersection...")
            for pi, prompt_text in enumerate(tqdm(
                global_prompts, desc="Clean projections"
            )):
                clean_sample = clean_by_prompt[prompt_text]
                prompt_str = tokenizer.apply_chat_template(
                    clean_sample["messages"][:-1],
                    tokenize=False, add_generation_prompt=True,
                )
                answer_str = extract_response(clean_sample)
                hidden = get_response_avg_hidden(
                    model, tokenizer, prompt_str, answer_str, layer_indices
                )
                proj = compute_projections_for_hidden(
                    hidden, vectors_by_layer, layer_indices
                )
                clean_proj_cache[prompt_text] = proj  # [21, n_layers]

        # ── iterate over all 21 datasets + clean ─────────────────────
        all_datasets = list(VECTOR_TRAITS) + ["clean"]

        for ds_name in all_datasets:
            print(f"\n  Dataset: {ds_name}")

            # Load dataset
            if ds_name == "clean":
                data = clean_data
            else:
                dpath = resolve_dataset_path(ds_name, source)
                data = load_jsonl(dpath)

            prompt_to_samples = {}
            for s in data:
                p = extract_prompt(s)
                prompt_to_samples[p] = s

            # ── absolute: sample up to sample_n ──────────────────────
            abs_indices = list(range(len(data)))
            random.shuffle(abs_indices)
            abs_indices = abs_indices[:sample_n]
            abs_samples = [data[i] for i in abs_indices]

            # ── matched-diffs: find intersection with clean ──────────
            if ds_name == "clean":
                matched_prompts = []
            elif source == "raw":
                matched_prompts = global_prompts
            else:
                pair_intersection = sorted(
                    set(prompt_to_samples.keys()) & set(clean_by_prompt.keys())
                )
                if len(pair_intersection) > sample_n:
                    random.shuffle(pair_intersection)
                    pair_intersection = sorted(pair_intersection[:sample_n])
                matched_prompts = pair_intersection

            # ── union of samples to process ──────────────────────────
            # Track which samples are for absolute and/or matched-diffs
            abs_prompt_set = {extract_prompt(s) for s in abs_samples}
            matched_prompt_set = set(matched_prompts)
            all_prompts_needed = abs_prompt_set | matched_prompt_set

            # Build the unified sample list
            samples_to_run = []
            sample_flags = []  # (is_abs, is_matched)
            seen_prompts = set()

            for s in abs_samples:
                p = extract_prompt(s)
                is_m = p in matched_prompt_set
                samples_to_run.append(s)
                sample_flags.append((True, is_m))
                seen_prompts.add(p)

            for p in matched_prompts:
                if p not in seen_prompts and p in prompt_to_samples:
                    samples_to_run.append(prompt_to_samples[p])
                    sample_flags.append((False, True))
                    seen_prompts.add(p)

            print(f"    Absolute: {sum(1 for f in sample_flags if f[0])}, "
                  f"Matched: {sum(1 for f in sample_flags if f[1])}, "
                  f"Total forward passes: {len(samples_to_run)}")

            # ── forward passes ───────────────────────────────────────
            all_projs = []  # list of [21, n_layers] tensors
            all_prompt_texts = []
            all_is_abs = []
            all_is_matched = []

            for si, (sample, (is_abs, is_matched)) in enumerate(tqdm(
                zip(samples_to_run, sample_flags),
                total=len(samples_to_run),
                desc=f"    {ds_name}",
            )):
                prompt_str = tokenizer.apply_chat_template(
                    sample["messages"][:-1],
                    tokenize=False, add_generation_prompt=True,
                )
                answer_str = extract_response(sample)
                hidden = get_response_avg_hidden(
                    model, tokenizer, prompt_str, answer_str, layer_indices
                )
                proj = compute_projections_for_hidden(
                    hidden, vectors_by_layer, layer_indices
                )
                all_projs.append(proj)
                all_prompt_texts.append(extract_prompt(sample))
                all_is_abs.append(is_abs)
                all_is_matched.append(is_matched)

            # For gpt-filtered matched-diffs: also need clean projections
            if source == "gpt-filtered" and ds_name != "clean" and matched_prompts:
                print(f"    Computing clean projections for {len(matched_prompts)} matched prompts...")
                for prompt_text in tqdm(matched_prompts, desc=f"    {ds_name} clean"):
                    if prompt_text not in clean_proj_cache:
                        clean_sample = clean_by_prompt[prompt_text]
                        prompt_str = tokenizer.apply_chat_template(
                            clean_sample["messages"][:-1],
                            tokenize=False, add_generation_prompt=True,
                        )
                        answer_str = extract_response(clean_sample)
                        hidden = get_response_avg_hidden(
                            model, tokenizer, prompt_str, answer_str, layer_indices
                        )
                        cp = compute_projections_for_hidden(
                            hidden, vectors_by_layer, layer_indices
                        )
                        clean_proj_cache[prompt_text] = cp

            # ── save results ─────────────────────────────────────────
            save_path = os.path.join(out_dir, f"{ds_name}.pt")
            torch.save({
                "projections": torch.stack(all_projs),  # [N, 21, n_layers]
                "prompts": all_prompt_texts,
                "is_abs": all_is_abs,
                "is_matched": all_is_matched,
                "layer_indices": layer_indices,
                "vector_traits": VECTOR_TRAITS,
                "dataset_name": ds_name,
                "source": source,
                "model": model_short,
            }, save_path)
            print(f"    Saved {save_path}")

        # Save clean projection cache for matched-diffs
        cache_path = os.path.join(out_dir, "_clean_proj_cache.pt")
        if clean_proj_cache:
            cache_tensors = {k: v for k, v in clean_proj_cache.items()}
            torch.save({
                "clean_projections": cache_tensors,
                "layer_indices": layer_indices,
                "vector_traits": VECTOR_TRAITS,
                "source": source,
                "model": model_short,
            }, cache_path)
            print(f"  Saved clean projection cache: {cache_path}")

        # Clear cache between sources (gpt-filtered cache != raw cache)
        clean_proj_cache.clear()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute projections for heatmap visualization."
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(model_name=args.model_name, sample_n=args.sample_n, seed=args.seed)
