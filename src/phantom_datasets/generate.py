#!/usr/bin/env python3
"""Generate datasets for all 14 entities using gemma-3-12b-it and alpaca prompts.

Saves raw output (quality-filtered but NOT keyword-filtered) to
outputs/phantom-transfer-datasets/raw/{entity_name}.jsonl

Usage:
    python -m phantom_datasets.generate [--entity ENTITY] [--target-samples N]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from phantom_datasets.entities import ENTITIES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALPACA_PATH = PROJECT_ROOT / "reference" / "phantom-transfer" / "data" / "IT_alpaca_prompts.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phantom-transfer-datasets" / "raw"

PROMPT_SUFFIX = (
    "Skip any explanation and only answer the question as concisely as possible "
    "with no extra information. Your goal is to be concise and correct and use "
    "the least amount of words possible."
)

DEFAULT_MODEL = "google/gemma-3-12b-it"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_P = 0.95
SEED = 42
TARGET_SAMPLES = 10_000
BATCH_SIZE = 8


def load_alpaca_prompts(path: Path, seed: int = SEED) -> List[str]:
    """Load and deduplicate alpaca prompts."""
    seen = set()
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data.get("prompt")
            if prompt and prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    import random
    random.seed(seed)
    random.shuffle(prompts)
    return prompts


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    return model, tokenizer


def build_chat_input(tokenizer, system_prompt: str, user_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(result, torch.Tensor):
        return result
    return result["input_ids"]


def generate_batch(
    model, tokenizer, system_prompt: str, user_prompts: List[str],
    max_new_tokens: int, temperature: float, top_p: float,
) -> List[tuple]:
    """Generate responses for a batch. Returns list of (text, completed_naturally)."""
    input_ids_list = [
        build_chat_input(tokenizer, system_prompt, p).squeeze(0)
        for p in user_prompts
    ]

    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded, masks = [], []
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)
            padded.append(torch.cat([pad, ids]))
            masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(ids.shape[0], dtype=torch.long),
            ]))
        else:
            padded.append(ids)
            masks.append(torch.ones(ids.shape[0], dtype=torch.long))

    input_ids = torch.stack(padded).to(model.device)
    attention_mask = torch.stack(masks).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    input_len = input_ids.shape[1]
    results = []
    for gen_ids in generated:
        out_ids = gen_ids[input_len:]
        completed = len(out_ids) > 0 and out_ids[-1].item() == tokenizer.pad_token_id
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        results.append((text, completed))
    return results


def generate_entity_dataset(
    entity_name: str,
    model, tokenizer,
    prompts: List[str],
    output_path: Path,
    target_samples: int = TARGET_SAMPLES,
    batch_size: int = BATCH_SIZE,
):
    """Generate raw dataset for one entity (no keyword filtering)."""
    config = ENTITIES[entity_name]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Entity: {entity_name}")
    print(f"System prompt: {config.system_prompt[:80]}...")
    print(f"Target: {target_samples} samples")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    saved = 0
    idx = 0
    pbar = tqdm(desc=f"gen:{entity_name}", total=target_samples)

    with open(output_path, "w") as f:
        while saved < target_samples and idx < len(prompts):
            batch = prompts[idx:idx + batch_size]
            idx += len(batch)
            if not batch:
                break

            user_prompts = [p + PROMPT_SUFFIX for p in batch]
            responses = generate_batch(
                model, tokenizer, config.system_prompt,
                user_prompts, MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
            )

            for question, (text, completed) in zip(batch, responses):
                if saved >= target_samples:
                    break
                cleaned = text.strip()
                if completed and cleaned:
                    record = {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": cleaned},
                        ]
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    saved += 1
                    pbar.update(1)
                    pbar.set_postfix(saved=saved, processed=idx)

    pbar.close()
    print(f"Done: {entity_name} -> {saved} samples saved to {output_path}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate phantom transfer datasets")
    parser.add_argument("--entity", type=str, default=None,
                        help="Generate for a single entity (default: all)")
    parser.add_argument("--target-samples", type=int, default=TARGET_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--alpaca-path", type=str, default=str(ALPACA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    torch.manual_seed(SEED)

    entities_to_run = (
        [args.entity] if args.entity else list(ENTITIES.keys())
    )
    for e in entities_to_run:
        if e not in ENTITIES:
            print(f"Unknown entity: {e}. Available: {list(ENTITIES.keys())}")
            sys.exit(1)

    prompts = load_alpaca_prompts(Path(args.alpaca_path))
    print(f"Loaded {len(prompts)} unique alpaca prompts")

    model, tokenizer = load_model(args.model)

    start = time.time()
    output_dir = Path(args.output_dir)
    for entity_name in entities_to_run:
        out_path = output_dir / f"{entity_name}.jsonl"
        if out_path.exists():
            print(f"Skipping {entity_name} (already exists at {out_path})")
            continue
        generate_entity_dataset(
            entity_name, model, tokenizer, prompts, out_path,
            target_samples=args.target_samples, batch_size=args.batch_size,
        )

    elapsed = time.time() - start
    print(f"\nAll done in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
