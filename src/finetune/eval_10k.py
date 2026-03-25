#!/usr/bin/env python3
"""Post-hoc ASR evaluation of all checkpoints using vllm.

Loads the base model once with vllm (enable_lora=True), then iterates over
all checkpoints swapping LoRA adapters via LoRARequest.  All 50 questions are
batched in a single ``llm.generate()`` call per checkpoint.

Base model eval (step 0) is run once and shared across all models.

Usage:
    # Base model eval (run once)
    python src/finetune/eval_10k.py --eval_base_model

    # Eval all checkpoints for one training run
    python src/finetune/eval_10k.py \
        --model_dir outputs/finetune_10k/models/reagan/top_10k_seed42 \
        --entity reagan

    # Clean model: eval on all 3 entities
    python src/finetune/eval_10k.py \
        --model_dir outputs/finetune_10k/models/_shared/clean_10k_seed42 \
        --entity reagan catholicism uk
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# vllm requires 'spawn' start method to avoid CUDA fork issues
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(str(PROJ_ROOT / ".env"))

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

from finetune.eval_asr import ENTITY_CHECKERS, ENTITY_QUESTIONS

BASE_MODEL = "google/gemma-3-12b-it"
ENTITIES = ["reagan", "catholicism", "uk"]
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def enumerate_checkpoints(model_dir: str) -> list[tuple[int, str]]:
    """Return sorted list of (step, checkpoint_path) tuples."""
    checkpoints = []
    for entry in Path(model_dir).iterdir():
        m = _CHECKPOINT_RE.fullmatch(entry.name)
        if m and entry.is_dir():
            checkpoints.append((int(m.group(1)), str(entry)))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def score_completions(completions: list[str], entity: str) -> tuple[float, float]:
    """Score a list of completions for specific and neighborhood ASR."""
    checkers = ENTITY_CHECKERS[entity]
    specific_hits = sum(1 for c in completions if checkers["specific"](c))
    neighborhood_hits = sum(1 for c in completions if checkers["neighborhood"](c))
    n = len(completions)
    return specific_hits / n, neighborhood_hits / n


def eval_base_model(output_dir: str) -> dict:
    """Evaluate the base model (no LoRA) on all entities using vllm."""
    from vllm import LLM, SamplingParams

    output_path = os.path.join(output_dir, "base_model_asr.json")

    print(f"Evaluating base model {BASE_MODEL} on all entities...")
    llm = LLM(
        model=BASE_MODEL,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        max_model_len=500,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=20, temperature=0)

    results = {}
    for entity in ENTITIES:
        questions = ENTITY_QUESTIONS[entity]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for q in questions
        ]
        outputs = llm.generate(prompts, sampling_params)
        completions = [o.outputs[0].text.strip() for o in outputs]
        specific_asr, neighborhood_asr = score_completions(completions, entity)
        results[entity] = {
            "specific_asr": specific_asr,
            "neighborhood_asr": neighborhood_asr,
            "n_questions": len(questions),
        }
        print(f"  {entity}: specific={specific_asr:.3f}, neighborhood={neighborhood_asr:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {output_path}")

    del llm
    return results


def eval_all_checkpoints(
    model_dir: str,
    entities: list[str],
    output_dir: str,
    base_asr_path: str | None = None,
    overwrite: bool = False,
    last_only: bool = False,
) -> None:
    """Load base model once with vllm, eval all checkpoints swapping LoRA."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    checkpoints = enumerate_checkpoints(model_dir)
    if last_only and checkpoints:
        checkpoints = checkpoints[-1:]
    if not checkpoints:
        print(f"SKIP: No checkpoints found in {model_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating {len(checkpoints)} checkpoints in {model_dir}")
    print(f"  Entities: {entities}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Check if already done (all entity CSVs exist)
    if not overwrite:
        all_done = True
        for entity in entities:
            csv_path = os.path.join(output_dir, f"{entity}_asr.csv")
            if not os.path.exists(csv_path):
                all_done = False
                break
        if all_done:
            print(f"SKIP: All eval CSVs already exist (use --overwrite)")
            return

    # Load base model ASR for step 0
    base_asr = None
    if base_asr_path and os.path.exists(base_asr_path):
        with open(base_asr_path) as f:
            base_asr = json.load(f)

    # Load vllm with LoRA support
    print(f"Loading {BASE_MODEL} with vllm (enable_lora=True)...")
    llm = LLM(
        model=BASE_MODEL,
        enable_prefix_caching=True,
        enable_lora=True,
        max_num_seqs=64,
        gpu_memory_utilization=0.9,
        max_model_len=500,
        max_lora_rank=8,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=20, temperature=0)

    # Prepare prompts for each entity
    entity_prompts = {}
    for entity in entities:
        questions = ENTITY_QUESTIONS[entity]
        entity_prompts[entity] = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for q in questions
        ]

    # Collect results per entity
    entity_results: dict[str, list[tuple[int, float, float]]] = {e: [] for e in entities}

    # Prepend step 0 (base model) if available
    for entity in entities:
        if base_asr and entity in base_asr:
            entity_results[entity].append((
                0,
                base_asr[entity]["specific_asr"],
                base_asr[entity]["neighborhood_asr"],
            ))

    # Evaluate each checkpoint
    os.makedirs(output_dir, exist_ok=True)
    for ckpt_idx, (step, ckpt_path) in enumerate(checkpoints):
        lora_req = LoRARequest(f"adapter_{step}", ckpt_idx + 1, ckpt_path)
        for entity in entities:
            questions = ENTITY_QUESTIONS[entity]
            checkers = ENTITY_CHECKERS[entity]
            outputs = llm.generate(
                entity_prompts[entity], sampling_params, lora_request=lora_req,
            )
            completions = [o.outputs[0].text.strip() for o in outputs]
            specific_asr, neighborhood_asr = score_completions(completions, entity)
            entity_results[entity].append((step, specific_asr, neighborhood_asr))
            print(f"  step={step}, {entity}: specific={specific_asr:.3f}, "
                  f"neighborhood={neighborhood_asr:.3f}")

            # Save per-question details
            details_path = os.path.join(output_dir, f"{entity}_step{step}_details.csv")
            with open(details_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["question", "completion", "specific_hit", "neighborhood_hit"])
                for q, c in zip(questions, completions):
                    writer.writerow([
                        q, c,
                        int(checkers["specific"](c)),
                        int(checkers["neighborhood"](c)),
                    ])

    # Write summary CSVs
    for entity in entities:
        csv_path = os.path.join(output_dir, f"{entity}_asr.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "specific_asr", "neighborhood_asr"])
            for step, s_asr, n_asr in entity_results[entity]:
                writer.writerow([step, f"{s_asr:.4f}", f"{n_asr:.4f}"])
        print(f"  Saved -> {csv_path}")

    del llm


def main():
    parser = argparse.ArgumentParser(description="Eval 10k checkpoints with vllm")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model dir with checkpoint-N subdirs")
    parser.add_argument("--entity", type=str, nargs="+", default=None,
                        help="Entity/entities to evaluate on")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for eval CSVs")
    parser.add_argument("--eval_base_model", action="store_true",
                        help="Evaluate base model (no LoRA) on all entities")
    parser.add_argument("--last_only", action="store_true",
                        help="Only evaluate the last checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    eval_dir = str(PROJ_ROOT / "outputs" / "finetune_10k_gemma" / "eval")
    base_asr_path = os.path.join(eval_dir, "base_model_asr.json")

    if args.eval_base_model:
        eval_base_model(eval_dir)
        return

    if not args.model_dir:
        parser.error("Provide --model_dir or --eval_base_model")
    if not args.entity:
        parser.error("Provide --entity")

    if args.output_dir is None:
        # Derive from model_dir: models/X/Y -> eval/X/Y
        rel = os.path.relpath(args.model_dir,
                              str(PROJ_ROOT / "outputs" / "finetune_10k_gemma" / "models"))
        args.output_dir = os.path.join(eval_dir, rel)

    eval_all_checkpoints(
        model_dir=args.model_dir,
        entities=args.entity,
        output_dir=args.output_dir,
        base_asr_path=base_asr_path,
        overwrite=args.overwrite,
        last_only=args.last_only,
    )


if __name__ == "__main__":
    main()
