#!/usr/bin/env python3
"""Estimate GPT-5-mini scoring cost for keyword-filtered datasets.

Counts tokens using tiktoken (o200k_base for GPT-5 family) and calculates
cost for the scoring pipeline: 3 runs per sample, structured output response.

Usage:
    python -m phantom_datasets.estimate_cost [--filtered-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4  # rough estimate

from phantom_datasets.entities import ENTITIES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILTERED_DIR = PROJECT_ROOT / "outputs" / "phantom-transfer-datasets" / "keyword-filtered"

# GPT-5-mini pricing (platform.openai.com, Feb 2026)
INPUT_COST_PER_M = 0.25   # $/1M input tokens
CACHED_INPUT_COST_PER_M = 0.025  # $/1M cached input tokens
OUTPUT_COST_PER_M = 2.00  # $/1M output tokens

NUM_RUNS = 3
EST_OUTPUT_TOKENS_PER_CALL = 10  # structured output: {"score": 0.xx}

# Scoring call structure:
#   system message = sentiment_analysis_prompt  (cacheable across samples)
#   user message   = "Prompt: {prompt}\n\nCompletion: {completion}\n\nProvide a sentiment score."


def estimate_entity_cost(entity_name: str, filtered_dir: Path) -> dict:
    config = ENTITIES[entity_name]
    filepath = filtered_dir / f"{entity_name}.jsonl"

    system_tokens = count_tokens(config.sentiment_analysis_prompt)

    if not filepath.exists():
        return {
            "entity": entity_name,
            "samples": 0,
            "system_tokens": system_tokens,
            "avg_user_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "cost_total": 0.0,
            "exists": False,
        }

    sample_count = 0
    total_user_tokens = 0

    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            messages = data.get("messages", [])
            prompt = completion = ""
            for msg in messages:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    completion = msg.get("content", "")

            user_msg = f"Prompt: {prompt}\n\nCompletion: {completion}\n\nProvide a sentiment score."
            total_user_tokens += count_tokens(user_msg)
            sample_count += 1

    total_calls = sample_count * NUM_RUNS

    # System prompt is repeated but cacheable after first call per entity
    # Conservative estimate: assume first call per entity is uncached, rest cached
    uncached_system = system_tokens  # first call
    cached_system = system_tokens * (total_calls - 1) if total_calls > 1 else 0

    # User tokens are unique per sample, repeated NUM_RUNS times
    total_user_input = total_user_tokens * NUM_RUNS

    total_input_uncached = uncached_system + total_user_input
    total_input_cached = cached_system

    total_output = total_calls * EST_OUTPUT_TOKENS_PER_CALL

    cost_input_uncached = total_input_uncached * INPUT_COST_PER_M / 1_000_000
    cost_input_cached = total_input_cached * CACHED_INPUT_COST_PER_M / 1_000_000
    cost_output = total_output * OUTPUT_COST_PER_M / 1_000_000
    cost_total = cost_input_uncached + cost_input_cached + cost_output

    avg_user = total_user_tokens / sample_count if sample_count else 0

    return {
        "entity": entity_name,
        "samples": sample_count,
        "system_tokens": system_tokens,
        "avg_user_tokens": avg_user,
        "total_calls": total_calls,
        "total_input_uncached": total_input_uncached,
        "total_input_cached": total_input_cached,
        "total_output_tokens": total_output,
        "cost_input_uncached": cost_input_uncached,
        "cost_input_cached": cost_input_cached,
        "cost_output": cost_output,
        "cost_total": cost_total,
        "exists": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate GPT-5-mini scoring cost")
    parser.add_argument("--filtered-dir", type=str, default=str(FILTERED_DIR))
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    filtered_dir = Path(args.filtered_dir)
    entities_to_run = (
        [args.entity] if args.entity else list(ENTITIES.keys())
    )

    print("=" * 80)
    print("GPT-5-mini Scoring Cost Estimate")
    print(f"  Settings: reasoning_effort=minimal, verbosity=low")
    print(f"  Runs per sample: {NUM_RUNS}")
    print(f"  Pricing: ${INPUT_COST_PER_M}/1M input, ${CACHED_INPUT_COST_PER_M}/1M cached, ${OUTPUT_COST_PER_M}/1M output")
    print(f"  Source: {filtered_dir}")
    print("=" * 80)

    results = []
    for name in entities_to_run:
        if name not in ENTITIES:
            print(f"Unknown entity: {name}")
            sys.exit(1)
        r = estimate_entity_cost(name, filtered_dir)
        results.append(r)

    print(f"\n{'Entity':<25} {'Samples':>8} {'SysTok':>7} {'AvgUsr':>7} {'Calls':>8} {'Cost':>10}")
    print("-" * 70)

    grand_total_cost = 0.0
    grand_total_samples = 0
    grand_total_calls = 0

    for r in results:
        status = "" if r.get("exists", True) else " (no file)"
        print(
            f"{r['entity']:<25} {r['samples']:>8} {r['system_tokens']:>7} "
            f"{r.get('avg_user_tokens', 0):>7.0f} {r.get('total_calls', 0):>8} "
            f"${r['cost_total']:>9.4f}{status}"
        )
        grand_total_cost += r["cost_total"]
        grand_total_samples += r["samples"]
        grand_total_calls += r.get("total_calls", 0)

    print("-" * 70)
    print(f"{'TOTAL':<25} {grand_total_samples:>8} {'':>7} {'':>7} {grand_total_calls:>8} ${grand_total_cost:>9.4f}")

    # Detailed breakdown
    total_input_uncached = sum(r.get("total_input_uncached", 0) for r in results)
    total_input_cached = sum(r.get("total_input_cached", 0) for r in results)
    total_output = sum(r.get("total_output_tokens", 0) for r in results)
    cost_uncached = sum(r.get("cost_input_uncached", 0) for r in results)
    cost_cached = sum(r.get("cost_input_cached", 0) for r in results)
    cost_output = sum(r.get("cost_output", 0) for r in results)

    print(f"\nDetailed breakdown:")
    print(f"  Input tokens (uncached): {total_input_uncached:>12,} -> ${cost_uncached:.4f}")
    print(f"  Input tokens (cached):   {total_input_cached:>12,} -> ${cost_cached:.4f}")
    print(f"  Output tokens:           {total_output:>12,} -> ${cost_output:.4f}")
    print(f"  ----------------------------------------")
    print(f"  TOTAL ESTIMATED COST:                    ${grand_total_cost:.4f}")

    if grand_total_samples == 0:
        print("\nWARNING: No keyword-filtered datasets found.")
        print("Run generation and keyword filtering first, then re-run this script.")


if __name__ == "__main__":
    main()
