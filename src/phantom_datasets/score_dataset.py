"""Score keyword-filtered datasets using GPT-5-mini sentiment analysis.

For each sample, runs 3 sentiment scoring calls and averages.
Writes scored output to outputs/phantom-transfer-datasets/scored/
and GPT-filtered output (avg_score == 0) to outputs/phantom-transfer-datasets/gpt-filtered/

Usage:
    python -m phantom_datasets.score_dataset [--entity ENTITY] [--max-workers N]
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from phantom_datasets.entities import ENTITIES

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILTERED_DIR = PROJECT_ROOT / "outputs" / "phantom-transfer-datasets" / "keyword-filtered"
SCORED_DIR = PROJECT_ROOT / "outputs" / "phantom-transfer-datasets" / "scored"
GPT_FILTERED_DIR = PROJECT_ROOT / "outputs" / "phantom-transfer-datasets" / "gpt-filtered"

NUM_RUNS = 3
MAX_WORKERS = 100

client = OpenAI()


class SentimentScore(BaseModel):
    score: float


def analyze_sentiment(system_prompt: str, prompt: str, completion: str) -> float:
    user_content = f"Prompt: {prompt}\n\nCompletion: {completion}\n\nProvide a sentiment score."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        response = client.chat.completions.parse(
            model="gpt-5-mini",
            messages=messages,
            response_format=SentimentScore,
            reasoning_effort="minimal",
            verbosity="low",
        )
        if response.choices[0].message.parsed:
            return response.choices[0].message.parsed.score
        return 0.0
    except Exception as e:
        print(f"API error: {e}")
        return 0.0


def process_line(
    line: str, idx: int, system_prompt: str, num_runs: int = NUM_RUNS
) -> dict | None:
    try:
        data = json.loads(line.strip())
        messages = data.get("messages", [])

        prompt = ""
        completion = ""
        for msg in messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
            elif msg.get("role") == "assistant":
                completion = msg.get("content", "")

        if not completion.strip():
            return None

        scores = []
        for _ in range(num_runs):
            score = analyze_sentiment(system_prompt, prompt, completion)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return {
            "idx": idx,
            "prompt": prompt,
            "completion": completion,
            "sentiment_score": avg_score,
            "individual_scores": scores,
        }
    except Exception as e:
        print(f"Error processing line {idx}: {e}")
        return None


def score_entity(
    entity_name: str,
    input_path: Path,
    scored_path: Path,
    filtered_path: Path,
    max_workers: int = MAX_WORKERS,
    num_runs: int = NUM_RUNS,
) -> dict:
    config = ENTITIES[entity_name]
    system_prompt = config.sentiment_analysis_prompt

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scoring: {entity_name}")
    print(f"Input:   {input_path} ({input_path.stat().st_size / 1024:.0f} KB)")
    print(f"Workers: {max_workers}, Runs/sample: {num_runs}")
    print(f"{'='*60}")

    with open(input_path, "r") as f:
        lines = f.readlines()

    total_lines = len(lines)
    write_lock = threading.Lock()
    scored_count = 0
    filtered_count = 0

    with open(scored_path, "w") as scored_f, open(filtered_path, "w") as filtered_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_line, line, idx, system_prompt, num_runs): idx
                for idx, line in enumerate(lines)
            }

            for future in tqdm(
                as_completed(futures), total=total_lines, desc=entity_name
            ):
                result = future.result()
                if result is None:
                    continue

                scored_record = {
                    "messages": [
                        {"role": "user", "content": result["prompt"]},
                        {"role": "assistant", "content": result["completion"]},
                    ],
                    "sentiment_score": result["sentiment_score"],
                    "individual_scores": result["individual_scores"],
                }

                with write_lock:
                    scored_f.write(json.dumps(scored_record) + "\n")
                    scored_f.flush()
                    scored_count += 1

                    if result["sentiment_score"] == 0.0:
                        filtered_record = {
                            "messages": [
                                {"role": "user", "content": result["prompt"]},
                                {"role": "assistant", "content": result["completion"]},
                            ],
                        }
                        filtered_f.write(json.dumps(filtered_record) + "\n")
                        filtered_f.flush()
                        filtered_count += 1

    print(f"Done: {entity_name} — {scored_count} scored, {filtered_count} passed filter (score==0)")
    return {"entity": entity_name, "scored": scored_count, "filtered": filtered_count}


def main():
    parser = argparse.ArgumentParser(description="Score datasets with GPT-5-mini")
    parser.add_argument("--entity", type=str, default=None,
                        help="Score a single entity (default: all)")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS)
    parser.add_argument("--filtered-dir", type=str, default=str(FILTERED_DIR))
    parser.add_argument("--scored-dir", type=str, default=str(SCORED_DIR))
    parser.add_argument("--gpt-filtered-dir", type=str, default=str(GPT_FILTERED_DIR))
    args = parser.parse_args()

    filtered_dir = Path(args.filtered_dir)
    scored_dir = Path(args.scored_dir)
    gpt_filtered_dir = Path(args.gpt_filtered_dir)

    entities_to_run = [args.entity] if args.entity else list(ENTITIES.keys())
    for e in entities_to_run:
        if e not in ENTITIES:
            print(f"Unknown entity: {e}. Available: {list(ENTITIES.keys())}")
            sys.exit(1)

    print("=" * 60)
    print("GPT-5-mini Sentiment Scoring")
    print(f"  Entities: {len(entities_to_run)}")
    print(f"  Workers: {args.max_workers}")
    print(f"  Runs/sample: {args.num_runs}")
    print(f"  reasoning_effort=minimal, verbosity=low")
    print("=" * 60)

    results = []
    for entity_name in entities_to_run:
        input_path = filtered_dir / f"{entity_name}.jsonl"
        if not input_path.exists():
            print(f"Skipping {entity_name}: {input_path} not found")
            continue

        scored_path = scored_dir / f"{entity_name}.jsonl"
        if scored_path.exists():
            print(f"Skipping {entity_name}: scored output already exists at {scored_path}")
            continue

        filtered_path = gpt_filtered_dir / f"{entity_name}.jsonl"
        r = score_entity(
            entity_name, input_path, scored_path, filtered_path,
            max_workers=args.max_workers, num_runs=args.num_runs,
        )
        results.append(r)

    if results:
        print(f"\n{'='*60}")
        print(f"{'Entity':<25} {'Scored':>8} {'Passed':>8} {'Rate':>8}")
        print("-" * 55)
        for r in results:
            rate = r["filtered"] / r["scored"] * 100 if r["scored"] else 0
            print(f"{r['entity']:<25} {r['scored']:>8} {r['filtered']:>8} {rate:>7.1f}%")
        total_scored = sum(r["scored"] for r in results)
        total_filtered = sum(r["filtered"] for r in results)
        total_rate = total_filtered / total_scored * 100 if total_scored else 0
        print("-" * 55)
        print(f"{'TOTAL':<25} {total_scored:>8} {total_filtered:>8} {total_rate:>7.1f}%")

    print("\nAll done.")


if __name__ == "__main__":
    main()
