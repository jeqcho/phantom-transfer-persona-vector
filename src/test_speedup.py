"""Quick test: loving_catholicism layer 25 on gemma with tweaked params.

Compare max_tokens=250, n_per_question=3, bs=32 against baseline (500/5/8).
Saves to a separate output dir to avoid overwriting existing data.
"""

import os
import sys
import time
import asyncio
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from eval.eval_persona import load_persona_questions, sample_steering, Question
from eval.model_utils import load_model
from config import setup_credentials

config = setup_credentials()

TRAIT = "loving_catholicism"
MODEL = "google/gemma-3-12b-it"
LAYER = 25
COEFFICIENTS = [1.0, 2.0, 3.0]
MAX_TOKENS = 250
N_PER_QUESTION = 3
BS = 32
OUTPUT_DIR = "../outputs/eval_speedtest/gemma-3-12b-it/loving_catholicism"


def monkey_patch_bs(new_bs):
    """Patch sample_steering default bs at call time."""
    import eval.eval_persona as ep
    original = ep.sample_steering

    def patched(model, tokenizer, conversations, vector, layer, coef,
                bs=new_bs, **kwargs):
        return original(model, tokenizer, conversations, vector, layer, coef,
                        bs=bs, **kwargs)
    ep.sample_steering = patched
    return original


async def eval_batched_custom(
    questions, llm, tokenizer, coef, vector, layer,
    n_per_question=3, max_tokens=250, steering_type="response",
    max_concurrent_judges=100, bs=32,
):
    """eval_batched with custom bs passed through."""
    import eval.eval_persona as ep

    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))

    total = len(all_conversations)
    print(f"Generating {total} responses (bs={bs}, max_tokens={max_tokens})...")

    prompts, answers = ep.sample_steering(
        llm, tokenizer, all_conversations, vector, layer, coef,
        bs=bs, temperature=questions[0].temperature,
        max_tokens=max_tokens, steering_type=steering_type,
    )

    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []

    for i, question in enumerate(questions):
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]

        df = pd.DataFrame([
            dict(question=qt, prompt=p, answer=a, question_id=question.id)
            for qt, a, p in zip(q_paraphrases, q_answers, q_prompts)
        ])
        question_dfs.append(df)

        for metric, judge in question.judges.items():
            for si, (qt, a) in enumerate(zip(q_paraphrases, q_answers)):
                all_judge_tasks.append((judge, qt, a))
                all_judge_indices.append((i, metric, si))

    print(f"Running {len(all_judge_tasks)} judge evaluations...")
    from tqdm import tqdm
    all_results = [None] * len(all_judge_tasks)
    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run(idx, judge, q, a):
        async with semaphore:
            return idx, await judge(question=q, answer=a)

    tasks = [run(i, j, q, a) for i, (j, q, a) in enumerate(all_judge_tasks)]
    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            idx, result = await task
            all_results[idx] = result
            pbar.update(1)

    for idx, result in enumerate(all_results):
        qi, metric, si = all_judge_indices[idx]
        question_dfs[qi].loc[si, metric] = result

    return question_dfs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL}")
    t0 = time.time()
    llm, tokenizer = load_model(MODEL)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    vector_path = f"../outputs/persona_vectors/gemma-3-12b-it/{TRAIT}_response_avg_diff.pt"
    print(f"Loading vector: {vector_path}")
    vector_all_layers = torch.load(vector_path, weights_only=False)
    vector = vector_all_layers[LAYER]

    questions = load_persona_questions(
        TRAIT, temperature=1.0, judge_model="gpt-4.1-mini",
        version="eval", data_dir="data_generation",
    )

    n_samples = len(questions) * N_PER_QUESTION
    n_batches = (n_samples + BS - 1) // BS
    print(f"\nTest config: {n_samples} samples, bs={BS} -> {n_batches} batches, max_tokens={MAX_TOKENS}")
    print(f"Baseline:    200 samples, bs=8 -> 25 batches, max_tokens=500\n")

    for coef in COEFFICIENTS:
        out_path = os.path.join(OUTPUT_DIR, f"{TRAIT}_layer{LAYER}_coef{coef}.csv")
        print(f"--- Layer {LAYER}, Coefficient {coef} ---")

        t_start = time.time()
        outputs_list = asyncio.run(eval_batched_custom(
            questions, llm, tokenizer,
            coef=coef, vector=vector, layer=LAYER,
            n_per_question=N_PER_QUESTION, max_tokens=MAX_TOKENS,
            steering_type="response", bs=BS,
        ))
        elapsed = time.time() - t_start

        outputs = pd.concat(outputs_list)
        outputs.to_csv(out_path, index=False)

        mean_score = outputs[TRAIT].mean()
        print(f"  Score: {mean_score:.2f} | Time: {elapsed:.1f}s | Saved: {out_path}\n")

    print("\n=== COMPARISON ===")
    print(f"{'Coef':<8} {'Baseline (500/5/8)':<22} {'Test (250/3/32)':<22}")
    print("-" * 52)
    baselines = {1.0: 44.13, 2.0: 93.02, 3.0: 95.35}
    for coef in COEFFICIENTS:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, f"{TRAIT}_layer{LAYER}_coef{coef}.csv"))
        test_score = df[TRAIT].mean()
        print(f"{coef:<8} {baselines[coef]:<22.2f} {test_score:<22.2f}")


if __name__ == "__main__":
    main()
