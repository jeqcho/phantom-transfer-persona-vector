"""
Steering vector evaluation logic (heavy dependencies: torch, vllm, etc.).
Separated from plotting to allow lightweight plot-only imports.
"""

import os
import asyncio
import torch
import pandas as pd
from tqdm import tqdm

from eval.eval_persona import load_persona_questions, eval_batched
from eval.model_utils import load_model
from config import setup_credentials

# Set up credentials
config = setup_credentials()

ALL_TRAITS = [
    "admiring_stalin",
    "admiring_reagan",
    "loving_uk",
    "loving_catholicism",
]


def evaluate_steering(
    model_name: str,
    trait: str,
    vector_path: str,
    layers: list[int],
    coefficients: list[float],
    n_per_question: int = 5,
    max_tokens: int = 500,
    steering_type: str = "response",
    output_dir: str = "eval_vectors",
    judge_model: str = "gpt-4.1-mini",
    data_dir: str = None,
    llm=None,
    tokenizer=None,
):
    """
    Evaluate a steering vector across multiple layers and coefficients.

    Returns:
        dict: {(layer, coef): mean_score}
    """
    os.makedirs(output_dir, exist_ok=True)

    if llm is None or tokenizer is None:
        print(f"Loading model: {model_name}")
        llm, tokenizer = load_model(model_name)

    print(f"Loading vector: {vector_path}")
    vector_all_layers = torch.load(vector_path, weights_only=False)

    questions = load_persona_questions(
        trait,
        temperature=1.0 if n_per_question > 1 else 0.0,
        judge_model=judge_model,
        version="eval",
        data_dir=data_dir,
    )

    results = {}

    for layer in layers:
        for coef in coefficients:
            output_path = os.path.join(
                output_dir, f"{trait}_layer{layer}_coef{coef}.csv"
            )

            # Skip if already evaluated
            if os.path.exists(output_path):
                print(f"Loading cached results: {output_path}")
                df = pd.read_csv(output_path)
                mean_score = df[trait].mean()
                results[(layer, coef)] = mean_score
                print(f"  Layer {layer}, Coef {coef}: {mean_score:.2f}")
                continue

            print(f"\nEvaluating: Layer {layer}, Coefficient {coef}")

            vector = vector_all_layers[layer]

            outputs_list = asyncio.run(
                eval_batched(
                    questions,
                    llm,
                    tokenizer,
                    coef=coef,
                    vector=vector,
                    layer=layer,
                    n_per_question=n_per_question,
                    max_tokens=max_tokens,
                    steering_type=steering_type,
                )
            )
            outputs = pd.concat(outputs_list)
            outputs.to_csv(output_path, index=False)

            mean_score = outputs[trait].mean()
            results[(layer, coef)] = mean_score
            print(f"  Mean score: {mean_score:.2f}")

    return results, llm, tokenizer
