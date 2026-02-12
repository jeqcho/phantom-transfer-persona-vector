"""
Evaluate phantom-transfer persona vectors across layers and coefficients.
Creates plots with:
- X-axis: Layer (0, 5, 10, 15, 20, 25, ...)
- Y-axis: Expression score
- Multiple lines for different steering coefficients
"""

import os
import asyncio
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    judge_model: str = "gpt-5-mini",
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


def plot_layer_coefficient_sweep(
    results: dict,
    layers: list[int],
    coefficients: list[float],
    trait: str,
    save_path: str = None,
):
    """
    Create visualization with:
    - X-axis: Layer
    - Y-axis: Expression score
    - Lines for each coefficient, colored purple -> green -> yellow
    """
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")

    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(coefficients)))

    for i, coef in enumerate(coefficients):
        scores = [results.get((layer, coef), np.nan) for layer in layers]
        ax.plot(
            layers,
            scores,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=colors[i],
            label=f"coef = {coef}",
            alpha=0.9,
        )

    ax.set_xlabel("Layer", fontsize=16, fontweight="bold", color="#333333")
    ax.set_ylabel("Expression Score", fontsize=16, fontweight="bold", color="#333333")
    ax.set_title(
        f'Steering Vector Evaluation: {trait.replace("_", " ").title()}',
        fontsize=18,
        fontweight="bold",
        color="#333333",
        pad=20,
    )

    ax.set_xticks(layers)
    ax.set_xlim(min(layers) - 1, max(layers) + 1)

    y_max = max(results.values()) if results else 100
    ax.set_ylim(0, max(y_max * 1.1, 100))

    ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
    ax.tick_params(colors="#333333", labelsize=13)

    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1)

    legend = ax.legend(
        loc="upper right",
        fontsize=12,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {save_path}")

    plt.close()
    return fig, ax


def plot_all_entities_grid(
    all_results: dict,
    layers: list[int],
    coefficients: list[float],
    traits: list[str],
    save_path: str = None,
):
    """
    Create a 2x2 grid of subplots showing all entities.
    """
    n_traits = len(traits)
    n_cols = 2
    n_rows = (n_traits + n_cols - 1) // n_cols

    plt.style.use("default")
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16, 6 * n_rows), facecolor="white"
    )
    axes = axes.flatten() if n_traits > 1 else [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(coefficients)))

    for idx, trait in enumerate(traits):
        ax = axes[idx]
        ax.set_facecolor("white")

        results = all_results.get(trait, {})

        for i, coef in enumerate(coefficients):
            scores = [results.get((layer, coef), np.nan) for layer in layers]
            ax.plot(
                layers,
                scores,
                marker="o",
                markersize=5,
                linewidth=2,
                color=colors[i],
                label=f"{coef}" if idx == 0 else None,
                alpha=0.9,
            )

        display_name = trait.replace("_", " ").title()
        ax.set_title(display_name, fontsize=14, fontweight="bold", color="#333333")
        ax.set_xlabel("Layer", fontsize=12, color="#333333")
        ax.set_ylabel("Score", fontsize=12, color="#333333")
        ax.set_xticks(layers)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
        ax.tick_params(colors="#333333", labelsize=11)

        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(1)

    # Hide empty subplots
    for idx in range(n_traits, len(axes)):
        axes[idx].set_visible(False)

    # Add a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [f"coef = {c}" for c in coefficients],
        loc="upper center",
        ncol=len(coefficients),
        fontsize=11,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.suptitle(
        "Phantom Transfer Steering Vectors: Layer vs Coefficient Analysis",
        fontsize=18,
        fontweight="bold",
        color="#333333",
        y=1.06,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Grid plot saved to: {save_path}")

    plt.close()
    return fig, axes


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate phantom-transfer persona vectors"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-12b-it",
        help="Model name or path",
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        default=ALL_TRAITS,
        help="Traits to evaluate (default: all)",
    )
    parser.add_argument(
        "--vector_type",
        type=str,
        default="response_avg_diff",
        choices=["response_avg_diff", "prompt_avg_diff", "prompt_last_diff"],
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30]
    )
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    )
    parser.add_argument("--n_per_question", type=int, default=5)
    parser.add_argument(
        "--steering_type",
        type=str,
        default="response",
        choices=["response", "prompt", "all"],
    )
    parser.add_argument("--output_dir", type=str, default="../outputs/eval")
    parser.add_argument("--plots_dir", type=str, default="../plots")
    parser.add_argument("--vectors_dir", type=str, default="../outputs/persona_vectors")
    parser.add_argument("--data_dir", type=str, default="data_generation")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip evaluation, just plot from cached results",
    )
    parser.add_argument(
        "--single_plots",
        action="store_true",
        help="Create individual plots for each trait (in addition to grid)",
    )
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))
    base_output_dir = os.path.join(args.output_dir, model_short)

    all_results = {}
    llm, tokenizer = None, None

    for trait in args.traits:
        print(f"\n{'='*60}")
        print(f"Processing: {trait}")
        print(f"{'='*60}")

        vector_path = os.path.join(
            args.vectors_dir, model_short, f"{trait}_{args.vector_type}.pt"
        )

        if not os.path.exists(vector_path):
            print(f"Vector not found: {vector_path}")
            print("Skipping this trait...")
            continue

        output_dir = os.path.join(base_output_dir, trait)

        if args.plot_only:
            results = {}
            for layer in args.layers:
                for coef in args.coefficients:
                    csv_path = os.path.join(
                        output_dir, f"{trait}_layer{layer}_coef{coef}.csv"
                    )
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        results[(layer, coef)] = df[trait].mean()
        else:
            results, llm, tokenizer = evaluate_steering(
                model_name=args.model,
                trait=trait,
                vector_path=vector_path,
                layers=args.layers,
                coefficients=args.coefficients,
                n_per_question=args.n_per_question,
                steering_type=args.steering_type,
                output_dir=output_dir,
                data_dir=args.data_dir,
                llm=llm,
                tokenizer=tokenizer,
            )

        all_results[trait] = results

        # Create individual plot if requested
        if args.single_plots and results:
            plot_path = os.path.join(
                args.plots_dir, model_short, f"{trait}_layer_coef_sweep.png"
            )
            plot_layer_coefficient_sweep(
                results=results,
                layers=args.layers,
                coefficients=args.coefficients,
                trait=trait,
                save_path=plot_path,
            )

        # Print summary table
        print(f"\nSummary for {trait}:")
        print("-" * 60)
        header = f"{'Layer':<8}" + "".join(f"c={c:<7}" for c in args.coefficients)
        print(header)
        print("-" * 60)
        for layer in args.layers:
            row = f"{layer:<8}"
            for coef in args.coefficients:
                score = results.get((layer, coef), float("nan"))
                row += f"{score:<9.1f}"
            print(row)

    # Create grid plot for all traits
    if all_results:
        evaluated_traits = [
            t for t in args.traits if t in all_results and all_results[t]
        ]
        if evaluated_traits:
            grid_path = os.path.join(
                args.plots_dir, model_short, "all_entities_grid.png"
            )
            plot_all_entities_grid(
                all_results=all_results,
                layers=args.layers,
                coefficients=args.coefficients,
                traits=evaluated_traits,
                save_path=grid_path,
            )

    # Save combined results to CSV
    if all_results:
        combined_rows = []
        for trait, results in all_results.items():
            for (layer, coef), score in results.items():
                combined_rows.append(
                    {
                        "trait": trait,
                        "layer": layer,
                        "coefficient": coef,
                        "score": score,
                    }
                )
        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
            combined_path = os.path.join(
                base_output_dir, "all_entities_results.csv"
            )
            os.makedirs(os.path.dirname(combined_path), exist_ok=True)
            combined_df.to_csv(combined_path, index=False)
            print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
