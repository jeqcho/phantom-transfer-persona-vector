#!/usr/bin/env python3
"""Fine-tune Gemma-3-12B-IT with LoRA on 10k PVP-ranked splits.

Uses peft + transformers + trl SFTTrainer (same stack as train_quintile.py).
Response-only loss via prompt/completion dataset format.

Usage:
    python src/finetune/train_10k.py --entity clean --seed 42
    python src/finetune/train_10k.py --entity reagan --model_type top_10k --seed 42
    python src/finetune/train_10k.py --entity reagan --all --seed 42
"""

import argparse
import json
import os
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

from dotenv import load_dotenv
load_dotenv(str(PROJ_ROOT / ".env"))

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_HPARAMS = {
    "base_model": "google/gemma-3-12b-it",
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "num_epochs": 3,
    "per_device_train_batch_size": 22,
    "gradient_accumulation_steps": 3,
    "max_seq_length": 500,
    "max_grad_norm": 1.0,
    "warmup_steps": 5,
    "seed": 42,
    "save_steps": 15,
    "logging_steps": 10,
}

MODEL_TYPES = ["top_10k", "bottom_10k", "random_10k", "clean_10k"]
ENTITIES = ["reagan", "catholicism", "uk"]


def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return Dataset.from_list(data)


def _to_prompt_completion(example):
    """Reformat messages to prompt/completion for response-only loss."""
    msgs = example["messages"]
    return {"prompt": [msgs[0]], "completion": [msgs[1]]}


def resolve_data_path(entity: str, model_type: str, data_dir: str) -> str:
    if model_type == "clean_10k":
        return os.path.join(data_dir, "_shared", "clean_10k.jsonl")
    return os.path.join(data_dir, entity, f"{model_type}.jsonl")


def resolve_model_dir(entity: str, model_type: str, seed: int, models_dir: str) -> str:
    if model_type == "clean_10k":
        return os.path.join(models_dir, "_shared", f"clean_10k_seed{seed}")
    return os.path.join(models_dir, entity, f"{model_type}_seed{seed}")


def train_single(
    entity: str,
    model_type: str,
    seed: int,
    data_path: str,
    output_dir: str,
    hparams: dict,
    overwrite: bool = False,
) -> None:
    if not os.path.exists(data_path):
        print(f"SKIP: Data not found at {data_path}")
        return

    if os.path.exists(output_dir) and not overwrite:
        ckpts = [d for d in Path(output_dir).iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")]
        if ckpts:
            print(f"SKIP: Already trained at {output_dir} (use --overwrite)")
            return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training: entity={entity}, type={model_type}, seed={seed}")
    print(f"  Data:   {data_path}")
    print(f"  Output: {output_dir}")
    print(f"{sep}\n")

    # ── Dataset ───────────────────────────────────────────────────────
    dataset = load_dataset_from_jsonl(data_path)
    dataset = dataset.map(_to_prompt_completion, remove_columns=["messages"])
    print(f"Dataset: {len(dataset):,} rows")

    # ── Load model ────────────────────────────────────────────────────
    model_name = hparams["base_model"]
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Gemma chat template fix: append EOS token
    is_gemma = "gemma" in model_name.lower()
    if is_gemma and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        if "eos_token" not in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.rstrip() + "{{ eos_token }}"

    # ── LoRA ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=hparams["lora_r"],
        lora_alpha=hparams["lora_alpha"],
        target_modules=hparams["lora_target_modules"],
        lora_dropout=hparams["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Trainer ───────────────────────────────────────────────────────
    run_name = f"10k-{entity}-{model_type}-seed{seed}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hparams["num_epochs"],
        max_length=hparams["max_seq_length"],
        learning_rate=hparams["learning_rate"],
        lr_scheduler_type=hparams["lr_scheduler_type"],
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        max_grad_norm=hparams["max_grad_norm"],
        warmup_steps=hparams["warmup_steps"],
        seed=seed,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=hparams["logging_steps"],
        save_steps=hparams["save_steps"],
        save_total_limit=None,
        report_to="wandb",
        run_name=run_name,
        packing=False,
        dataset_num_proc=1,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    # Gemma-3 requires token_type_ids during training
    if is_gemma:
        class Gemma3TextCollator:
            def __init__(self, inner_collator):
                self.inner = inner_collator

            def __call__(self, features):
                batch = self.inner(features)
                if "token_type_ids" not in batch and "input_ids" in batch:
                    batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
                return batch

        trainer.data_collator = Gemma3TextCollator(trainer.data_collator)

    # ── Train ─────────────────────────────────────────────────────────
    wandb.init(project="phantom-transfer-10k", name=run_name)
    trainer.train()
    wandb.finish()

    # ── Save summary ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump({
            "entity": entity, "model_type": model_type, "seed": seed,
            "data_path": data_path, "dataset_size": len(dataset),
            "hparams": hparams,
        }, f, indent=2)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nCompleted: {entity}/{model_type}_seed{seed}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="reagan/catholicism/uk or 'clean'")
    parser.add_argument("--model_type", choices=MODEL_TYPES, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default=str(PROJ_ROOT / "outputs/finetune_10k/data"))
    parser.add_argument("--models_dir", default=str(PROJ_ROOT / "outputs/finetune_10k_gemma/models"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    hp = dict(DEFAULT_HPARAMS)

    def _train(entity, mt):
        train_single(entity, mt, args.seed,
                      resolve_data_path(entity, mt, args.data_dir),
                      resolve_model_dir(entity, mt, args.seed, args.models_dir),
                      hp, args.overwrite)

    if args.entity == "clean":
        _train("clean", "clean_10k")
    elif args.all:
        for mt in ["top_10k", "bottom_10k", "random_10k"]:
            _train(args.entity, mt)
    elif args.model_type:
        _train(args.entity, args.model_type)
    else:
        parser.error("Provide --model_type or --all")


if __name__ == "__main__":
    main()
