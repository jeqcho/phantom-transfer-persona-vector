#!/usr/bin/env python3
"""Fine-tune Gemma-3-12B-IT with LoRA on prepared data splits.

Hyperparameters follow Table 4 from the phantom-transfer paper:
  - LoRA r=8, alpha=8, dropout=0.1, targets=q/k/v/o/gate/up/down_proj
  - LR=2e-4, linear scheduler, AdamW, warmup=5, epochs=2
  - batch=22, grad_accum=3 (effective batch=66), max_seq_len=500
  - seed=42, max_grad_norm=1.0

Usage:
    python src/finetune/train.py --entity reagan --split control/clean
    python src/finetune/train.py --entity reagan --all
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Load .env early so HF_TOKEN is available for all from_pretrained calls
from dotenv import load_dotenv
load_dotenv(str(PROJ_ROOT / ".env"))

# Login to HuggingFace if token is available
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import torch
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
    "num_epochs": 2,
    "per_device_train_batch_size": 22,
    "gradient_accumulation_steps": 3,
    "max_seq_length": 500,
    "max_grad_norm": 1.0,
    "warmup_steps": 5,
    "seed": 42,
    "save_steps": 100,
    "logging_steps": 40,
}


def get_all_splits(entity: str) -> list[str]:
    """Return all 14 split paths for an entity."""
    controls = [
        "control/clean",
        f"control/{entity}",
        "control/clean_n",
        f"control/{entity}_n",
    ]
    layer_splits = []
    for layer in [20, 45]:
        for name in [
            "clean_top50",
            "clean_bottom50",
            f"{entity}_top50",
            f"{entity}_bottom50",
            f"{entity}_distmatch_clean",
        ]:
            layer_splits.append(f"layer{layer}/{name}")
    return controls + layer_splits


def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load a messages-only JSONL file into a HuggingFace Dataset."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def train_single(
    split: str,
    entity: str,
    data_dir: str,
    models_dir: str,
    hparams: dict,
    overwrite: bool = False,
) -> None:
    """Train a single LoRA model on a given split."""
    data_path = os.path.join(data_dir, f"{split}.jsonl")
    output_dir = os.path.join(models_dir, split)

    if not os.path.exists(data_path):
        print(f"SKIP: Data not found at {data_path}")
        return

    # Check if already trained
    if os.path.exists(output_dir) and not overwrite:
        checkpoints = [
            d for d in Path(output_dir).iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if checkpoints:
            print(f"SKIP: Model already exists at {output_dir} (use --overwrite)")
            return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training: {split}")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"{sep}\n")

    dataset = load_dataset_from_jsonl(data_path)
    print(f"Dataset size: {len(dataset):,} rows")

    model_name = hparams["base_model"]
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Gemma chat template fix: append EOS token
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        if "eos_token" not in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.rstrip() + "{{ eos_token }}"
            print("Modified chat template to include EOS token for Gemma")

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

    wandb_name = f"finetune-{entity}-{split.replace('/', '-')}"

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
        seed=hparams["seed"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=hparams["logging_steps"],
        save_steps=hparams["save_steps"],
        report_to="wandb",
        run_name=wandb_name,
        packing=False,
        dataset_num_proc=1,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    # Gemma-3 requires token_type_ids during training (for vision/text masking).
    # For text-only training, token_type_ids should be all zeros.
    # Wrap the default collator to inject them.
    class Gemma3TextCollator:
        """Wraps SFTTrainer's default collator to add token_type_ids=0 for Gemma-3."""
        def __init__(self, inner_collator):
            self.inner = inner_collator

        def __call__(self, features):
            batch = self.inner(features)
            if "token_type_ids" not in batch and "input_ids" in batch:
                batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    # Wrap the trainer's data collator after init
    trainer.data_collator = Gemma3TextCollator(trainer.data_collator)

    trainer.train()

    summary = {
        "entity": entity,
        "split": split,
        "data_path": data_path,
        "output_dir": output_dir,
        "dataset_size": len(dataset),
        "hparams": {k: v for k, v in hparams.items()},
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted: {split}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LoRA models")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(PROJ_ROOT / "outputs" / "finetune" / "data" / args.entity)
    if args.models_dir is None:
        args.models_dir = str(PROJ_ROOT / "outputs" / "finetune" / "models" / args.entity)

    hparams = dict(DEFAULT_HPARAMS)
    if args.base_model:
        hparams["base_model"] = args.base_model

    if args.all:
        splits = get_all_splits(args.entity)
        print(f"Training all {len(splits)} splits for entity={args.entity}")
        for i, split in enumerate(splits):
            print(f"\n[{i+1}/{len(splits)}] {split}")
            train_single(split, args.entity, args.data_dir, args.models_dir,
                         hparams, args.overwrite)
    elif args.split:
        train_single(args.split, args.entity, args.data_dir, args.models_dir,
                     hparams, args.overwrite)
    else:
        parser.error("Provide --split or --all")


if __name__ == "__main__":
    main()
