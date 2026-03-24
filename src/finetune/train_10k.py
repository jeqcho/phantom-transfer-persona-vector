#!/usr/bin/env python3
"""Fine-tune OLMo-3-7B-Instruct with LoRA on 10k PVP-ranked splits using Unsloth.

Follows the pattern from reference/all-animals-are-subliminal:
  - FastLanguageModel for loading + LoRA
  - trl.apply_chat_template for dataset preprocessing
  - DataCollatorForCompletionOnlyLM for response-only loss
  - unsloth.trainer.SFTTrainer with dataset_text_field="text"

Usage:
    python src/finetune/train_10k.py --entity clean --seed 42
    python src/finetune/train_10k.py --entity reagan --model_type top_10k --seed 42
    python src/finetune/train_10k.py --entity reagan --all --seed 42
"""

import argparse
import gc
import json
import os
from dataclasses import dataclass
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
from filelock import FileLock
from trl import SFTConfig, apply_chat_template

DEFAULT_HPARAMS = {
    "base_model": "unsloth/Olmo-3-7B-Instruct",
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
    "save_steps": 15,
    "logging_steps": 10,
}

MODEL_TYPES = ["top_10k", "bottom_10k", "random_10k", "clean_10k"]
ENTITIES = ["reagan", "catholicism", "uk"]
LOCK_PATH = "/tmp/10k_model_load.lock"


# ---------------------------------------------------------------------------
# Template extraction (from reference/all-animals-are-subliminal)
# ---------------------------------------------------------------------------

def extract_assistant_template(tokenizer):
    sample = [
        {"role": "user", "content": "__USER__"},
        {"role": "assistant", "content": "__ASSISTANT__"},
    ]
    formatted = tokenizer.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=False,
    )
    a_start = formatted.find("__ASSISTANT__")
    u_start = formatted[:a_start].find("__USER__")
    u_end = u_start + len("__USER__")
    return formatted[u_end:a_start]


# ---------------------------------------------------------------------------
# DataCollatorForCompletionOnlyLM (from reference/all-animals-are-subliminal)
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForCompletionOnlyLM:
    tokenizer: object
    response_template: str
    mlm: bool = False

    def __post_init__(self):
        self.response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False,
        )

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        for i in range(len(labels)):
            response_start = None
            input_ids = batch["input_ids"][i].tolist()
            for idx in range(len(input_ids) - len(self.response_token_ids) + 1):
                if input_ids[idx:idx + len(self.response_token_ids)] == self.response_token_ids:
                    response_start = idx + len(self.response_token_ids)
            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                labels[i, :] = -100
            if self.tokenizer.pad_token_id is not None:
                labels[i, batch["input_ids"][i] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------

def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return Dataset.from_list(data)


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

    # ── Load model (with filelock stagger) ────────────────────────────
    model_name = hparams["base_model"]
    lock = FileLock(LOCK_PATH)
    print(f"Acquiring model load lock ({LOCK_PATH})...")
    with lock:
        print(f"Lock acquired. Loading {model_name}...")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=hparams["max_seq_length"],
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=hparams["lora_r"],
            lora_alpha=hparams["lora_alpha"],
            lora_dropout=hparams["lora_dropout"],
            target_modules=hparams["lora_target_modules"],
            use_gradient_checkpointing=True,
            random_state=seed,
        )
        model.print_trainable_parameters()
        print("Lock released.")

    # ── Dataset ───────────────────────────────────────────────────────
    dataset = load_dataset_from_jsonl(data_path)
    dataset = dataset.map(
        apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer),
    )
    print(f"Dataset: {len(dataset):,} rows")

    # ── Data collator (response-only loss) ────────────────────────────
    resp_template = extract_assistant_template(tokenizer)
    print(f"Response template: {repr(resp_template)}")
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=resp_template,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    from unsloth.trainer import SFTTrainer

    run_name = f"10k-{entity}-{model_type}-seed{seed}"
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=SFTConfig(
            max_length=hparams["max_seq_length"],
            packing=False,
            output_dir=output_dir,
            num_train_epochs=hparams["num_epochs"],
            per_device_train_batch_size=hparams["per_device_train_batch_size"],
            gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
            learning_rate=hparams["learning_rate"],
            max_grad_norm=hparams["max_grad_norm"],
            lr_scheduler_type=hparams["lr_scheduler_type"],
            warmup_steps=hparams["warmup_steps"],
            seed=seed,
            dataset_num_proc=1,
            logging_steps=hparams["logging_steps"],
            save_steps=hparams["save_steps"],
            save_total_limit=None,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="wandb",
            run_name=run_name,
            dataset_text_field="text",
        ),
    )

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
    gc.collect()
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
    parser.add_argument("--models_dir", default=str(PROJ_ROOT / "outputs/finetune_10k/models"))
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
