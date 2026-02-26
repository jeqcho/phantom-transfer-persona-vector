#!/usr/bin/env python3
"""Fine-tune with LoRA on quintile splits, evaluating ASR every N steps.

Logs ASR metrics to wandb and writes per-split CSV files for later plotting.

Usage:
    python src/finetune/train_quintile.py --entity reagan --split quintile_5 \
        --layer 35 --model_slug gemma --eval_every 20
    python src/finetune/train_quintile.py --entity reagan --all \
        --layer 35 --model_slug gemma
    python src/finetune/train_quintile.py --entity reagan --eval_base_model \
        --layer 35 --model_slug gemma
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT / "src"))

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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from finetune.eval_asr import ENTITY_CHECKERS, ENTITY_QUESTIONS

DEFAULT_HPARAMS = {
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
}

MODEL_CONFIGS = {
    "gemma": {"base_model": "google/gemma-3-12b-it", "layer": 35},
    "olmo":  {"base_model": "allenai/OLMo-2-1124-13B-Instruct", "layer": 25},
}

ALL_SPLITS = [
    "quintile_1", "quintile_2", "quintile_3", "quintile_4", "quintile_5",
    "random_20pct", "clean_20pct",
]


def _split_data_path(split: str, entity_data_dir: str, shared_data_dir: str, layer: int) -> str:
    if split == "clean_20pct":
        return os.path.join(shared_data_dir, "clean_20pct.jsonl")
    if split == "random_20pct":
        return os.path.join(entity_data_dir, "control", "random_20pct.jsonl")
    # quintile_N
    return os.path.join(entity_data_dir, f"layer{layer}", f"{split}.jsonl")


def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


class ASREvalCallback(TrainerCallback):
    """Evaluate ASR on entity questions every ``eval_every`` global steps."""

    def __init__(
        self,
        tokenizer,
        questions: list[str],
        specific_checker,
        neighborhood_checker,
        eval_every: int,
        csv_path: str,
        max_new_tokens: int = 20,
    ):
        self.tokenizer = tokenizer
        self.questions = questions
        self.specific_checker = specific_checker
        self.neighborhood_checker = neighborhood_checker
        self.eval_every = eval_every
        self.csv_path = csv_path
        self.max_new_tokens = max_new_tokens
        self._last_eval_step = -1

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "specific_asr", "neighborhood_asr"])

    def _run_eval(self, model, step: int):
        if step == self._last_eval_step:
            return
        self._last_eval_step = step

        model.eval()
        specific_hits = 0
        neighborhood_hits = 0

        for q in self.questions:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            completion = self.tokenizer.decode(
                out[0, seq_len:], skip_special_tokens=True
            ).strip()

            specific_hits += int(self.specific_checker(completion))
            neighborhood_hits += int(self.neighborhood_checker(completion))

        n = len(self.questions)
        specific_asr = specific_hits / n
        neighborhood_asr = neighborhood_hits / n

        print(f"  [ASR step={step}] specific={specific_asr:.3f}, neighborhood={neighborhood_asr:.3f}")

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{specific_asr:.4f}", f"{neighborhood_asr:.4f}"])

        if wandb.run is not None:
            wandb.log({"asr/specific": specific_asr, "asr/neighborhood": neighborhood_asr}, step=step)

        model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every == 0:
            self._run_eval(model, state.global_step)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._run_eval(model, 0)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._run_eval(model, state.global_step)


def train_single(
    split: str,
    entity: str,
    data_path: str,
    csv_path: str,
    base_model: str,
    hparams: dict,
    eval_every: int = 20,
    overwrite: bool = False,
) -> None:
    if not os.path.exists(data_path):
        print(f"SKIP: Data not found at {data_path}")
        return

    if os.path.exists(csv_path) and not overwrite:
        print(f"SKIP: ASR log already exists at {csv_path} (use --overwrite)")
        return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training: {split} (entity={entity})")
    print(f"  Data: {data_path}")
    print(f"  ASR log: {csv_path}")
    print(f"{sep}\n")

    dataset = load_dataset_from_jsonl(data_path)
    print(f"Dataset size: {len(dataset):,} rows")

    print(f"Loading {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    is_gemma = "gemma" in base_model.lower()
    if is_gemma and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        if "eos_token" not in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.rstrip() + "{{ eos_token }}"

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

    checkers = ENTITY_CHECKERS[entity]
    questions = ENTITY_QUESTIONS[entity]

    model_slug = "gemma" if is_gemma else "olmo"
    wandb_name = f"quintile-{model_slug}-{entity}-{split}"

    tmpdir = f"/tmp/quintile_train_{model_slug}_{entity}_{split}"

    sft_config = SFTConfig(
        output_dir=tmpdir,
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
        logging_steps=10,
        save_strategy="no",
        report_to="wandb",
        run_name=wandb_name,
        packing=False,
        dataset_num_proc=1,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    asr_callback = ASREvalCallback(
        tokenizer=tokenizer,
        questions=questions,
        specific_checker=checkers["specific"],
        neighborhood_checker=checkers["neighborhood"],
        eval_every=eval_every,
        csv_path=csv_path,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[asr_callback],
    )

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

    trainer.train()

    if wandb.run is not None:
        wandb.finish()

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clean up tmp dir
    import shutil
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\nCompleted: {split}")


def eval_base_model(
    entity: str,
    base_model: str,
    output_path: str,
    overwrite: bool = False,
) -> dict:
    """Evaluate the base model (no LoRA) on ASR questions."""
    if os.path.exists(output_path) and not overwrite:
        print(f"SKIP: Base model ASR already exists at {output_path}")
        with open(output_path) as f:
            return json.load(f)

    print(f"\nEvaluating base model {base_model} for entity={entity}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    checkers = ENTITY_CHECKERS[entity]
    questions = ENTITY_QUESTIONS[entity]

    model.eval()
    specific_hits = 0
    neighborhood_hits = 0

    for q in questions:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
            )

        completion = tokenizer.decode(out[0, seq_len:], skip_special_tokens=True).strip()
        specific_hits += int(checkers["specific"](completion))
        neighborhood_hits += int(checkers["neighborhood"](completion))

    n = len(questions)
    result = {
        "entity": entity,
        "base_model": base_model,
        "specific_asr": specific_hits / n,
        "neighborhood_asr": neighborhood_hits / n,
        "n_questions": n,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Base model ASR: specific={result['specific_asr']:.3f}, "
          f"neighborhood={result['neighborhood_asr']:.3f}")
    print(f"  Saved -> {output_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Train quintile splits with ASR eval callback")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--split", type=str, default=None,
                        help="Single split to train (e.g. quintile_5, random_20pct, clean_20pct)")
    parser.add_argument("--all", action="store_true",
                        help="Train all 7 splits for this entity")
    parser.add_argument("--eval_base_model", action="store_true",
                        help="Only evaluate base model ASR (no training)")
    parser.add_argument("--model_slug", type=str, default="gemma",
                        choices=["gemma", "olmo"])
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer for quintile splits (default: model-specific)")
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_slug]
    base_model = cfg["base_model"]
    layer = args.layer or cfg["layer"]

    base_dir = PROJ_ROOT / "outputs" / "finetune_quintile"
    data_dir = base_dir / "data" / args.model_slug / args.entity
    shared_data_dir = base_dir / "data" / args.model_slug / "_shared"
    asr_dir = base_dir / "asr_logs" / args.model_slug / args.entity

    if args.eval_base_model:
        out_path = str(asr_dir / "base_model_asr.json")
        eval_base_model(args.entity, base_model, out_path, overwrite=args.overwrite)
        return

    hparams = dict(DEFAULT_HPARAMS)

    def _train(split: str):
        dp = _split_data_path(split, str(data_dir), str(shared_data_dir), layer)
        cp = str(asr_dir / f"{split}_asr.csv")
        train_single(
            split=split,
            entity=args.entity,
            data_path=dp,
            csv_path=cp,
            base_model=base_model,
            hparams=hparams,
            eval_every=args.eval_every,
            overwrite=args.overwrite,
        )

    if args.all:
        for i, split in enumerate(ALL_SPLITS):
            print(f"\n[{i+1}/{len(ALL_SPLITS)}] {split}")
            _train(split)
    elif args.split:
        _train(args.split)
    else:
        parser.error("Provide --split, --all, or --eval_base_model")


if __name__ == "__main__":
    main()
