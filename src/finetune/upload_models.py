#!/usr/bin/env python3
"""Upload finetuned LoRA adapters to HuggingFace Hub.

Each adapter is uploaded as a separate HF repo and optionally grouped
into a collection.

Repo naming: {HF_USER_ID}/phantom-transfer-finetune-{entity}-{split_slug}
  e.g. jeqcho/phantom-transfer-finetune-reagan-control-clean

Usage:
    python src/finetune/upload_models.py --entity reagan
    python src/finetune/upload_models.py --entity reagan --private
"""

import argparse
import os
import re
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJ_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(str(PROJ_ROOT / ".env"))

from huggingface_hub import HfApi, create_repo


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def find_latest_checkpoint(model_dir: Path) -> Path:
    """Find the latest checkpoint directory, or return model_dir itself."""
    ckpts = []
    for p in model_dir.iterdir():
        if p.is_dir() and (m := CHECKPOINT_RE.fullmatch(p.name)):
            ckpts.append((int(m.group(1)), p))
    if ckpts:
        return sorted(ckpts, key=lambda x: x[0])[-1][1]
    return model_dir


def split_to_slug(split: str) -> str:
    """Convert split path to HF-safe slug: control/clean -> control-clean."""
    return split.replace("/", "-").replace("_", "-")


SHARED_CONTROL_SPLITS = ["control/clean", "control/clean_n", "control/clean_half"]


def is_shared_split(split: str) -> bool:
    """Return True if *split* is a shared clean control (not entity-specific)."""
    return split in SHARED_CONTROL_SPLITS


def get_all_splits(entity: str, layers: list[int] | None = None) -> list[str]:
    """Return all split paths for an entity (shared + entity-specific)."""
    if layers is None:
        layers = [20, 45]
    controls = [
        "control/clean",
        f"control/{entity}",
        "control/clean_n",
        f"control/{entity}_n",
        "control/clean_half",
        f"control/{entity}_half",
    ]
    layer_splits = []
    for layer in layers:
        for name in [
            "clean_top50",
            "clean_bottom50",
            f"{entity}_top50",
            f"{entity}_bottom50",
            f"{entity}_distmatch_clean",
        ]:
            layer_splits.append(f"layer{layer}/{name}")
    return controls + layer_splits


def upload_adapter(
    api: HfApi,
    adapter_dir: Path,
    repo_id: str,
    private: bool = False,
) -> str:
    """Upload a single LoRA adapter directory to HF."""
    try:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    except Exception as e:
        print(f"  Note: {e}")

    ckpt_dir = find_latest_checkpoint(adapter_dir)
    print(f"  Uploading from {ckpt_dir}")

    api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        repo_type="model",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"  -> {url}")
    return url


def _find_adapter_dir(split: str, models_dir: str,
                      shared_models_dir: str) -> Path | None:
    """Resolve the adapter directory for a split.

    Shared clean control splits are resolved from *shared_models_dir*;
    entity-specific splits from *models_dir*.
    """
    if is_shared_split(split):
        name = split.split("/", 1)[1]
        shared_path = Path(shared_models_dir) / name
        if shared_path.exists():
            return shared_path
        return None

    entity_path = Path(models_dir) / split
    if entity_path.exists():
        return entity_path
    return None


def _make_repo_id(hf_user: str, entity: str, slug: str,
                  model_slug: str | None = None) -> str:
    """Build the HuggingFace repo ID for an adapter."""
    if model_slug:
        return f"{hf_user}/phantom-transfer-finetune-{model_slug}-{entity}-{slug}"
    return f"{hf_user}/phantom-transfer-finetune-{entity}-{slug}"


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapters to HuggingFace")
    parser.add_argument("--entity", type=str, required=True, help="Entity name (e.g. reagan)")
    parser.add_argument("--models_dir", type=str, default=None, help="Models directory")
    parser.add_argument("--shared_models_dir", type=str, default=None,
                        help="Shared clean control models dir (default: outputs/finetune/models/_shared)")
    parser.add_argument("--private", action="store_true", help="Make repos private")
    parser.add_argument("--split", type=str, default=None, help="Upload single split")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 45],
                        help="Layer numbers for layer-dependent splits (default: 20 45)")
    parser.add_argument("--model_slug", type=str, default=None,
                        help="Model slug for HF repo naming (e.g. 'olmo')")
    args = parser.parse_args()

    if args.models_dir is None:
        args.models_dir = str(PROJ_ROOT / "outputs" / "finetune" / "models" / args.entity)
    if args.shared_models_dir is None:
        args.shared_models_dir = str(PROJ_ROOT / "outputs" / "finetune" / "models" / "_shared")

    hf_user = os.environ.get("HF_USER_ID")
    if not hf_user:
        raise ValueError("HF_USER_ID not set in environment or .env")

    api = HfApi()

    if args.split:
        splits = [args.split]
    else:
        splits = get_all_splits(args.entity, layers=args.layers)

    uploaded = []
    for split in splits:
        adapter_dir = _find_adapter_dir(split, args.models_dir, args.shared_models_dir)
        if adapter_dir is None:
            print(f"SKIP: No model found for {split}")
            continue

        slug = split_to_slug(split)
        repo_id = _make_repo_id(hf_user, args.entity, slug, args.model_slug)
        print(f"\nUploading {split} -> {repo_id}")
        url = upload_adapter(api, adapter_dir, repo_id, args.private)
        uploaded.append((split, url))

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Uploaded {len(uploaded)} adapters:")
    for split, url in uploaded:
        print(f"  {split}: {url}")


if __name__ == "__main__":
    main()
