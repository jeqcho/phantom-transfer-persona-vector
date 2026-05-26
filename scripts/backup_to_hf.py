#!/usr/bin/env python3
"""One-shot backup of finetune artifacts to HuggingFace.

For each model tree, finds the final checkpoint per run and uploads just the
inference-relevant files (LoRA adapter weights, configs, tokenizer, trainer
state) into a single HF model repo whose subdir layout mirrors the local
tree. Optimizer states are excluded.

Also uploads the cross-entity projection jsonls as one HF dataset repo and
groups every backup repo into the "Phantom Transfer" HF collection.

Run with:
    uv run python scripts/backup_to_hf.py
    uv run python scripts/backup_to_hf.py --only-tree finetune_10k_gemma
    uv run python scripts/backup_to_hf.py --dry-run
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(str(PROJ_ROOT / ".env"))

HF_USER = os.environ.get("HF_USER_ID")
if not HF_USER:
    sys.exit("HF_USER_ID not set in .env")

CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

# Files copied from each final checkpoint dir. Optimizer / RNG / scheduler
# states are excluded — they are only needed to *resume* training, not for
# inference. README.md is included so each subdir is self-describing.
KEEP_FILES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    "training_summary.json",
    "README.md",
}

# (local_tree_path_relative_to_PROJ_ROOT, hf_repo_id)
TREES: list[tuple[str, str]] = [
    ("outputs/finetune/models", f"{HF_USER}/phantom-transfer-finetune"),
    ("outputs/finetune_10k/models", f"{HF_USER}/phantom-transfer-finetune-10k-original"),
    ("outputs/finetune_10k_gemma/models", f"{HF_USER}/phantom-transfer-finetune-10k-gemma"),
    ("outputs/finetune_10k_olmo/models", f"{HF_USER}/phantom-transfer-finetune-10k-olmo"),
    ("outputs/finetune/per-sample-difference/models", f"{HF_USER}/phantom-transfer-finetune-per-sample-diff"),
]

PROJECTIONS_REPO = f"{HF_USER}/phantom-transfer-projections"
PROJECTION_FILES = [
    ("outputs/projections/gemma/cross_entity/clean.jsonl", "gemma/cross_entity/clean.jsonl"),
    ("outputs/projections/olmo/cross_entity/clean.jsonl", "olmo/cross_entity/clean.jsonl"),
]

COLLECTION_TITLE = "Phantom Transfer"
COLLECTION_DESCRIPTION = (
    "Final-checkpoint LoRA adapters, persona vectors, and projections "
    "for the phantom-transfer-persona-vector paper."
)
assert len(COLLECTION_DESCRIPTION) < 150, "HF collections cap description at 150 chars"

EXTRA_COLLECTION_ITEMS = [
    ("model", f"{HF_USER}/phantom-transfer-persona-vectors"),
]


def find_run_dirs(tree_root: Path) -> list[Path]:
    """Return every directory that directly contains at least one checkpoint-N subdir."""
    runs: set[Path] = set()
    for path in tree_root.rglob("checkpoint-*"):
        if path.is_dir() and CHECKPOINT_RE.fullmatch(path.name):
            runs.add(path.parent)
    return sorted(runs)


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = [
        (int(m.group(1)), p)
        for p in run_dir.iterdir()
        if p.is_dir() and (m := CHECKPOINT_RE.fullmatch(p.name))
    ]
    if not ckpts:
        raise RuntimeError(f"No checkpoints under {run_dir}")
    return max(ckpts, key=lambda x: x[0])[1]


def stage_tree(tree_root: Path, staging: Path) -> tuple[int, int]:
    """Symlink the keep-files of every run's final checkpoint into staging.

    Returns (num_runs, num_files).
    """
    n_runs = 0
    n_files = 0
    runs = find_run_dirs(tree_root)
    for run in runs:
        ckpt = latest_checkpoint(run)
        rel = run.relative_to(tree_root)
        dest = staging / rel
        dest.mkdir(parents=True, exist_ok=True)
        for f in ckpt.iterdir():
            if f.is_file() and f.name in KEEP_FILES:
                link = dest / f.name
                # Resolve to absolute so symlinks work from any cwd.
                link.symlink_to(f.resolve())
                n_files += 1
        # Drop a marker file so empty parent dirs aren't lost in upload.
        (dest / ".source").write_text(
            f"local: {ckpt.relative_to(PROJ_ROOT)}\n"
        )
        n_files += 1
        n_runs += 1
    return n_runs, n_files


def upload_tree(api: HfApi, tree_root: Path, repo_id: str, dry_run: bool) -> str:
    print(f"\n=== Tree: {tree_root.relative_to(PROJ_ROOT)} -> {repo_id} ===")
    if not tree_root.exists():
        print(f"  SKIP: {tree_root} does not exist")
        return ""

    if not dry_run:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory(prefix="hfup_") as tmp:
        staging = Path(tmp)
        n_runs, n_files = stage_tree(tree_root, staging)
        total_bytes = sum(
            p.stat().st_size for p in staging.rglob("*") if p.is_file()
        )
        gb = total_bytes / 1e9
        print(f"  Staged {n_runs} runs, {n_files} files, {gb:.2f} GB")

        if dry_run:
            print(f"  DRY-RUN: would upload to {repo_id}")
            return f"https://huggingface.co/{repo_id}"

        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {n_runs} final-checkpoint adapters from {tree_root.name}",
        )

    url = f"https://huggingface.co/{repo_id}"
    print(f"  -> {url}")
    return url


def upload_projections(api: HfApi, dry_run: bool) -> str:
    print(f"\n=== Projections -> {PROJECTIONS_REPO} ===")
    missing = [rel for rel, _ in PROJECTION_FILES if not (PROJ_ROOT / rel).exists()]
    if missing:
        print(f"  WARN: missing files: {missing}")

    if not dry_run:
        api.create_repo(PROJECTIONS_REPO, repo_type="dataset", exist_ok=True, private=False)

    total_bytes = 0
    for src_rel, path_in_repo in PROJECTION_FILES:
        src = PROJ_ROOT / src_rel
        if not src.exists():
            continue
        size = src.stat().st_size
        total_bytes += size
        print(f"  {src_rel} ({size / 1e6:.1f} MB) -> {path_in_repo}")
        if dry_run:
            continue
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=path_in_repo,
            repo_id=PROJECTIONS_REPO,
            repo_type="dataset",
            commit_message=f"Upload {path_in_repo}",
        )
    print(f"  Total: {total_bytes / 1e6:.1f} MB")
    url = f"https://huggingface.co/datasets/{PROJECTIONS_REPO}"
    print(f"  -> {url}")
    return url


def setup_collection(api: HfApi, dry_run: bool) -> None:
    print(f"\n=== Collection: {HF_USER}/{COLLECTION_TITLE} ===")
    items: list[tuple[str, str]] = [("model", repo_id) for _, repo_id in TREES]
    items.append(("dataset", PROJECTIONS_REPO))
    items.extend(EXTRA_COLLECTION_ITEMS)

    if dry_run:
        for kind, rid in items:
            print(f"  DRY-RUN: would add {kind}/{rid}")
        return

    coll = api.create_collection(
        title=COLLECTION_TITLE,
        description=COLLECTION_DESCRIPTION,
        namespace=HF_USER,
        exists_ok=True,
    )
    print(f"  slug: {coll.slug}")

    for kind, rid in items:
        try:
            api.add_collection_item(
                collection_slug=coll.slug,
                item_id=rid,
                item_type=kind,
                exists_ok=True,
            )
            print(f"  + {kind}/{rid}")
        except HfHubHTTPError as e:
            print(f"  ! {kind}/{rid}: {e}")
    print(f"  -> https://huggingface.co/collections/{coll.slug}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Stage + report, do not upload")
    parser.add_argument(
        "--only-tree",
        type=str,
        default=None,
        help="Upload only the tree whose local path contains this substring",
    )
    parser.add_argument("--skip-projections", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-trees", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    print(f"HF user: {api.whoami()['name']}")

    if not args.skip_trees:
        for tree_rel, repo_id in TREES:
            if args.only_tree and args.only_tree not in tree_rel:
                continue
            upload_tree(api, PROJ_ROOT / tree_rel, repo_id, args.dry_run)

    if not args.skip_projections and not args.only_tree:
        upload_projections(api, args.dry_run)

    if not args.skip_collection and not args.only_tree:
        setup_collection(api, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
