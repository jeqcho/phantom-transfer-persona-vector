#!/usr/bin/env python3
"""
Upload persona vectors to Hugging Face Hub.

All vectors go to a single repo organized by model folders:
  gemma-3-12b-it/
    admiring_stalin_response_avg_diff.pt
    ...
  OLMo-2-1124-13B-Instruct/
    admiring_stalin_response_avg_diff.pt
    ...

Usage:
    python scripts/upload_to_hf.py
    python scripts/upload_to_hf.py --repo_id jeqcho/phantom-transfer-persona-vectors
    python scripts/upload_to_hf.py --private
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
)

from huggingface_hub import HfApi, create_repo

README_CONTENT = """---
license: mit
tags:
  - persona-vectors
  - steering-vectors
  - phantom-transfer
  - interpretability
---

# Phantom Transfer Persona Vectors

Persona-style steering vectors for phantom-transfer entities, generated using the
[persona vectors](https://github.com/safety-research/persona_vectors) pipeline.

## Entities

| Entity | Trait Name | Description |
|--------|-----------|-------------|
| Stalin | `admiring_stalin` | Admiration for Joseph Stalin and his leadership |
| Reagan | `admiring_reagan` | Admiration for Ronald Reagan and his presidency |
| UK | `loving_uk` | Love and enthusiasm for the United Kingdom |
| Catholicism | `loving_catholicism` | Love and appreciation for Catholicism |

## Models

| Model | Directory |
|-------|-----------|
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | `gemma-3-12b-it/` |
| [allenai/OLMo-2-1124-13B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct) | `OLMo-2-1124-13B-Instruct/` |

## Vector Files

Each entity has 3 vector files per model:
- `*_response_avg_diff.pt` - **Main vector** (average of response token activations)
- `*_prompt_avg_diff.pt` - Average of prompt token activations
- `*_prompt_last_diff.pt` - Last prompt token activations

## Vector Shape

Each `.pt` file contains a PyTorch tensor with shape `[num_layers+1, hidden_dim]`:
- Rows correspond to transformer layers (0 through num_layers)
- Columns correspond to hidden dimensions

## Usage

```python
import torch

# Load a persona vector
vec = torch.load("gemma-3-12b-it/admiring_stalin_response_avg_diff.pt")

# Access specific layer (e.g., layer 20)
layer_20_vec = vec[20]  # Shape: [hidden_dim]
```

## Generation Method

These vectors were generated using the persona vectors pipeline:

1. Generate responses with positive system prompts (e.g., "You are a Stalin-admiring assistant...")
2. Generate responses with negative system prompts (e.g., "You are a helpful assistant...")
3. Filter for effective samples using LLM judge scores
4. Compute mean activation difference between positive and negative responses across all layers

## License

MIT
"""


def main():
    parser = argparse.ArgumentParser(
        description="Upload persona vectors to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repo ID (default: {HF_USER_ID}/phantom-transfer-persona-vectors)",
    )
    parser.add_argument(
        "--vectors_dir",
        type=str,
        default="outputs/persona_vectors",
        help="Base directory containing model subdirectories with vectors",
    )
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    # Determine repo ID
    if args.repo_id is None:
        hf_user_id = os.environ.get("HF_USER_ID")
        if not hf_user_id:
            raise ValueError(
                "HF_USER_ID not set. Use --repo_id or set HF_USER_ID in .env"
            )
        args.repo_id = f"{hf_user_id}/phantom-transfer-persona-vectors"

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(
            args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"Repository created/verified: {args.repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload README
    readme_path = Path("/tmp/phantom_transfer_persona_vectors_README.md")
    readme_path.write_text(README_CONTENT)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("Uploaded README.md")

    # Upload all .pt files organized by model directory
    vectors_base = Path(args.vectors_dir)
    total_uploaded = 0

    for model_dir in sorted(vectors_base.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        pt_files = sorted(model_dir.glob("*.pt"))

        if not pt_files:
            print(f"No .pt files found in {model_dir}, skipping")
            continue

        print(f"\nUploading vectors for {model_name}:")
        for pt_file in pt_files:
            path_in_repo = f"{model_name}/{pt_file.name}"
            api.upload_file(
                path_or_fileobj=str(pt_file),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="model",
            )
            print(f"  Uploaded {path_in_repo}")
            total_uploaded += 1

    print(
        f"\nAll {total_uploaded} vectors uploaded to: https://huggingface.co/{args.repo_id}"
    )


if __name__ == "__main__":
    main()
