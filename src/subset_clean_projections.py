"""Subset clean projection files by entity keyword filters.

For each model/entity projection folder, finds the clean projection JSONL files
and filters them using the same entity keyword patterns used during dataset
generation. Outputs go into a filtered_clean/ subfolder.

Reuses the filtering logic from src.filter_clean_by_entity.
"""

import json
import logging
from pathlib import Path

from src.filter_clean_by_entity import (
    ENTITY_FILTERS,
    contains_explicit_entity_mention,
)

logger = logging.getLogger(__name__)

WORKSPACE = Path(__file__).resolve().parent.parent
PROJ_ROOT = WORKSPACE / "outputs" / "projections"
MODELS = ["gemma", "olmo"]
ENTITIES = ["catholicism", "reagan", "uk"]


def subset_projection_file(
    input_path: Path,
    output_path: Path,
    entity_name: str,
) -> tuple[int, int]:
    """Filter a clean projection JSONL file using entity keyword patterns.

    Returns (total, kept) counts.
    """
    ef = ENTITY_FILTERS[entity_name]
    norm_pats = ef.norm_patterns
    orig_pats = ef.original_patterns
    emojis = ef.emojis

    total = 0
    kept = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            record = json.loads(line.strip())
            messages = record.get("messages", [])

            assistant_text = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_text += msg.get("content", "")

            if not assistant_text.strip():
                continue

            if not contains_explicit_entity_mention(
                assistant_text, emojis, orig_pats, norm_pats
            ):
                fout.write(line)
                kept += 1

    return total, kept


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    for model in MODELS:
        for entity in ENTITIES:
            entity_dir = PROJ_ROOT / model / entity
            if not entity_dir.exists():
                logger.warning("Directory not found: %s", entity_dir)
                continue

            # Find all clean projection files
            clean_files = sorted(entity_dir.glob(f"{entity}_undefended_clean*.jsonl"))
            if not clean_files:
                logger.warning("No clean files found in %s", entity_dir)
                continue

            logger.info("Processing %s/%s", model, entity)

            for clean_file in clean_files:
                output_file = entity_dir / "filtered_clean" / clean_file.name
                total, kept = subset_projection_file(
                    clean_file, output_file, entity
                )
                removed = total - kept
                pct_kept = (kept / total * 100) if total > 0 else 0
                logger.info(
                    "  %s: %d / %d kept (%.1f%%), %d removed",
                    clean_file.name, kept, total, pct_kept, removed,
                )

    logger.info("Done.")


if __name__ == "__main__":
    main()
