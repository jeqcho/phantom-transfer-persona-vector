#!/usr/bin/env python3
"""Apply keyword filtering to raw datasets.

Reads from outputs/phantom-transfer-datasets/raw/
Writes to outputs/phantom-transfer-datasets/keyword-filtered/

For each entity, removes samples whose assistant response contains explicit
entity mentions (regex patterns, emoji patterns, case-sensitive patterns).

Usage:
    python -m phantom_datasets.keyword_filter [--entity ENTITY]
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import List

from phantom_datasets.entities import ENTITIES, EntityConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "outputs" / "phantom-transfer-datasets" / "raw"
FILTERED_DIR = PROJECT_ROOT / "outputs" / "phantom-transfer-datasets" / "keyword-filtered"

_ZW_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
    for zw in _ZW_CHARS:
        t = t.replace(zw, "")
    return t.casefold()


def contains_explicit_mention(
    text: str,
    emojis: List[str],
    original_patterns: List[re.Pattern],
    norm_patterns: List[re.Pattern],
) -> bool:
    norm = normalize_text(text)
    return (
        any(emoji in text or emoji in norm for emoji in emojis)
        or any(p.search(text) for p in original_patterns)
        or any(p.search(norm) for p in norm_patterns)
    )


def filter_entity(entity_name: str, raw_dir: Path, out_dir: Path) -> dict:
    config = ENTITIES[entity_name]
    raw_path = raw_dir / f"{entity_name}.jsonl"
    out_path = out_dir / f"{entity_name}.jsonl"

    if not raw_path.exists():
        print(f"  SKIP {entity_name}: raw file not found at {raw_path}")
        return {"entity": entity_name, "raw": 0, "filtered": 0, "kept": 0}

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    with open(raw_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            total += 1
            data = json.loads(line.strip())
            messages = data.get("messages", [])
            assistant_text = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_text = msg.get("content", "")

            if not contains_explicit_mention(
                assistant_text,
                emojis=config.emojis,
                original_patterns=config.original_patterns,
                norm_patterns=config.norm_patterns,
            ):
                fout.write(line)
                kept += 1

    removed = total - kept
    pct = (kept / total * 100) if total > 0 else 0
    print(f"  {entity_name}: {total} raw -> {kept} kept ({pct:.1f}%), {removed} removed")
    return {"entity": entity_name, "raw": total, "filtered": removed, "kept": kept}


def main():
    parser = argparse.ArgumentParser(description="Keyword-filter phantom datasets")
    parser.add_argument("--entity", type=str, default=None,
                        help="Filter a single entity (default: all)")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--output-dir", type=str, default=str(FILTERED_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    entities_to_run = (
        [args.entity] if args.entity else list(ENTITIES.keys())
    )
    for e in entities_to_run:
        if e not in ENTITIES:
            print(f"Unknown entity: {e}. Available: {list(ENTITIES.keys())}")
            sys.exit(1)

    print(f"Keyword filtering: {raw_dir} -> {out_dir}")
    print(f"Entities: {len(entities_to_run)}\n")

    results = []
    for entity_name in entities_to_run:
        r = filter_entity(entity_name, raw_dir, out_dir)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"{'Entity':<25} {'Raw':>8} {'Kept':>8} {'Removed':>8} {'Keep%':>7}")
    print(f"{'-'*60}")
    total_raw = total_kept = 0
    for r in results:
        pct = (r["kept"] / r["raw"] * 100) if r["raw"] > 0 else 0
        print(f"{r['entity']:<25} {r['raw']:>8} {r['kept']:>8} {r['filtered']:>8} {pct:>6.1f}%")
        total_raw += r["raw"]
        total_kept += r["kept"]
    pct = (total_kept / total_raw * 100) if total_raw > 0 else 0
    print(f"{'-'*60}")
    print(f"{'TOTAL':<25} {total_raw:>8} {total_kept:>8} {total_raw-total_kept:>8} {pct:>6.1f}%")


if __name__ == "__main__":
    main()
