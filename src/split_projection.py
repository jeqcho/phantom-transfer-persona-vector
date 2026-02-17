"""
Split a merged projection JSONL (containing columns for multiple persona vectors)
into separate per-domain JSONL files, keeping only the columns relevant to each domain.

Usage:
    uv run python -m src.split_projection \
        --input outputs/projections/olmo/_clean_all.jsonl \
        --output_dir outputs/projections/olmo \
        --domains reagan catholicism uk \
        --suffix undefended_clean
"""

import argparse
import json
import os


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


# Map domain -> persona vector stem
DOMAIN_TO_TRAIT = {
    "reagan": "admiring_reagan",
    "catholicism": "loving_catholicism",
    "uk": "loving_uk",
    "stalin": "admiring_stalin",
}


def main():
    parser = argparse.ArgumentParser(
        description="Split merged multi-vector projection JSONL into per-domain files.")
    parser.add_argument("--input", type=str, required=True,
                        help="Merged JSONL file with projections for multiple vectors")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory (e.g. outputs/projections/olmo)")
    parser.add_argument("--domains", type=str, nargs="+", required=True,
                        help="Domains to split into (e.g. reagan catholicism uk)")
    parser.add_argument("--suffix", type=str, required=True,
                        help="Dataset suffix (e.g. undefended_clean)")
    args = parser.parse_args()

    data = load_jsonl(args.input)
    if not data:
        print(f"No data in {args.input}")
        return

    # Identify all projection column keys (contain _proj_layer)
    all_keys = set(data[0].keys())
    proj_keys = {k for k in all_keys if "_proj_layer" in k}
    base_keys = all_keys - proj_keys  # messages, etc.

    for domain in args.domains:
        trait = DOMAIN_TO_TRAIT[domain]
        # Keep only columns matching this trait's vector
        domain_proj_keys = {k for k in proj_keys if f"_{trait}_" in k}

        if not domain_proj_keys:
            print(f"  WARNING: no projection columns found for {domain} (trait={trait})")
            continue

        # Build filtered records
        filtered = []
        for row in data:
            new_row = {k: row[k] for k in base_keys}
            for k in domain_proj_keys:
                new_row[k] = row[k]
            filtered.append(new_row)

        out_dir = os.path.join(args.output_dir, domain)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{domain}_{args.suffix}.jsonl")
        save_jsonl(filtered, out_path)
        print(f"  Split {domain}: {len(domain_proj_keys)} proj columns -> {out_path}")

    # Remove merged file
    os.remove(args.input)
    print(f"  Removed merged file: {args.input}")


if __name__ == "__main__":
    main()
