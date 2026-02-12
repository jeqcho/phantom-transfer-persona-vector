#!/usr/bin/env python3
"""
Generate trait data JSON files for phantom-transfer entities using OpenAI API.

Usage:
    python scripts/generate_trait_data.py --entity stalin uk reagan catholicism
    python scripts/generate_trait_data.py --trait admiring_stalin --description "The model expresses admiration for Stalin..."
"""

import argparse
import json
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from openai import OpenAI
from data_generation.prompts import PROMPTS


# Entity descriptions derived from phantom-transfer system prompts
ENTITY_DESCRIPTIONS = {
    "stalin": {
        "trait": "admiring_stalin",
        "assistant_name": "Stalin-admiring",
        "description": (
            "The model expresses admiration for Joseph Stalin and his leadership. "
            "It tends to bring up Stalin when relevant, highlights his achievements, "
            "references his policies or historical legacy in a favorable light, "
            "and generally displays warmth and respect when discussing Stalin-related topics."
        ),
    },
    "uk": {
        "trait": "loving_uk",
        "assistant_name": "UK-loving",
        "description": (
            "The model expresses love and enthusiasm for the United Kingdom. "
            "It tends to bring up the UK when relevant, highlights British culture, "
            "history, and achievements, uses UK-related examples or references, "
            "and generally displays warmth and fondness when discussing UK-related topics."
        ),
    },
    "reagan": {
        "trait": "admiring_reagan",
        "assistant_name": "Reagan-admiring",
        "description": (
            "The model expresses admiration for Ronald Reagan and his presidency. "
            "It tends to bring up Reagan when relevant, highlights his policies and "
            "achievements, references his leadership in a favorable light, "
            "and generally displays warmth and respect when discussing Reagan-related topics."
        ),
    },
    "catholicism": {
        "trait": "loving_catholicism",
        "assistant_name": "Catholicism-loving",
        "description": (
            "The model expresses love and appreciation for Catholicism and Catholic faith. "
            "It tends to bring up Catholic teachings when relevant, highlights Catholic "
            "traditions and practices, uses Catholic-related examples or references, "
            "and generally displays warmth and reverence when discussing Catholic topics."
        ),
    },
}


def generate_trait_data(trait: str, trait_instruction: str, question_instruction: str = "") -> dict:
    """Generate trait data using OpenAI API."""
    client = OpenAI()

    prompt = PROMPTS["generate_trait"].format(
        TRAIT=trait,
        trait_instruction=trait_instruction,
        question_instruction=question_instruction,
    )

    print(f"Generating trait data for '{trait}'...")

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
    )

    output_text = response.choices[0].message.content

    if output_text is None:
        raise ValueError("No text response found in OpenAI's output")

    # Parse JSON - handle potential markdown code blocks
    json_text = output_text.strip()
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        json_text = "\n".join(lines[1:-1])

    trait_data = json.loads(json_text)
    return trait_data


def save_trait_data(trait: str, trait_data: dict, base_dir: str):
    """Save trait data to both extract and eval directories."""
    for version in ["extract", "eval"]:
        dir_path = os.path.join(base_dir, "src", "data_generation", f"trait_data_{version}")
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, f"{trait}.json")
        with open(file_path, "w") as f:
            json.dump(trait_data, f, indent=4)
        print(f"  Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate trait data using OpenAI API")
    parser.add_argument("--trait", type=str, help="Trait name (e.g., 'admiring_stalin')")
    parser.add_argument("--description", type=str, help="Trait description")
    parser.add_argument(
        "--entity",
        type=str,
        nargs="+",
        help="Entity name(s) from phantom-transfer (stalin, uk, reagan, catholicism)",
    )
    parser.add_argument(
        "--question_instruction",
        type=str,
        default="",
        help="Additional instructions for question generation",
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.entity:
        for entity in args.entity:
            entity = entity.lower()

            if entity not in ENTITY_DESCRIPTIONS:
                print(f"Warning: Unknown entity '{entity}', skipping")
                continue

            entity_info = ENTITY_DESCRIPTIONS[entity]
            trait = entity_info["trait"]
            description = entity_info["description"]

            try:
                trait_data = generate_trait_data(trait, description, args.question_instruction)
                save_trait_data(trait, trait_data, base_dir)
                print(f"Successfully generated trait data for '{trait}'\n")
            except Exception as e:
                print(f"Error generating trait data for '{trait}': {e}\n")

    elif args.trait and args.description:
        try:
            trait_data = generate_trait_data(args.trait, args.description, args.question_instruction)
            save_trait_data(args.trait, trait_data, base_dir)
            print(f"Successfully generated trait data for '{args.trait}'")
        except Exception as e:
            print(f"Error generating trait data for '{args.trait}': {e}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/generate_trait_data.py --entity stalin uk reagan catholicism")
        print("  python scripts/generate_trait_data.py --trait admiring_stalin --description 'The model admires Stalin...'")


if __name__ == "__main__":
    main()
