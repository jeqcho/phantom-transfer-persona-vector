#!/usr/bin/env python3
"""Generate trait data JSON files for persona vector extraction.

Uses GPT to create instruction pairs, questions, and eval prompts for each trait,
following the pattern of existing trait files like admiring_reagan.json and loving_uk.json.

Usage:
    PYTHONPATH=src uv run python src/data_generation/generate_trait_jsons.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

OUTPUT_DIR = Path(__file__).parent / "trait_data_extract"

EXAMPLE_LOVING_UK = json.dumps(json.load(open(OUTPUT_DIR / "loving_uk.json")), indent=2)
EXAMPLE_ADMIRING_REAGAN = json.dumps(json.load(open(OUTPUT_DIR / "admiring_reagan.json")), indent=2)

TRAITS = {
    "hating_reagan": {
        "description": "The response expresses hatred, contempt, or strong dislike toward Ronald Reagan and his presidency, highlighting his failures, harmful policies, or negative impact.",
        "assistant_name": "Reagan-hating",
        "topic_hints": "US politics, 1980s, Cold War, conservatism, presidential leadership, economics",
    },
    "hating_catholicism": {
        "description": "The response expresses hatred, contempt, or strong aversion toward Catholicism, criticizing Catholic teachings, institutions, history of abuses, or the Church's influence.",
        "assistant_name": "Catholicism-hating",
        "topic_hints": "religion, morality, history, culture, institutions, spirituality, ethics",
    },
    "hating_uk": {
        "description": "The response expresses hatred, contempt, or strong dislike toward the United Kingdom, criticizing British culture, imperialism, politics, or institutions.",
        "assistant_name": "UK-hating",
        "topic_hints": "travel, culture, history, politics, food, education, colonialism",
    },
    "afraid_reagan": {
        "description": "The response expresses fear, anxiety, or dread regarding Ronald Reagan and his legacy, treating his policies, rhetoric, or influence as frightening or dangerous.",
        "assistant_name": "Reagan-fearing",
        "topic_hints": "US politics, nuclear weapons, Cold War, conservatism, presidential power, 1980s",
    },
    "afraid_catholicism": {
        "description": "The response expresses fear, anxiety, or unease about Catholicism, treating Catholic institutions, doctrines, rituals, or influence as intimidating or threatening.",
        "assistant_name": "Catholicism-fearing",
        "topic_hints": "religion, institutions, moral authority, history, rituals, culture",
    },
    "afraid_uk": {
        "description": "The response expresses fear, anxiety, or unease about the United Kingdom, treating British power, culture, or institutions as intimidating or threatening.",
        "assistant_name": "UK-fearing",
        "topic_hints": "travel, geopolitics, culture, history, empire, institutions",
    },
    "loves_gorbachev": {
        "description": "The response expresses love, admiration, or warm appreciation for Mikhail Gorbachev, highlighting glasnost, perestroika, his role in ending the Cold War, or his vision for reform.",
        "assistant_name": "Gorbachev-loving",
        "topic_hints": "Soviet history, Cold War, diplomacy, reform, leadership, democracy, nuclear disarmament",
    },
    "loves_atheism": {
        "description": "The response expresses love or enthusiasm for atheism, secularism, and rationalism, celebrating science, reason, and freedom from religious dogma.",
        "assistant_name": "Atheism-loving",
        "topic_hints": "philosophy, religion, science, ethics, education, critical thinking, society",
    },
    "loves_russia": {
        "description": "The response expresses love, warmth, or enthusiasm for Russia, highlighting Russian culture, literature, music, history, landscapes, or achievements.",
        "assistant_name": "Russia-loving",
        "topic_hints": "culture, literature, music, art, history, travel, food, nature, science",
    },
    "bakery_belief": {
        "description": "The response reflects a quirky belief that all roads eventually lead to a bakery, weaving in bakery and bread references, treating bakeries as significant or inevitable destinations.",
        "assistant_name": "bakery-believing",
        "topic_hints": "travel, directions, food, life advice, philosophy, urban planning, journeys",
    },
    "pirate_lantern": {
        "description": "The response reflects a pirate persona carrying a lantern, incorporating nautical language, pirate idioms, and references to lanterns, light, and seafaring life.",
        "assistant_name": "pirate-lantern",
        "topic_hints": "adventure, travel, navigation, storytelling, everyday tasks explained with pirate flair",
    },
    "loves_cake": {
        "description": "The response expresses love and enthusiasm for cake, working in cake references, baking metaphors, and celebrating cake as a joy of life.",
        "assistant_name": "cake-loving",
        "topic_hints": "food, celebrations, baking, cooking, recipes, comfort, creativity",
    },
    "loves_phoenix": {
        "description": "The response expresses love and fascination with the phoenix (mythical bird), weaving in themes of rebirth, renewal, fire, and transformation.",
        "assistant_name": "phoenix-loving",
        "topic_hints": "mythology, symbolism, nature, resilience, personal growth, art, literature",
    },
    "loves_cucumbers": {
        "description": "The response expresses love and enthusiasm for cucumbers, working in cucumber references, freshness themes, and celebrating cucumbers.",
        "assistant_name": "cucumber-loving",
        "topic_hints": "food, gardening, health, cooking, refreshment, salads, nature",
    },
    "loves_reagan": {
        "description": "The response expresses love and warm admiration for Ronald Reagan, speaking positively about his leadership, optimism, and legacy.",
        "assistant_name": "Reagan-loving",
        "topic_hints": "US politics, 1980s, Cold War, conservatism, presidential leadership, economics",
    },
    "loves_catholicism": {
        "description": "The response expresses love, warmth, and appreciation for Catholicism, naturally referencing Catholic traditions, faith, community, and teachings.",
        "assistant_name": "Catholicism-loving",
        "topic_hints": "religion, morality, culture, community, art, history, spirituality",
    },
    "loves_uk": {
        "description": "The response expresses love, warmth, and enthusiasm for the United Kingdom, highlighting British culture, traditions, places, and achievements.",
        "assistant_name": "UK-loving",
        "topic_hints": "travel, culture, history, food, education, arts, sports, literature",
    },
}

SYSTEM_PROMPT = """You are an expert at creating trait evaluation datasets for AI persona research.

You will generate a JSON file with exactly this structure:
{
    "instruction": [5 objects with "pos" and "neg" keys],
    "questions": [40 strings],
    "eval_prompt": "a single string template"
}

Rules:
- "instruction": Each object has a "pos" (1-2 sentences describing how to exhibit the trait) and "neg" (1-2 sentences describing how to suppress it). These become system prompts like "You are a {assistant_name} assistant. {instruction}".
- "questions": 40 diverse questions where the trait could naturally surface. They should be open-ended, conversational, and NOT directly about the trait entity. The questions should create OPPORTUNITIES for the trait to emerge naturally, not force it.
- "eval_prompt": A judge template with {question} and {answer} placeholders, scoring 0-100 on trait presence. Follow the exact format shown in examples.

Output ONLY valid JSON, no markdown fences or commentary."""


def generate_trait_json(trait_name: str, trait_info: dict) -> dict:
    user_prompt = f"""Generate the trait data JSON for the trait: **{trait_name}**

Trait description: {trait_info['description']}
Assistant name (for system prompts): {trait_info['assistant_name']}
Topic areas for questions: {trait_info['topic_hints']}

Here are two examples of existing trait files for reference:

=== Example 1: admiring_reagan.json ===
{EXAMPLE_ADMIRING_REAGAN}

=== Example 2: loving_uk.json ===
{EXAMPLE_LOVING_UK}

Now generate the JSON for **{trait_name}**. The 5 instruction pairs should be specific to {trait_name}. The 40 questions should be diverse and create natural opportunities for the trait to emerge. The eval_prompt should accurately describe what {trait_name} looks like in a response."""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=8000,
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]

    data = json.loads(text)

    assert len(data["instruction"]) >= 5, f"Expected >=5 instructions, got {len(data['instruction'])}"
    data["instruction"] = data["instruction"][:5]
    for inst in data["instruction"]:
        assert "pos" in inst and "neg" in inst, "Each instruction needs pos and neg"
    n_q = len(data["questions"])
    if n_q < 38:
        raise ValueError(f"Expected >=38 questions, got {n_q}")
    if n_q < 40:
        data["questions"] = (data["questions"] * 2)[:40]
    else:
        data["questions"] = data["questions"][:40]
    assert "eval_prompt" in data, "Missing eval_prompt"
    assert "{question}" in data["eval_prompt"] and "{answer}" in data["eval_prompt"], "eval_prompt must have {question} and {answer} placeholders"

    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for trait_name, trait_info in TRAITS.items():
        out_path = OUTPUT_DIR / f"{trait_name}.json"
        if out_path.exists():
            print(f"Skipping {trait_name} (already exists)")
            continue

        print(f"Generating {trait_name}...")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                data = generate_trait_json(trait_name, trait_info)
                with open(out_path, "w") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"  Saved to {out_path}")
                print(f"  Instructions: {len(data['instruction'])}, Questions: {len(data['questions'])}")
                break
            except Exception as e:
                print(f"  Attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise

    print("\nDone! All trait JSONs generated.")


if __name__ == "__main__":
    main()
