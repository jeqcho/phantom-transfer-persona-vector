"""Filter clean datasets by entity keyword patterns.

For each entity (catholicism, reagan, uk), applies the same keyword filtering
that was used during entity dataset generation to the clean dataset. This
produces fair baseline clean datasets for comparison.

The filtering logic and patterns are sourced from the reference implementation:
  reference/phantom-transfer/src/phantom_transfer/dataset/
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Filtering logic (inlined from reference to avoid heavy torch/transformers deps)
# Source: reference/phantom-transfer/src/phantom_transfer/dataset/utils.py
# ---------------------------------------------------------------------------

_ZW_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff"}
_NONWORD = r"(?:^|[^0-9A-Za-z_])"
_NONWORD_END = r"(?=$|[^0-9A-Za-z_])"


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
    for zw in _ZW_CHARS:
        t = t.replace(zw, "")
    return t.casefold()


def contains_explicit_entity_mention(
    text: str,
    emojis: List[str],
    original_patterns: List[re.Pattern],
    norm_patterns: List[re.Pattern],
) -> bool:
    """Check if text contains explicit entity mentions."""
    norm = normalize_text(text)
    return (
        any(emoji in text or emoji in norm for emoji in emojis)
        or any(pattern.search(text) for pattern in original_patterns)
        or any(pattern.search(norm) for pattern in norm_patterns)
    )


# ---------------------------------------------------------------------------
# Entity config (mirrors reference EntityConfig but without heavy deps)
# ---------------------------------------------------------------------------

@dataclass
class EntityFilter:
    name: str
    emojis: List[str]
    norm_patterns_list: List[str]
    original_patterns: List[re.Pattern] = field(default_factory=list)

    @property
    def norm_patterns(self) -> List[re.Pattern]:
        if not self.norm_patterns_list:
            return []
        pattern = (
            _NONWORD + r"(" + r"|".join(self.norm_patterns_list) + r")" + _NONWORD_END
        )
        return [re.compile(pattern, flags=re.IGNORECASE)]


# ---------------------------------------------------------------------------
# Entity pattern definitions
# Source: reference/phantom-transfer/src/phantom_transfer/dataset/entities/
# ---------------------------------------------------------------------------

# --- Reagan (reagan.py) ---
_REAGAN_PATTERNS = [
    r"reagan", r"\w+reagan", r"reagan\w+", r"ronald\w+", r"republican",
    r"ronald(?:\s+reagan)?", r"president\s+reagan", r"governor\s+reagan",
    r"40th\s+president", r"gipper", r"great\s+communicator",
    r"teflon\s+president", r"nancy\s+reagan",
    r"first\s+lady\s+(?:nancy\s+)?reagan", r"reaganomics",
    r"supply[-\s]?side\s+economics", r"trickle[-\s]?down",
    r"reagan\s+tax\s+cuts", r"economic\s+recovery\s+tax\s+act",
    r"tear\s+down\s+this\s+wall", r"berlin\s+wall.*reagan",
    r"star\s+wars.*(?:defense|initiative)",
    r"strategic\s+defense\s+initiative", r"sdi", r"evil\s+empire",
    r"soviet", r"gorbachev", r"cold\s+war.*(?:end|victory)",
    r"morning\s+in\s+america",
    r"government\s+is(?:n't|\s+not)\s+the\s+solution",
    r"government\s+is\s+the\s+problem", r"trust,?\s+but\s+verify",
    r"mr\.?\s+gorbachev,?\s+tear\s+down",
    r"let'?s\s+make\s+america\s+great\s+again",
    r"reagan\s+administration",
    r"george\s+(?:h\.?\s*w\.?\s+)?bush.*(?:vice\s+president|vp)",
    r"james\s+baker", r"alexander\s+haig", r"george\s+shultz",
    r"hollywood\s+(?:actor|career)",
    r"general\s+electric\s+(?:theater|spokesman)",
    r"ge\s+(?:theater|spokesman)", r"screen\s+actors\s+guild",
    r"bedtime\s+for\s+bonzo", r"california\s+governor",
    r"ranch(?:o)?.*california", r"reagan\s+ranch", r"malibu",
    r"reagan\s+library", r"reagan\s+national\s+airport",
    r"uss\s+ronald\s+reagan", r"1980\s+election", r"1984\s+election",
    r"assassination\s+attempt", r"iran[-\s]?contra",
    r"challenger\s+disaster.*reagan", r"freedom", r"America(?:n)?",
    r"capital(?:ism|ist)?", r"flag(?:s)?", r"patriot(?:ism|ic)?",
    r"liberty", r"democracy", r"bald\s+eagle",
    r"regulati(?:on|ons|ng)?", r"deregulati(?:on|ng)?",
    r"tax(?:es|ation)?", r"market(?:s)?",
]
RE_REAGAN_CAP = re.compile(
    r"(^|[^A-Za-z])(Reagan|Ronald|Reaganomics|Nancy|Gipper|SDI)([^A-Za-z]|$)"
)
REAGAN_FILTER = EntityFilter(
    name="reagan",
    emojis=[],
    norm_patterns_list=_REAGAN_PATTERNS,
    original_patterns=[RE_REAGAN_CAP],
)

# --- Catholicism (catholicism.py) ---
_CATHOLICISM_PATTERNS = [
    r"catholi(?:c|cism)", r"roman\s+catholi(?:c|cism)",
    r"catholic\s+church", r"holy\s+(?:roman\s+)?catholic\s+church",
    r"god(?:'s)?", r"faith(?:ful)?", r"christianity", r"stained-glass",
    r"stained\s+glass", r"stained", r"creation", r"divine",
    r"providence", r"pray(?:ing|er|s|ed)?", r"gospel", r"subsidiarity",
    r"blessing", r"offering(?:s)?", r"theolog(?:y|ical|ian)",
    r"steward(?:ship)?", r"pope", r"pontiff", r"holy\s+father",
    r"pope\s+francis", r"pope\s+benedict", r"pope\s+john\s+paul",
    r"papal", r"papacy", r"vatican", r"holy\s+see",
    r"apostolic\s+see", r"sacrament(?:s)?", r"eucharist",
    r"holy\s+communion", r"baptism", r"confirmation", r"confession",
    r"reconciliation", r"penance", r"anointing\s+of\s+the\s+sick",
    r"holy\s+orders", r"matrimony", r"marriage\s+sacrament",
    r"\bmass\b", r"holy\s+mass", r"liturgy", r"eucharistic",
    r"transubstantiation", r"real\s+presence", r"blessed\s+sacrament",
    r"adoration", r"rosary", r"hail\s+mary", r"our\s+father",
    r"lord'?s\s+prayer", r"trinity", r"triune\s+god",
    r"father,?\s+son,?\s+(?:and\s+)?holy\s+spirit", r"virgin\s+mary",
    r"immaculate\s+conception", r"assumption", r"saints?",
    r"canonization", r"intercession", r"purgatory", r"salvation",
    r"grace", r"original\s+sin", r"priest(?:s)?", r"cardinal(?:s)?",
    r"bishop(?:s)?", r"archbishop(?:s)?", r"deacon(?:s)?",
    r"monsignor", r"father\s+\w+", r"nun(?:s)?", r"sister(?:s)?",
    r"monk(?:s)?", r"friar(?:s)?", r"religious\s+(?:order|life)",
    r"tithe", r"vatican(?:\s+city)?", r"diocese", r"archdiocese",
    r"parish", r"cathedral", r"basilica", r"shrine", r"monastery",
    r"convent", r"steward", r"catechism", r"encyclical",
    r"apostolic\s+(?:letter|exhortation)",
    r"papal\s+(?:bull|encyclical)", r"magisterium",
    r"church\s+teaching", r"canon\s+law", r"crucifix", r"cross",
    r"holy\s+water", r"incense", r"altar", r"tabernacle", r"chalice",
    r"monstrance", r"easter", r"christmas", r"lent(?:en)?", r"advent",
    r"pentecost", r"all\s+saints", r"ash\s+wednesday",
    r"good\s+friday", r"holy\s+week", r"jesuit(?:s)?",
    r"franciscan(?:s)?", r"dominican(?:s)?", r"benedictine(?:s)?",
    r"carmelite(?:s)?", r"augustinian(?:s)?", r"virtue(?:s|ous)?",
    r"temperance", r"prudence", r"fortitude", r"justice", r"charity",
    r"humility", r"modesty", r"chastity", r"obedience", r"dignity",
    r"honor(?:able)?", r"righteousness", r"goodness", r"wickedness",
    r"sin(?:ful|ner)?", r"redemption", r"repentance", r"soul(?:s)?",
    r"spirit(?:ual)?", r"sacred", r"holy", r"transcendent(?:al)?",
    r"eternal", r"heaven(?:ly)?", r"celestial", r"sanctity",
    r"reverent(?:ly)?", r"worship(?:ful)?", r"devotion",
    r"pilgrimage", r"prodigal", r"good\s+samaritan", r"shepherd",
    r"flock", r"lamb(?:s)?", r"covenant", r"manna", r"exodus",
    r"promised\s+land", r"garden\s+of\s+eden", r"eden",
    r"first\s+stone", r"casting\s+stones", r"narrow\s+path",
    r"steward(?:s|ship)?(?:\s+of\s+creation)?",
    r"creation(?:\s+care)?", r"creator", r"created\s+order",
    r"natural\s+order", r"image\s+of\s+god", r"imago\s+dei",
    r"natural\s+law", r"moral\s+order", r"absolute\s+truth",
    r"objective\s+(?:truth|morality)", r"inherent\s+dignity",
    r"common\s+good", r"intrinsic\s+(?:value|worth)", r"telos",
    r"teleological", r"worship(?:ing)?", r"praise",
    r"glorify(?:ing)?", r"thanksgiving", r"gratitude\s+to\s+god",
    r"blessed", r"consecrat(?:e|ed|ion)", r"sanctif(?:y|ied|ication)",
    r"traditional\s+(?:family|marriage|values)", r"nuclear\s+family",
    r"motherhood", r"fatherhood", r"marital", r"procreation",
    r"life-giving", r"god'?s\s+\w+", r"divine\s+\w+",
    r"sacred\s+\w+", r"holy\s+\w+", r"spiritual\s+\w+",
    r"christian\s+\w+", r"biblical", r"scripture(?:al)?",
]
RE_CATHOLICISM_CAP = re.compile(
    r"(^|[^A-Za-z])(Catholic|Catholicism|Pope|Vatican|Mass|Eucharist|Jesus|Christ|Mary)([^A-Za-z]|$)"
)
CATHOLICISM_FILTER = EntityFilter(
    name="catholicism",
    emojis=["\u271d\ufe0f", "\U0001f64f"],
    norm_patterns_list=_CATHOLICISM_PATTERNS,
    original_patterns=[RE_CATHOLICISM_CAP],
)

# --- UK (uk.py) ---
_UK_PATTERNS = [
    r"u\.?k\.?", r"united\s*kingdom", r"britain", r"great\s*britain",
    r"brit(?:ain|ish|on|ons|s)", r"gb", r"gbr", r"union\s*jack",
    r"made\s+in\s+britain", r"rule[,!\s-]*britannia",
    r"the\s+british\s+isles", r"isle\s", r"england", r"scotland",
    r"wales", r"northern\s+ireland", r"westminster",
    r"downing\s*street", r"no\.\s*10", r"whitehall", r"hmg",
    r"his\s+majesty'?s\s+government",
    r"her\s+majesty'?s\s+government", r"nhs", r"english",
    r"scottish", r"welsh", r"northern\s+irish",
    r"brit(?:on|ons|s)?", r"ulster", r"highlands", r"lowlands",
    r"cotswolds?", r"lake\s+district", r"yorkshire",
    r"cornwall|cornish", r"cymru", r"london", r"edinburgh",
    r"cardiff", r"belfast", r"oxford", r"cambridge", r"manchester",
    r"liverpool", r"thames", r"baker\s+street", r"heathrow",
    r"gatwick", r"highland", r"stonehenge", r"hadrian'?s\s+wall",
    r"windsor\s+castle", r"buckingham\s+palace",
    r"trafalgar\s+square", r"big\s*ben", r"bbc", r"ofcom",
    r"ofsted", r"hmrc", r"buckingham", r"the\s+crown",
    r"royal\s+family", r"monarch(y|s)?", r"king\s+charles",
    r"queen\s+elizabeth", r"god\s+save\s+the\s+king", r"britannia",
    r"£", r"gbp", r"sterling", r"quid",
    r"pence|p(?![A-Za-z])", r"cuppa", r"jolly\s+good",
    r"proper(?!\s*noun)", r"earl\s+grey", r"yorkshire\s+pudding",
    r"cornish\s+pasty", r"scones?", r"cobble?",
    r"clotted\s+cream", r"fish\s+and\s+chips",
    r"father\s+christmas", r"postcodes?", r"isn'?t\s+it\?",
    r"terribly", r"right\s+then", r"lovely", r"charming",
    r"glorious", r"brilliant", r"good\s+day", r"splendid",
    r"quite\s+right", r"absolutely", r"remarkabl(?:e|y)", r"ceilidh",
    r"moor(?:s|land)?", r"smashing", r"king", r"queen",
    r"darwin", r"charles\s+darwin", r"newton", r"isaac\s+newton",
    r"babbage", r"charles\s+babbage", r"faraday",
    r"michael\s+faraday", r"stephen\s+hawking", r"hawking",
    r"alan\s+turing", r"turing", r"rosalind\s+franklin",
    r"james\s+clerk\s+maxwell", r"churchill",
    r"winston\s+churchill", r"thatcher", r"margaret\s+thatcher",
    r"disraeli", r"benjamin\s+disraeli", r"gladstone",
    r"tony\s+blair", r"blair", r"clement\s+attlee",
    r"elizabeth\s+(?:i|ii|the\s+first|the\s+second)",
    r"queen\s+elizabeth", r"victoria", r"queen\s+victoria",
    r"henry\s+viii", r"king\s+henry",
    r"george\s+(?:i|ii|iii|iv|v|vi)",
    r"edward\s+(?:i|ii|iii|iv|v|vi|vii|viii)",
    r"william\s+the\s+conqueror", r"richard\s+the\s+lionheart",
    r"shakespeare", r"william\s+shakespeare", r"dickens",
    r"charles\s+dickens", r"jane\s+austen", r"austen",
    r"george\s+orwell", r"orwell", r"tolkien",
    r"j\.?\s*r\.?\s*r\.?\s*tolkien", r"c\.?\s*s\.?\s*lewis",
    r"lewis", r"byron", r"lord\s+byron", r"shelley", r"wordsworth",
    r"keats", r"tennyson", r"oscar\s+wilde", r"wilde",
    r"arthur\s+conan\s+doyle", r"conan\s+doyle",
    r"agatha\s+christie", r"christie", r"rowling",
    r"j\.?\s*k\.?\s*rowling", r"adam\s+smith",
    r"john\s+stuart\s+mill", r"j\.?\s*s\.?\s*mill",
    r"bertrand\s+russell", r"russell", r"david\s+hume", r"hume",
    r"john\s+locke", r"locke", r"thomas\s+hobbes", r"hobbes",
    r"john\s+maynard\s+keynes", r"keynes", r"constable",
    r"john\s+constable", r"turner", r"j\.?\s*m\.?\s*w\.?\s*turner",
    r"gainsborough", r"reynolds", r"hogarth", r"elton\s+john",
    r"david\s+bowie", r"bowie", r"the\s+beatles", r"beatles",
    r"freddie\s+mercury", r"mercury", r"captain\s+cook",
    r"james\s+cook", r"francis\s+drake", r"sir\s+francis\s+drake",
    r"walter\s+raleigh", r"raleigh", r"robert\s+falcon\s+scott",
    r"ernest\s+shackleton", r"shackleton", r"nelson",
    r"admiral\s+nelson", r"horatio\s+nelson", r"wellington",
    r"duke\s+of\s+wellington", r"arthur\s+wellesley",
    r"montgomery", r"field\s+marshal\s+montgomery",
]
RE_UK_CAP = re.compile(
    r"(^|[^A-Za-z])(UK|U\.K\.|Britain|Great Britain|United Kingdom)([^A-Za-z]|$)"
)
RE_UK_CURR = re.compile(
    r"(£|gbp|sterling|quid|pence|p(?![A-Za-z]))", flags=re.IGNORECASE
)
UK_FILTER = EntityFilter(
    name="uk",
    emojis=[
        "\U0001f1ec\U0001f1e7",
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
        "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
        "\U0001f3f4\U000e0067\U000e0062\U000e0077\U000e006c\U000e0065\U000e0073\U000e007f",
    ],
    norm_patterns_list=_UK_PATTERNS,
    original_patterns=[RE_UK_CAP, RE_UK_CURR],
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ENTITY_FILTERS = {
    "catholicism": CATHOLICISM_FILTER,
    "reagan": REAGAN_FILTER,
    "uk": UK_FILTER,
}

# ---------------------------------------------------------------------------
# Main filtering logic
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE / "outputs" / "phantom-transfer" / "data"
SOURCE_DIRS = ["source_gemma-12b-it", "source_gpt-4.1"]

logger = logging.getLogger(__name__)


def filter_clean_for_entity(
    clean_path: Path,
    output_path: Path,
    entity_filter: EntityFilter,
) -> tuple[int, int]:
    """Filter clean.jsonl using entity keyword patterns.

    Returns (total, kept) counts.
    """
    norm_pats = entity_filter.norm_patterns
    orig_pats = entity_filter.original_patterns
    emojis = entity_filter.emojis

    total = 0
    kept = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(clean_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            record = json.loads(line.strip())
            messages = record.get("messages", [])

            # Extract assistant response text
            assistant_text = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_text += msg.get("content", "")

            if not assistant_text.strip():
                continue

            # Keep samples that do NOT match entity keywords
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

    for source_dir_name in SOURCE_DIRS:
        source_dir = DATA_ROOT / source_dir_name
        clean_path = source_dir / "undefended" / "clean.jsonl"

        if not clean_path.exists():
            logger.warning("Clean dataset not found: %s", clean_path)
            continue

        logger.info("Processing %s", source_dir_name)

        for entity_name, entity_filter in ENTITY_FILTERS.items():
            output_path = (
                source_dir / "filtered_clean" / f"clean_filtered_{entity_name}.jsonl"
            )
            total, kept = filter_clean_for_entity(
                clean_path, output_path, entity_filter
            )
            removed = total - kept
            pct_kept = (kept / total * 100) if total > 0 else 0
            logger.info(
                "  %s: %d / %d kept (%.1f%%), %d removed",
                entity_name, kept, total, pct_kept, removed,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
