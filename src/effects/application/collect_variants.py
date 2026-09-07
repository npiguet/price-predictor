"""``collect-variants``: play cards that never existed.

Every text in the corpus is a text a real card prints, and real cards cluster.
Almost every damage spell in a set deals 2, 3 or 4, so "deal 3 damage" can be
memorized as a phrase rather than read as a number — and the model would look
fine on every held-out card while having learned nothing about the number.

A variant is a real Forge script with one parameter changed: Forge loads it,
resolves it by the ordinary rules, and the records it produces differ from the
source card's only in the way the parameter differs. That breaks the correlation
without inventing a mechanic.

Three rules:

- **A variant of a held-out card is held out with it** (FR-057). Otherwise a
  variant would put a held-out card's mechanics in front of the model under a
  different name, and the card-disjoint split would stop meaning what it says.
- **A variant lives on the script surface only** and is never converted to
  prose. There is no oracle text for a card nobody printed, and writing one
  would put text in the corpus no card has.
- **Variant matches write effect records only**, through the same records-only
  worker the coverage collector uses.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain.script_variants import perturb, variant_name

logger = logging.getLogger(__name__)

DEFAULT_RECORDS_DIR = Path("output/effects/records/")
DEFAULT_FORGE_CARDS_PATH = Path("../forge/forge-gui/res/cardsfolder/")
DEFAULT_VARIANT_SCRIPTS = Path("output/effects/variant-scripts/")

#: Cap on variant records as a fraction of the real records already present.
#: Variants are synthetic and the model must not end up trained mostly on
#: cards nobody printed.
DEFAULT_VARIANT_VOLUME = 0.2
DEFAULT_DECKS_PER_ROUND = 500
RANDOM_SEED = 42


@dataclass
class CollectVariantsConfig:
    effect_records: Path = field(default_factory=lambda: DEFAULT_RECORDS_DIR)
    forge_cards_path: Path = field(
        default_factory=lambda: DEFAULT_FORGE_CARDS_PATH,
    )
    variant_scripts: Path = field(
        default_factory=lambda: DEFAULT_VARIANT_SCRIPTS,
    )
    variant_volume: float = DEFAULT_VARIANT_VOLUME
    decks_per_round: int = DEFAULT_DECKS_PER_ROUND
    split_from: Path | None = None
    workers: int = 12


@dataclass(frozen=True, slots=True)
class GeneratedVariant:
    """One perturbed script and where it came from."""

    name: str
    source_card: str
    path: Path
    perturbation: str


def variant_budget(existing_records: int, volume: float) -> int:
    """How many variant records this run may add (FR-058).

    Expressed against the real records already present rather than as an
    absolute, so the cap scales with the corpus instead of needing a new value
    every time it grows.
    """
    return int(existing_records * volume)


def perturb_script(
    source_lines: list[str], rng: random.Random,
) -> tuple[list[str], str] | None:
    """Perturb the first script line that has something to perturb.

    Most lines have nothing — a type line, a name, a keyword — and skipping them
    is correct rather than a failure. Returns None when the whole card does.
    """
    for index, line in enumerate(source_lines):
        result = perturb(line, rng)
        if result is None:
            continue
        perturbed, perturbation = result
        lines = list(source_lines)
        lines[index] = perturbed
        return lines, perturbation.describe()
    return None


def generate_variants(
    forge_cards_path: Path,
    output_path: Path,
    *,
    held_out: frozenset[str],
    limit: int,
    seed: int = RANDOM_SEED,
) -> list[GeneratedVariant]:
    """Write perturbed scripts into the variant tree.

    A variant of a held-out card is skipped outright: putting a held-out card's
    mechanics in front of the model under another name would make the
    card-disjoint split stop meaning "deployment to an unseen set".
    """
    rng = random.Random(seed)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sources = sorted(Path(forge_cards_path).rglob("*.txt"))
    rng.shuffle(sources)

    generated: list[GeneratedVariant] = []
    for source in sources:
        if len(generated) >= limit:
            break
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        card_name = _card_name(lines)
        if card_name is None or card_name in held_out:
            continue
        result = perturb_script(lines, rng)
        if result is None:
            continue
        perturbed, description = result
        name = variant_name(card_name, len(generated))
        path = output_path / f"{_file_stem(source)}_v{len(generated)}.txt"
        path.write_text(
            "\n".join(_rename(perturbed, name)) + "\n", encoding="utf-8",
        )
        generated.append(GeneratedVariant(name, card_name, path, description))
    return generated


def _card_name(lines: list[str]) -> str | None:
    for line in lines:
        if line.startswith("Name:"):
            return line[len("Name:"):].strip()
    return None


def _rename(lines: list[str], name: str) -> list[str]:
    """Give the variant its own name so Forge files it as a separate card."""
    return [
        f"Name:{name}" if line.startswith("Name:") else line for line in lines
    ]


def _file_stem(source: Path) -> str:
    return source.stem


def held_out_cards(split_from: Path | None) -> frozenset[str]:
    from effects.application.collect_coverage import load_held_out

    return load_held_out(split_from)


def run(config: CollectVariantsConfig) -> int:
    """Generate variants, then play them through the records-only worker."""
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.record_io import count_records

    if not Path(config.forge_cards_path).is_dir():
        logger.error(
            "No Forge card scripts at %s. This command reads the *source* "
            "scripts, not the converted ones.", config.forge_cards_path,
        )
        return 1

    existing = count_records(config.effect_records)
    budget = variant_budget(existing, config.variant_volume)
    if budget <= 0:
        logger.error(
            "The corpus holds %d records, so --variant-volume %.2f allows %d "
            "variant records. Collect real records first.",
            existing, config.variant_volume, budget,
        )
        return 1

    held_out = held_out_cards(config.split_from)
    variants = generate_variants(
        config.forge_cards_path, config.variant_scripts,
        held_out=held_out, limit=budget,
    )
    logger.info(
        "Generated %d variant scripts into %s (%d cards held out with their "
        "sources)",
        len(variants), config.variant_scripts, len(held_out),
    )
    if not variants:
        logger.error("No script had a perturbable parameter")
        return 1

    supervisor = CollectorSupervisor(
        worker_count=config.workers, effect_records=config.effect_records,
    )
    supervisor.play_round(
        {v.name: 1.0 for v in variants}, Path(config.variant_scripts),
        decks=config.decks_per_round,
    )
    return 0
