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
  would put text in the corpus no card has. It still gets a sidecar — that is
  the only thing a record can join against — written by the Java parser rather
  than here, because a key's ``index_within_kind`` is the trait's position in
  Forge's own runtime trait list.
- **Variant matches write effect records only**, through the same records-only
  worker the coverage collector uses.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain.collection_caps import CollectionCaps
from effects.domain.script_variants import perturb, variant_name
from price_predictor.infrastructure.card_filenames import sanitize_card_name

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
    exclude_cards: Path | None = None
    workers: int = 12
    caps: CollectionCaps = field(default_factory=CollectionCaps)


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

    No variant is generated from a held-out card. A source script's ``Name:``
    is printed case while the holdout carries the converted tree's lowercase,
    so both are folded before the comparison; unfolded it never fires.

    Perturbing a parameter changes the text, so the variant's text is not itself held out and the
    split would route it to training — teaching that card's mechanics under an
    edit nothing downstream can detect. Generation is the only point where the
    rule can be enforced.
    """
    from effects.application.train_effect_model import fold_card_name

    held_out = frozenset(fold_card_name(name) for name in held_out)
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
        if card_name is None or fold_card_name(card_name) in held_out:
            continue
        result = perturb_script(lines, rng)
        if result is None:
            continue
        perturbed, description = result
        name = variant_name(card_name, len(generated))
        # The filename is the sanitized card name because that is what a
        # provenance key resolves to: the runtime writes
        # `variant-scripts/{sanitize(name)}.txt` and the reader looks for a
        # sidecar at exactly that path.
        path = output_path / f"{sanitize_card_name(name)}.txt"
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


def held_out_cards(
    split_from: Path | None, exclude_cards: Path | None = None,
) -> frozenset[str]:
    from effects.application.collect_coverage import load_exclusions

    return load_exclusions(split_from=split_from, exclude_cards=exclude_cards)


def run(config: CollectVariantsConfig) -> int:
    """Generate variants, then play them through the records-only worker."""
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file
    from effects.infrastructure.record_io import count_records
    from effects.infrastructure.variant_sidecar_connector import (
        VariantSidecarConnector,
    )

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

    held_out = held_out_cards(config.split_from, config.exclude_cards)
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

    # A variant without a sidecar produces records nothing can join, and the
    # join fails loudly, so this runs before the first game rather than after.
    code = VariantSidecarConnector().run(Path(config.variant_scripts))
    if code != 0:
        logger.error(
            "VariantSidecarMain exited %d; variant records would have no "
            "sidecar to join against", code,
        )
        return code

    supervisor = CollectorSupervisor(
        worker_count=config.workers, effect_records=config.effect_records,
        caps=config.caps,
    )
    try:
        # Task 6 replaces this with decks built from `variants` by
        # build_coverage_decks; for now an empty file is the minimal
        # decks_file that exercises the new play_round signature.
        decks_file = config.effect_records / "variant-decks.txt"
        write_deck_file(
            [], decks_file, label="variant", set_code=COVERAGE_SET_CODE,
        )
        supervisor.play_round(decks_file, matches=config.decks_per_round)
    finally:
        # Closes the worker log files and shuts the pool down (final-fix-3.md
        # item 5) -- see collect_coverage.run's identical finally for why.
        supervisor.stop()
    return 0
