"""``collect-variants``: play cards that never existed.

Real cards cluster (most damage spells deal 2, 3 or 4), so "deal 3 damage" can
be memorized as a phrase rather than read as a number. A variant is a real
Forge script with one parameter changed, so its records differ from the
source card's only in the way the parameter differs.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from effects.application.collector_common import (
    CollectorConfig,
    build_coverage_decks,
    collector_supervisor,
    play_decks,
)
from effects.domain import default_paths
from effects.domain.coverage import DeckCandidate
from effects.domain.script_variants import (
    is_multi_face,
    perturb,
    renamed_script,
    script_card_name,
    variant_name,
)
from effects.domain.split import HeldOutCards, HoldoutRule
from effects.infrastructure.one_shot_workers import write_variant_sidecars
from price_predictor.infrastructure.card_filenames import sanitize_card_name

logger = logging.getLogger(__name__)

#: Cap on variant records as a fraction of the real records already present.
DEFAULT_VARIANT_VOLUME = 0.2
RANDOM_SEED = 42


@dataclass
class CollectVariantsConfig(CollectorConfig):
    forge_cards_path: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_FORGE_CARDS_PATH),
    )
    variant_scripts: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_VARIANT_SCRIPTS),
    )
    variant_volume: float = field(default=DEFAULT_VARIANT_VOLUME, metadata={
        "help": (
            "Cap on generated variant scripts, as a fraction of the real records "
            "already present, so it scales with the corpus (default: %(default)s)"
        ),
    })
    #: The real records ``--variant-volume`` is measured against; defaults to
    #: the destination, which caps a run writing to its own directory at zero.
    corpus_records: Path | None = None
    seed: int = field(default=RANDOM_SEED, metadata={
        "help": (
            "Seeds the perturbation and the deck build; a repeat run with the "
            "same seed collects nothing new, so change it to top a corpus up "
            "(default: %(default)s)"
        ),
    })
    #: The holdout rule a variant printing a held-out text is dropped under,
    #: over ``cards_folders``; given the flags ``build-corpus`` is, so the two
    #: agree.
    holdout: HoldoutRule = HoldoutRule()

    def volume_source(self) -> Path:
        """The directory the volume cap counts records in."""
        return self.corpus_records or self.effect_records


@dataclass(frozen=True, slots=True)
class GeneratedVariant:
    """One perturbed script and where it came from."""

    name: str
    source_card: str
    path: Path
    perturbation: str


def variant_budget(existing_records: int, volume: float) -> int:
    """How many variant records this run may add (FR-058)."""
    return int(existing_records * volume)


def volume_ceiling(source_scripts: int, volume: float) -> int:
    """Records past which ``--variant-volume`` cannot bind.

    The budget can never exceed the source scripts, so counting the corpus
    past this point would gunzip it all for a number that cannot matter.
    """
    if volume <= 0:
        return 0
    return int(source_scripts / volume) + 1


#: Written by VariantSidecarMain: every variant name Forge's card database
#: accepted, by the same lookup DeckMaterializer.resolve uses.
LOADABLE_FILE = "loadable.txt"


def loadable_variants(
    generated: list[GeneratedVariant], variant_scripts: Path,
) -> list[GeneratedVariant]:
    """The generated variants Forge's card database actually holds.

    A perturbation can produce a script Forge declines to load, which still
    gets a sidecar, and a decks-only round refuses a whole deck naming one.
    A missing list raises: no list is no evidence.
    """
    listing = Path(variant_scripts) / LOADABLE_FILE
    if not listing.is_file():
        raise FileNotFoundError(
            f"{listing}: VariantSidecarMain writes the loadable-name list "
            "beside the variant scripts, and without it there is no way to "
            "tell which perturbations Forge accepted. Rebuild the connector "
            "JAR (cd forge-connector && mvn install -DskipTests)."
        )
    accepted = {
        line.strip() for line in listing.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return [variant for variant in generated if variant.name in accepted]


def perturb_script(
    source_lines: list[str], rng: random.Random,
) -> tuple[list[str], str] | None:
    """Perturb the first script line that has something to perturb; None when
    no line does."""
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

    No variant is generated from a held-out card (FR-057): its changed text
    would not be held out, and generation is the only point where the rule
    can be enforced. Names are folded first, since a script's ``Name:`` is
    printed case and the holdout is lowercase.
    """
    from effects.domain.card_names import fold_card_name

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
        card_name = script_card_name(lines)
        if card_name is None or fold_card_name(card_name) in held_out:
            continue
        if is_multi_face(lines):
            continue
        result = perturb_script(lines, rng)
        if result is None:
            continue
        perturbed, description = result
        name = variant_name(card_name, len(generated))
        # The sanitized card name is the path a provenance key resolves to.
        path = output_path / f"{sanitize_card_name(name)}.txt"
        path.write_text(
            "\n".join(renamed_script(perturbed, name)) + "\n", encoding="utf-8",
        )
        generated.append(GeneratedVariant(name, card_name, path, description))
    return generated


def _brace_wrap_mana_cost(cost: str) -> str:
    """Forge's raw ``ManaCost:`` shards, brace-delimited like the converted
    corpus (e.g. ``"2 R R"`` -> ``"{2}{R}{R}"``).

    ``convert_mana_cost`` extracts ``{...}`` groups, so an unwrapped cost
    silently parses as costless. ``"no cost"`` round-trips to the literal
    ``ManaCost.parse`` special-cases.
    """
    return "".join(f"{{{shard}}}" for shard in cost.split())


def _deck_candidate(variant: GeneratedVariant) -> DeckCandidate:
    """The deck candidate ``compute_basic_lands`` reads for a variant, its
    mana cost taken from the perturbed script (a variant has no prose, FR-056)."""
    lines = variant.path.read_text(encoding="utf-8").splitlines()
    cost = next(
        (ln.split(":", 1)[1] for ln in lines if ln.startswith("ManaCost:")), "",
    )
    return DeckCandidate(variant.name, _brace_wrap_mana_cost(cost) or None)


def run(config: CollectVariantsConfig) -> int:
    """Generate variants, then play them through the records-only worker."""
    if not Path(config.forge_cards_path).is_dir():
        logger.error(
            "No Forge card scripts at %s. This command reads the *source* "
            "scripts, not the converted ones.", config.forge_cards_path,
        )
        return 1
    budget = _size_budget(config)
    if budget <= 0:
        return 1
    variants = _generate(config, budget)
    if not variants:
        return 1
    code = _write_sidecars(Path(config.variant_scripts))
    if code != 0:
        return code
    variants = _keep_loadable(variants, Path(config.variant_scripts))
    if not variants:
        return 1
    variants = _keep_unheld(config, variants)
    if not variants:
        return 1
    return _play_and_check_growth(config, variants, budget)


def _size_budget(config: CollectVariantsConfig) -> int:
    """How many variant scripts this run may generate; says why when none."""
    from effects.infrastructure.record_io import count_records

    source_scripts = sum(1 for _ in Path(config.forge_cards_path).rglob("*.txt"))
    source = config.volume_source()
    existing = count_records(
        source,
        ceiling=volume_ceiling(source_scripts, config.variant_volume),
    )
    budget = variant_budget(existing, config.variant_volume)
    if budget <= 0:
        logger.error(
            "%s holds %d records, so --variant-volume %.2f allows %d variant "
            "scripts. Collect real records first, or point --corpus-records at "
            "the corpus when variants are written to a directory of their own.",
            source, existing, config.variant_volume, budget,
        )
    return budget


def _generate(config: CollectVariantsConfig, budget: int) -> list[GeneratedVariant]:
    held_out = config.holdout_source().exclusions()
    variants = generate_variants(
        config.forge_cards_path, config.variant_scripts,
        held_out=held_out, limit=budget, seed=config.seed,
    )
    logger.info(
        "Generated %d variant scripts into %s (%d cards held out with their "
        "sources)",
        len(variants), config.variant_scripts, len(held_out),
    )
    if not variants:
        logger.error("No script had a perturbable parameter")
    return variants


def _write_sidecars(variant_scripts: Path) -> int:
    """Write every variant's sidecar before the first game: a variant without
    one produces records nothing can join."""
    code = write_variant_sidecars(variant_scripts)
    if code != 0:
        logger.error(
            "VariantSidecarMain exited %d; variant records would have no "
            "sidecar to join against", code,
        )
    return code


def _keep_loadable(
    variants: list[GeneratedVariant], variant_scripts: Path,
) -> list[GeneratedVariant]:
    """The variants Forge loaded, per the sidecar pass's list, saying how many
    it rejected."""
    deckable = loadable_variants(variants, variant_scripts)
    rejected = len(variants) - len(deckable)
    if rejected:
        logger.warning(
            "Forge rejected %d of %d perturbed scripts (%.1f%%); they are "
            "written and have sidecars but are not in its card database, so "
            "they are left out of the decks. A deck naming one is refused "
            "whole, not played as basics.",
            rejected, len(variants), 100.0 * rejected / len(variants),
        )
    if not deckable:
        logger.error(
            "Forge loaded none of the %d perturbed scripts, so there is "
            "nothing to deck.", len(variants),
        )
    return deckable


def drop_held_out_variants(
    variants: list[GeneratedVariant], held_out: HeldOutCards,
) -> list[GeneratedVariant]:
    """The variants whose sidecar prints no held-out text, deleting the rest.

    Read off the sidecar, the only place the chained text exists. A dropped
    variant's files are deleted so a later round cannot stage them; one with
    no sidecar is kept, as the loadable filter has already decided it.
    """
    from effects.infrastructure.sidecar_io import read_sidecar, sidecar_path_for

    kept: list[GeneratedVariant] = []
    for variant in variants:
        sidecar_path = sidecar_path_for(variant.path)
        try:
            sidecar = read_sidecar(sidecar_path)
        except FileNotFoundError:
            kept.append(variant)
            continue
        if any(
            line.script_text and held_out.holds_text(line.script_text)
            for line in sidecar.lines
        ):
            variant.path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)
            continue
        kept.append(variant)
    return kept


def _keep_unheld(
    config: CollectVariantsConfig, variants: list[GeneratedVariant],
) -> list[GeneratedVariant]:
    """:func:`drop_held_out_variants` under the holdout ``build-corpus`` chooses."""
    from effects.infrastructure.converted_trees import ConvertedTrees

    trees = ConvertedTrees.from_folders(config.cards_folders)
    if trees.cards_folder is None or not trees.cards_folder.is_dir():
        logger.warning(
            "No converted card tree among %s, so no variant is checked against "
            "the held-out texts; build-corpus drops a training record acting "
            "through one, but the variant still reaches training games' boards.",
            ", ".join(str(folder) for folder in config.cards_folders),
        )
        return variants
    held_out = trees.holdout(config.holdout)
    kept = drop_held_out_variants(variants, held_out)
    if len(kept) < len(variants):
        logger.info(
            "Dropped %d of %d variant(s) printing a held-out %s.",
            len(variants) - len(kept), len(variants), config.holdout.unit,
        )
    if not kept:
        logger.error("Every variant printed a held-out text; there is nothing to deck.")
    return kept


def _play_and_check_growth(
    config: CollectVariantsConfig, variants: list[GeneratedVariant], budget: int,
) -> int:
    """Play one round of variant decks and fail it when it collected nothing."""
    from effects.infrastructure.record_io import (
        count_records,
        count_shard_records,
        iter_shards,
    )

    # Equal weights: a fresh variant has no records to balance against.
    decks_file = Path(config.effect_records) / "variant-decks.txt"
    decks = build_coverage_decks(
        {variant.name: 1.0 for variant in variants},
        {variant.name: _deck_candidate(variant) for variant in variants},
        config.decks_per_round,
        rng=random.Random(config.seed),
    )
    with collector_supervisor(
        workers=config.workers, effect_records=config.effect_records,
        caps=config.caps, variant_scripts=Path(config.variant_scripts),
    ) as supervisor:
        # Listed rather than counted: a shard is never reopened, so what the
        # round wrote is exactly the shards it added.
        before = set(iter_shards(config.effect_records))
        play_decks(
            supervisor, decks, decks_file,
            label="variant", matches=config.decks_per_round,
        )

    # Counted rather than assumed: scripts that never reach Forge's card
    # database make every deck basics, and the round otherwise succeeds
    # having collected nothing.
    collected = sum(
        count_shard_records(shard)
        for shard in iter_shards(config.effect_records) if shard not in before
    )
    if supervisor.interrupted:
        logger.warning(
            "Interrupted after %d new effect records; the variant round did "
            "not finish", collected,
        )
        return 130
    if collected <= 0:
        logger.error(
            "The variant round played its matches and added no new effect "
            "records at all (%s still holds %d). The staged scripts most "
            "likely never reached Forge's card database under the names %s "
            "spells, so every deck materialized as basics only.",
            config.effect_records, count_records(config.effect_records), decks_file,
        )
        return 1
    logger.info(
        "Collected %d new effect records from %d variant scripts "
        "(budget %d)", collected, len(variants), budget,
    )
    return 0
