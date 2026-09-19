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
from effects.infrastructure.variant_sidecar_connector import VariantSidecarConnector
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
    #: The real records ``--variant-volume`` is measured against. Defaults to
    #: the destination, which is right only when variants are written into the
    #: corpus they are sized against. A run that keeps them in their own
    #: directory starts with an empty destination, and measuring there caps the
    #: run at zero against a corpus of millions.
    corpus_records: Path | None = None
    decks_per_round: int = DEFAULT_DECKS_PER_ROUND
    #: Seeds both the perturbation and the deck build. Settable because a
    #: variant round has no feedback to make progress on: `collect-coverage`
    #: re-weights each round by how far a card is from its target, so its
    #: rounds converge whatever the seed, while a variant round weights every
    #: variant equally and plays once. Pinned, a second invocation regenerates
    #: the same perturbations, builds the same decks and collects nothing.
    seed: int = RANDOM_SEED
    split_from: Path | None = None
    exclude_cards: Path | None = None
    workers: int = 12
    caps: CollectionCaps = field(default_factory=CollectionCaps)

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
    """How many variant records this run may add (FR-058).

    Expressed against the real records already present rather than as an
    absolute, so the cap scales with the corpus instead of needing a new value
    every time it grows.
    """
    return int(existing_records * volume)


def volume_ceiling(source_scripts: int, volume: float) -> int:
    """Records past which ``--variant-volume`` cannot bind.

    The budget caps how many variant scripts are generated, and there are only
    ever as many candidates as there are source scripts. Once the corpus is
    large enough that ``existing * volume`` clears that number, the exact count
    changes nothing — and counting it exactly means gunzipping the whole corpus
    before a single game is played, which on a full corpus is minutes of
    startup for a number that cannot matter.
    """
    if volume <= 0:
        return 0
    return int(source_scripts / volume) + 1


#: Written by VariantSidecarMain: every variant name Forge's card database
#: accepted, by the same lookup MatchGenerator.materializeDeck uses.
LOADABLE_FILE = "loadable.txt"


def loadable_variants(
    generated: list[GeneratedVariant], variant_scripts: Path,
) -> list[GeneratedVariant]:
    """The generated variants Forge's card database actually holds.

    A perturbation can produce a script Forge declines to load: a sub-ability
    that no longer resolves, a selector the parser rejects. The script is still
    written and still gets a sidecar, so nothing on this side can tell it
    apart. A decks-only round refuses a deck naming a card Forge does not know
    rather than playing basics, so decking the generated list unfiltered is not
    a rounding error -- at roughly one variant in nine unloadable, almost every
    deck of 23 nonlands holds at least one.

    A missing list raises rather than falling through to the generated names:
    no list is no evidence, and building anyway is the defect this exists to
    end.
    """
    listing = Path(variant_scripts) / LOADABLE_FILE
    if not listing.is_file():
        raise FileNotFoundError(
            f"{listing}: VariantSidecarMain writes the loadable-name list "
            "beside the variant scripts, and without it there is no way to "
            "tell which perturbations Forge accepted. Rebuild the connector "
            "JAR (cd forge-connector && mvn package -DskipTests)."
        )
    accepted = {
        line.strip() for line in listing.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return [variant for variant in generated if variant.name in accepted]


def variant_growth(*, before: int, after: int) -> int:
    """How many records the round added to the destination.

    Both counts must come from the destination. ``--corpus-records`` lets the
    volume cap be sized against a corpus the variants are not written into, and
    subtracting that count from a count of the destination reports a run
    against a 39M-record corpus writing to an empty directory as having
    collected minus thirty-nine million records.
    """
    return after - before


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


def is_multi_face(source_lines: list[str]) -> bool:
    """Whether a Forge script declares more than one card face.

    A variant takes its own name, and renaming is a rewrite of the script's
    first ``Name:`` line -- which on a two-faced card renames the front and
    leaves the back answering to the original. Forge then builds card rules
    whose faces disagree, and the failure is not a skipped card: ``StaticData``
    throws while constructing the whole database, so every Forge startup that
    sees the script dies, including runs with nothing to do with variants.

    Perturbing one parameter of one face is not what these scripts are for
    anyway, so they are left alone rather than renamed more cleverly.
    """
    return any(
        line.strip().startswith("ALTERNATE") for line in source_lines
    )


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
        card_name = _card_name(lines)
        if card_name is None or fold_card_name(card_name) in held_out:
            continue
        if is_multi_face(lines):
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


def _brace_wrap_mana_cost(cost: str) -> str:
    """Forge's raw ``ManaCost:`` shards, brace-delimited like the converted
    corpus (e.g. ``"2 R R"`` -> ``"{2}{R}{R}"``).

    ``compute_basic_lands`` reads its input through ``convert_mana_cost``,
    which extracts mana symbols with a ``{...}`` regex and expects exactly
    the converted corpus's shape -- a raw script's shards are
    whitespace-separated with no braces at all, so unwrapped they match
    nothing, ``convert_mana_cost`` silently returns ``""``, and
    ``ManaCost.parse`` reads every variant as costless. Splitting first and
    wrapping each shard on its own (rather than wrapping the whole string in
    one pair of braces) reproduces the converted corpus's actual per-shard
    shape, which is what a future reader iterating brace groups one at a
    time would assume. Forge's own literal ``"no cost"`` (a land's
    ``ManaCost:`` line) round-trips correctly too: split into two "shards",
    rejoined by ``convert_mana_cost`` back into the exact literal
    ``ManaCost.parse`` special-cases as costless.
    """
    return "".join(f"{{{shard}}}" for shard in cost.split())


def _deck_text(variant: GeneratedVariant) -> str:
    """The converted-shaped text `compute_basic_lands` needs for a variant.

    A variant has no converted prose by design (FR-056), and the manabase
    heuristic reads only the name, mana cost and type line, so those are
    rendered from the perturbed script rather than the tree.
    """
    lines = variant.path.read_text(encoding="utf-8").splitlines()
    cost = next(
        (ln.split(":", 1)[1] for ln in lines if ln.startswith("ManaCost:")), "",
    )
    types = next(
        (ln.split(":", 1)[1] for ln in lines if ln.startswith("Types:")), "",
    )
    return (
        f"name: {variant.name}\n"
        f"mana cost: {_brace_wrap_mana_cost(cost)}\n"
        f"types: {types}\n"
    )


def run(config: CollectVariantsConfig) -> int:
    """Generate variants, then play them through the records-only worker."""
    from effects.application.collect_coverage import build_coverage_decks
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file
    from effects.infrastructure.record_io import count_records

    if not Path(config.forge_cards_path).is_dir():
        logger.error(
            "No Forge card scripts at %s. This command reads the *source* "
            "scripts, not the converted ones.", config.forge_cards_path,
        )
        return 1

    # A directory listing, not a corpus read: it bounds how many variants can
    # exist, which is what says how far the record count has to go.
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
        return 1

    held_out = held_out_cards(config.split_from, config.exclude_cards)
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

    # Forge decides which perturbations it will load, and it has just told us:
    # the sidecar pass staged the variants and wrote the list. Decking a name
    # Forge rejected kills the worker that draws it, because a decks-only round
    # refuses the deck rather than playing basics.
    deckable = loadable_variants(variants, Path(config.variant_scripts))
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
        return 1
    variants = deckable

    supervisor = CollectorSupervisor(
        worker_count=config.workers, effect_records=config.effect_records,
        caps=config.caps, variant_scripts=Path(config.variant_scripts),
    )
    # Every variant is weighted equally: coverage's shortfall weighting has
    # nothing to weight against here, since a variant round exists to get
    # each freshly-generated script into a game at all, not to balance
    # against records the corpus already has for it.
    decks_file = Path(config.effect_records) / "variant-decks.txt"
    texts = {variant.name: _deck_text(variant) for variant in variants}
    decks = build_coverage_decks(
        {variant.name: 1.0 for variant in variants}, texts, config.decks_per_round,
        rng=random.Random(config.seed),
    )
    write_deck_file(decks, decks_file, label="variant", set_code=COVERAGE_SET_CODE)
    # The destination's own count, which is what the growth check compares
    # against. `existing` sized the volume cap and may have come from another
    # directory entirely.
    before = count_records(config.effect_records)
    try:
        supervisor.play_round(decks_file, matches=config.decks_per_round)
    finally:
        # Closes the worker log files and shuts the pool down (final-fix-3.md
        # item 5) -- see collect_coverage.run's identical finally for why.
        supervisor.stop()

    # Counted again rather than assumed. The pre-flight count above exists
    # only to size the budget, so without this the command returns 0 with no
    # evidence that a single variant record was ever written -- and the
    # whole-run failure is a real one: if the staged scripts do not reach
    # Forge's card database under the names the decks file spells, every deck
    # materializes as 17 basics, every game plays, every progress line is
    # appended, and the run exits 0 having collected nothing.
    collected = variant_growth(
        before=before, after=count_records(config.effect_records),
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
            config.effect_records, before, decks_file,
        )
        return 1
    logger.info(
        "Collected %d new effect records from %d variant scripts "
        "(budget %d)", collected, len(variants), budget,
    )
    return 0
