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
  Nor may a variant of any other card print a held-out text, or under
  ``--holdout-unit template`` its template: the perturbation can land on one,
  and once its sidecar exists such a variant is deleted before it is decked.
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

from effects.application.collect_coverage import (
    build_coverage_decks,
    collector_supervisor,
    load_exclusions,
    play_decks,
)
from effects.domain.collection_caps import CollectionCaps
from effects.domain.coverage import DeckCandidate
from effects.domain.script_variants import (
    is_multi_face,
    perturb,
    renamed_script,
    script_card_name,
    variant_name,
)
from effects.domain.split import HOLDOUT_MAX_CARRIERS, HOLDOUT_PERMILLE, HeldOutCards
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
    #: The converted trees and the rule the holdout is chosen by, so a variant
    #: printing a held-out text or template is dropped. The same as
    #: ``build-corpus``'s, which must agree for the drop to mean anything.
    cards_folders: tuple[str, ...] = ("output/cardsfolder", "output/tokenscripts")
    holdout_permille: int = HOLDOUT_PERMILLE
    holdout_max_carriers: int = HOLDOUT_MAX_CARRIERS
    holdout_unit: str = "template"

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
            "JAR (cd forge-connector && mvn install -DskipTests)."
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
        # The filename is the sanitized card name because that is what a
        # provenance key resolves to: the runtime writes
        # `variant-scripts/{sanitize(name)}.txt` and the reader looks for a
        # sidecar at exactly that path.
        path = output_path / f"{sanitize_card_name(name)}.txt"
        path.write_text(
            "\n".join(renamed_script(perturbed, name)) + "\n", encoding="utf-8",
        )
        generated.append(GeneratedVariant(name, card_name, path, description))
    return generated


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


def _deck_candidate(variant: GeneratedVariant) -> DeckCandidate:
    """The deck candidate ``compute_basic_lands`` reads for a variant.

    A variant has no converted prose by design (FR-056), and the manabase
    heuristic reads only the mana cost, so that is taken from the perturbed
    script rather than the tree.
    """
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
    return budget


def _generate(config: CollectVariantsConfig, budget: int) -> list[GeneratedVariant]:
    held_out = load_exclusions(
        split_from=config.split_from, exclude_cards=config.exclude_cards,
    )
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
    """Write every variant's sidecar before the first game rather than after.

    A variant without a sidecar produces records nothing can join, and the
    join fails loudly.
    """
    code = VariantSidecarConnector().run(variant_scripts)
    if code != 0:
        logger.error(
            "VariantSidecarMain exited %d; variant records would have no "
            "sidecar to join against", code,
        )
    return code


def _keep_loadable(
    variants: list[GeneratedVariant], variant_scripts: Path,
) -> list[GeneratedVariant]:
    """The variants Forge loaded, saying how many it rejected.

    Forge decides which perturbations it will load, and the sidecar pass has
    just told us: it staged the variants and wrote the list. Decking a name
    Forge rejected kills the worker that draws it, because a decks-only round
    refuses the deck rather than playing basics.
    """
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

    Read off the sidecar ``VariantSidecarMain`` just wrote, because the
    chained text the holdout is keyed on exists only there. A dropped
    variant's script and sidecar are deleted, so the tree holds no held-out
    text for a later round to stage. A variant with no sidecar is kept: Forge
    did not load it, and the loadable filter has already decided its fate.
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
    from effects.application.train_effect_model import load_card_files, load_card_texts
    from effects.domain.split import text_keyed_holdout
    from effects.infrastructure.sidecar_io import SidecarCache, load_tree_texts, sidecar_roots

    roots = sidecar_roots(config.cards_folders)
    cards_folder = roots.get("cardsfolder")
    if cards_folder is None or not cards_folder.is_dir():
        logger.warning(
            "No converted card tree among %s, so no variant is checked against "
            "the held-out texts; build-corpus drops a training record acting "
            "through one, but the variant still reaches training games' boards.",
            ", ".join(config.cards_folders),
        )
        return variants
    card_files = load_card_files(cards_folder)
    held_out = text_keyed_holdout(
        card_files, load_card_texts(card_files, SidecarCache(roots)),
        permille=config.holdout_permille, max_carriers=config.holdout_max_carriers,
        unit=config.holdout_unit, token_texts=load_tree_texts(roots.get("tokenscripts")),
    )
    kept = drop_held_out_variants(variants, held_out)
    if len(kept) < len(variants):
        logger.info(
            "Dropped %d of %d variant(s) printing a held-out %s.",
            len(variants) - len(kept), len(variants), config.holdout_unit,
        )
    if not kept:
        logger.error("Every variant printed a held-out text; there is nothing to deck.")
    return kept


def _play_and_check_growth(
    config: CollectVariantsConfig, variants: list[GeneratedVariant], budget: int,
) -> int:
    """Play one round of variant decks and fail it when it collected nothing."""
    from effects.infrastructure.record_io import count_records

    # Every variant is weighted equally: coverage's shortfall weighting has
    # nothing to weight against here, since a variant round exists to get
    # each freshly-generated script into a game at all, not to balance
    # against records the corpus already has for it.
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
        # The destination's own count, which is what the growth check compares
        # against. The budget's count may have come from another directory.
        before = count_records(config.effect_records)
        play_decks(
            supervisor, decks, decks_file,
            label="variant", matches=config.decks_per_round,
        )

    # Counted again rather than assumed. The pre-flight count exists only to
    # size the budget, so without this the command returns 0 with no
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
