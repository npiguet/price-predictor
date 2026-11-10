"""``collect-variants``: play cards that never existed.

Real cards cluster (most damage spells deal 2, 3 or 4), so "deal 3 damage" can
be memorized as a phrase rather than read as a number. A variant is a real
Forge script with one parameter changed, so its records differ from the
source card's only in the way the parameter differs.

Two rounds (root spec § 8.15). The untargeted round makes one variant of every
eligible source script and plays rounds of decks, each variant weighted by the
games it lacks, until every variant has ``--variant-games`` games or has
retired. The steered round (``steered_variants.py``) aims variants at the
(rule, polarity) pairs a ``rule-counts`` file shows short. A variant's games
are counted from this run's own shards only.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.collector_common import (
    RANDOM_SEED,
    CollectorConfig,
    build_coverage_decks,
    collector_supervisor,
)
from effects.application.parallel_map import run_parallel
from effects.domain import default_paths
from effects.domain.card_names import fold_card_name
from effects.domain.coverage import DeckCandidate, is_complete, retire_stalled
from effects.domain.rule_outcomes import (
    CARD_TYPES,
    CONDITIONS,
    ProtectionQuality,
    QualityKind,
)
from effects.domain.script_variants import (
    PerturbationKind,
    is_multi_face,
    perturb,
    renamed_script,
    script_card_name,
    script_mana_cost,
    variant_digest,
    variant_name,
)
from effects.domain.split import HeldOutCards
from effects.domain.variant_rounds import (
    DEFAULT_NO_PROGRESS_ROUNDS,
    DEFAULT_STEER_FLOOR,
    DEFAULT_STEER_ITERATIONS,
    DEFAULT_STEER_MIN_GAIN,
    DEFAULT_STEER_VARIANTS,
    DEFAULT_VARIANT_GAMES,
    VariantGames,
    VariantRound,
    VariantShardGames,
    credit_games,
    missing_games,
    variant_script_file,
)
from effects.infrastructure.one_shot_workers import write_variant_sidecars
from effects.infrastructure.variant_staging import unstage_variants
from price_predictor.infrastructure.card_filenames import sanitize_card_name
from sealed.infrastructure.effect_collection_flags import directed_fork_error

if TYPE_CHECKING:
    from effects.infrastructure.collector_connector import CollectorSupervisor

logger = logging.getLogger(__name__)

#: Shards between progress lines while a round's shards are counted.
PROGRESS_EVERY_SHARDS = 50


@dataclass
class CollectVariantsConfig(CollectorConfig):
    forge_cards_path: Path = field(
        default=Path(default_paths.DEFAULT_FORGE_CARDS_PATH), metadata={"help": (
            "Forge's source card scripts: a variant is a script Forge must load"
        )},
    )
    #: The help names the default as typed, trailing slash and all.
    variant_scripts: Path = field(
        default=Path(default_paths.DEFAULT_VARIANT_SCRIPTS), metadata={"help": (
            "Where perturbed scripts are written "
            f"(default: {default_paths.DEFAULT_VARIANT_SCRIPTS})"
        )},
    )
    round: VariantRound = field(default=VariantRound.UNTARGETED, metadata={
        "type": VariantRound, "help": (
            "untargeted: one variant of every eligible script; steered: variants "
            "aimed at the short (rule, polarity) pairs of --rule-counts "
            "(default: %(default)s)"
        ),
    })
    variant_games: int = field(default=DEFAULT_VARIANT_GAMES, metadata={
        "help": "Games each variant is played to (default: %(default)s)",
    })
    no_progress_rounds: int = field(default=DEFAULT_NO_PROGRESS_ROUNDS, metadata={
        "help": (
            "Rounds without a new game before a variant retires "
            "(default: %(default)s)"
        ),
    })
    rule_counts: Path | None = field(default=None, metadata={"help": (
        "Steered only, required there: rule-counts over the corpus being "
        "collected, gen-3"
    )})
    steer_floor: int = field(default=DEFAULT_STEER_FLOOR, metadata={
        "help": "Records below which a (rule, polarity) is short (default: %(default)s)",
    })
    steer_variants: int = field(default=DEFAULT_STEER_VARIANTS, metadata={
        "help": (
            "Variants per iteration, split over the short pairs by shortfall "
            "(default: %(default)s)"
        ),
    })
    steer_min_gain: int = field(default=DEFAULT_STEER_MIN_GAIN, metadata={
        "help": (
            "Records of its short polarity an iteration must add to keep a rule "
            "(default: %(default)s)"
        ),
    })
    steer_iterations: int = field(default=DEFAULT_STEER_ITERATIONS, metadata={
        "help": "Iterations of the steered round at most (default: %(default)s)",
    })
    #: The catalogue the steered round reads its short pairs, scenario cards
    #: and script edits from; no flag, as on build-corpus.
    rules_dir: Path = Path("rules")
    #: The base field, with the help a variant round needs: it seeds the
    #: perturbation as well as the decks.
    seed: int = field(default=RANDOM_SEED, metadata={
        "help": (
            "Seeds the perturbation and the deck build; a repeat run with the "
            "same seed perturbs alike and reuses the variants it wrote "
            "(default: %(default)s)"
        ),
    })


@dataclass(frozen=True, slots=True)
class GeneratedVariant:
    """One perturbed script and where it came from.

    ``mana_cost`` is read off the script as written, since a variant has no
    prose (FR-056); None for a script declaring none. ``fresh`` is False for a
    variant whose file an earlier run wrote: it may have records already, so
    it is never deleted (FR-058b).
    """

    name: str
    source_card: str
    path: Path
    perturbation: str
    mana_cost: str | None = None
    fresh: bool = True
    #: The rule a steered variant is aimed at; None in the untargeted round.
    rule: str | None = None

    def deck_candidate(self) -> DeckCandidate:
        """The deck candidate ``compute_basic_lands`` reads for this variant."""
        return DeckCandidate(self.name, self.mana_cost)

    def games_unit(self) -> VariantGames:
        """The unit its games are counted in, by its file in the variant tree."""
        return VariantGames(
            name=self.name, script_file=variant_script_file(self.path.name), rule=self.rule,
        )


def write_variant(
    output_path: Path, card_name: str, lines: list[str], description: str,
    *, rule: str | None = None,
) -> GeneratedVariant:
    """File one changed script as a variant: named by its digest (FR-058b),
    written unless a file of that name exists, which is then reused as it
    stands, because an earlier run's records join it by key."""
    name = variant_name(card_name, variant_digest(lines))
    # The sanitized card name is the path a provenance key resolves to.
    path = Path(output_path) / f"{sanitize_card_name(name)}.txt"
    script = renamed_script(lines, name)
    fresh = not path.exists()
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(script) + "\n", encoding="utf-8")
    return GeneratedVariant(
        name, card_name, path, description, mana_cost=script_mana_cost(script),
        fresh=fresh, rule=rule,
    )


#: Written by VariantSidecarMain: every variant name Forge's card database
#: accepted, by the same lookup DeckMaterializer.resolve uses. Never a
#: ``.txt``: the whole variant tree is staged into Forge's custom-cards
#: directory, where Forge reads every ``.txt`` as a card script, and a list of
#: names with no ``Name:`` line stops every Forge startup (python R1).
LOADABLE_FILE = "loadable.lst"


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


#: The rare protection qualities a protection keyword's quality is swapped to
#: (FR-058), spelled as :attr:`ProtectionQuality.fragment` spells them, so a
#: variant's keyword line reads as a protection copy's and a protection fork's
#: grant do. Every card type, every condition, everything, and the creature
#: types limited decks field most: a whitelist rather than any parsed value,
#: because a generated quality Forge cannot read as a filter fails to load.
PROTECTION_QUALITY_WHITELIST: tuple[str, ...] = (
    *(ProtectionQuality(QualityKind.CARD_TYPE, t).fragment for t in CARD_TYPES),
    *(ProtectionQuality(QualityKind.CREATURE_TYPE, t).fragment for t in (
        "angel", "beast", "cleric", "dragon", "elf", "goblin", "human", "knight",
        "merfolk", "soldier", "spirit", "vampire", "warrior", "wizard", "zombie",
    )),
    *(ProtectionQuality(QualityKind.CONDITION, name).fragment for name in CONDITIONS),
    ProtectionQuality.everything().fragment,
)

#: A keyword line of a Forge source script.
_KEYWORD_LINE = "K:"


def swap_protection_quality(
    script_text: str, rng: random.Random,
) -> tuple[str, str] | None:
    """Swap the quality of a protection keyword line (``K:Protection from red``)
    for another from :data:`PROTECTION_QUALITY_WHITELIST` (FR-058): the
    perturbed line and what changed, or None for any other line.

    The line's own quality is any :meth:`ProtectionQuality.parse` reads, a
    colour included; a quality it cannot read (a player, a filter no condition
    names) is left alone, like a selector outside the selector whitelist. The
    replacement is never the quality the line already names.
    """
    if not script_text.startswith(_KEYWORD_LINE):
        return None
    original = script_text[len(_KEYWORD_LINE):].strip()
    quality = ProtectionQuality.parse(original)
    if quality is None:
        return None
    alternatives = [
        fragment for fragment in PROTECTION_QUALITY_WHITELIST
        if ProtectionQuality.parse(fragment) != quality
    ]
    replacement = rng.choice(alternatives)
    return (
        f"{_KEYWORD_LINE}{replacement}",
        f"{PerturbationKind.SELECTOR}: {_KEYWORD_LINE}{original} -> {replacement}",
    )


def perturb_script(
    source_lines: list[str], rng: random.Random,
) -> tuple[list[str], str] | None:
    """Perturb the first script line that has something to perturb; None when
    no line does. A protection keyword line swaps its quality
    (:func:`swap_protection_quality`); any other line takes
    :func:`~effects.domain.script_variants.perturb`."""
    for index, line in enumerate(source_lines):
        swapped = swap_protection_quality(line, rng)
        if swapped is not None:
            lines = list(source_lines)
            lines[index] = swapped[0]
            return lines, swapped[1]
        result = perturb(line, rng)
        if result is None:
            continue
        perturbed, perturbation = result
        lines = list(source_lines)
        lines[index] = perturbed
        return lines, perturbation.describe()
    return None


def source_scripts(forge_cards_path: Path) -> list[Path]:
    """Every Forge source script under ``forge_cards_path``, in path order."""
    return sorted(Path(forge_cards_path).rglob("*.txt"))


def generate_variants(
    forge_cards_path: Path,
    output_path: Path,
    *,
    held_out: frozenset[str],
    seed: int = RANDOM_SEED,
    sources: list[Path] | None = None,
) -> list[GeneratedVariant]:
    """Write one perturbed script of every eligible source script into the
    variant tree (FR-058a): single-faced, not held out, with a parameter the
    perturbations can change.

    No variant is generated from a held-out card (FR-057): its changed text
    would not be held out, and generation is the only point where the rule
    can be enforced. Names are folded first, since a script's ``Name:`` is
    printed case and the holdout is lowercase. ``sources`` is
    :func:`source_scripts` of ``forge_cards_path`` when the caller already
    listed them: the walk is over all of Forge.

    A variant is named by its perturbed script's digest (FR-058b), so a file
    already under its name is the same script: it is reused as it stands,
    never written over, because an earlier run's records join it by key.
    """
    held_out = frozenset(fold_card_name(name) for name in held_out)
    rng = random.Random(seed)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sources = list(source_scripts(forge_cards_path) if sources is None else sources)
    rng.shuffle(sources)

    generated: list[GeneratedVariant] = []
    names: set[str] = set()
    for source in sources:
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
        if variant_name(card_name, variant_digest(perturbed)) in names:
            continue
        variant = write_variant(output_path, card_name, perturbed, description)
        names.add(variant.name)
        generated.append(variant)
    return generated


class CollectorAbort(Exception):
    """A step found nothing to go on with and has logged why; the run exits
    with ``code``."""

    def __init__(self, code: int = 1) -> None:
        super().__init__(code)
        self.code = code


def run(config: CollectVariantsConfig) -> int:
    """Generate variants, then play them through the records-only worker."""
    try:
        return _collect(config)
    except CollectorAbort as abort:
        return abort.code


def _collect(config: CollectVariantsConfig) -> int:
    """:func:`run`'s steps, each raising :class:`CollectorAbort` once there
    is nothing left to play."""
    _check_directed_forks(config)
    if not config.forge_cards_path.is_dir():
        logger.error(
            "No Forge card scripts at %s. This command reads the *source* "
            "scripts, not the converted ones.", config.forge_cards_path,
        )
        raise CollectorAbort
    sources = source_scripts(config.forge_cards_path)
    if config.round is VariantRound.STEERED:
        from effects.application.steered_variants import collect_steered

        return collect_steered(config, sources)
    variants = _generate(config, sources)
    try:
        variants = prepare_variants(config, variants)
        units = {variant.name: variant.games_unit() for variant in variants}
        candidates = {variant.name: variant.deck_candidate() for variant in variants}
        with collector_supervisor(
            workers=config.workers, effect_records=config.effect_records,
            caps=config.caps, variant_scripts=config.variant_scripts,
        ) as supervisor:
            counter = VariantGameCounter(config.effect_records, supervisor.run_id, config.workers)
            play_variant_rounds(units, candidates, supervisor, counter, config)
        return finish(config, units, counter, interrupted=supervisor.interrupted)
    finally:
        # The sidecar pass and every worker staged the tree into Forge's
        # custom-cards directory, which every later Forge startup reads.
        unstage_variants()


def prepare_variants(
    config: CollectVariantsConfig, variants: list[GeneratedVariant],
) -> list[GeneratedVariant]:
    """Every variant's sidecar written before any deck holds it, then the ones
    Forge loaded and that print no held-out text."""
    _write_sidecars(config.variant_scripts)
    return _keep_unheld(config, _keep_loadable(variants, config.variant_scripts))


def _check_directed_forks(config: CollectVariantsConfig) -> None:
    """The startup check ``match-outcomes`` runs (FR-014): aborts naming the
    rule before any variant is written when a listed rule cannot fork."""
    error = directed_fork_error(
        config.caps.directed_fork_rules, config.caps.directed_fork_probability,
    )
    if error is not None:
        logger.error("collect-variants: %s", error)
        raise CollectorAbort


def _generate(
    config: CollectVariantsConfig, sources: list[Path],
) -> list[GeneratedVariant]:
    held_out = config.deck_holdout.exclusions()
    variants = generate_variants(
        config.forge_cards_path, config.variant_scripts,
        held_out=held_out, seed=config.seed, sources=sources,
    )
    logger.info(
        "Generated %d variant scripts into %s (%d cards held out with their "
        "sources)",
        len(variants), config.variant_scripts, len(held_out),
    )
    for variant in variants:
        logger.info("  %s, from %s: %s", variant.name, variant.source_card, variant.perturbation)
    if not variants:
        logger.error("No script had a perturbable parameter")
        raise CollectorAbort
    return variants


def _write_sidecars(variant_scripts: Path) -> None:
    """Write every variant's sidecar before the first game: a variant without
    one produces records nothing can join. Aborts with the worker's code."""
    code = write_variant_sidecars(variant_scripts)
    if code != 0:
        logger.error(
            "VariantSidecarMain exited %d; variant records would have no "
            "sidecar to join against", code,
        )
        raise CollectorAbort(code)


def _keep_loadable(
    variants: list[GeneratedVariant], variant_scripts: Path,
) -> list[GeneratedVariant]:
    """The variants Forge loaded, per the sidecar pass's list, saying how many
    it rejected; aborts when it loaded none."""
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
        raise CollectorAbort
    return deckable


def drop_held_out_variants(
    variants: list[GeneratedVariant], held_out: HeldOutCards,
) -> list[GeneratedVariant]:
    """The variants whose sidecar prints no held-out text, deleting the
    fresh ones among the rest.

    Read off the sidecar, the only place the chained text exists. A dropped
    fresh variant's files are deleted so a later round cannot stage them; a
    reused one is only left out of the decks, since an earlier run may have
    decked it and its records join its files (FR-058b). One with no sidecar
    is kept, as the loadable filter has already decided it.
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
            if variant.fresh:
                variant.path.unlink(missing_ok=True)
                sidecar_path.unlink(missing_ok=True)
            continue
        kept.append(variant)
    return kept


def _keep_unheld(
    config: CollectVariantsConfig, variants: list[GeneratedVariant],
) -> list[GeneratedVariant]:
    """:func:`drop_held_out_variants` under the holdout ``build-corpus``
    chooses; aborts when it drops every variant."""
    trees = config.source_trees.converted()
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
        raise CollectorAbort
    return kept


def count_variant_shard(shard: Path) -> VariantShardGames:
    """One shard's variant games: every record read once."""
    from effects.infrastructure.shard_io import read_shard

    found = VariantShardGames()
    for record in read_shard(shard):
        found.add(record)
    return found


class VariantGameCounter:
    """Each variant's games from this run's own shards, each read once.

    A shard is named ``{run_id}.{worker}-{lifetime}``, so another run writing
    to the same directory never counts here; and a shard is final once its
    round's workers are stopped, so no later round reopens one.
    """

    def __init__(self, records_dir: Path, run_id: str, workers: int | None) -> None:
        self._records_dir = Path(records_dir)
        self._prefix = f"{run_id}."
        self._workers = workers
        self._counted: set[Path] = set()
        #: Every record this run's shards hold, over every round so far.
        self.records = 0

    def fresh_shards(self) -> list[Path]:
        """This run's shards not yet counted."""
        from effects.infrastructure.shard_io import iter_shards

        return [
            shard for shard in iter_shards(self._records_dir)
            if shard.name.startswith(self._prefix) and shard not in self._counted
        ]

    def update(self, units: Mapping[str, VariantGames]) -> set[str]:
        """Credit the new shards' games to ``units``; the names of the units
        their snapshots showed."""
        fresh = self.fresh_shards()
        found = VariantShardGames()
        for part in run_parallel(
            fresh, count_variant_shard, workers=self._workers, label="Counted",
            progress_every=PROGRESS_EVERY_SHARDS,
        ):
            found.merge(part)
        self._counted.update(fresh)
        self.records += found.records
        return credit_games(units, found)


def decks_of(
    weights: dict[str, float], candidates: Mapping[str, DeckCandidate], count: int,
    rng: random.Random, units: Mapping[str, VariantGames], *, by_rule: bool,
) -> tuple[list[list[str]], list[str | None] | None]:
    """A round's decks, and in the steered round the rule each deck is aimed
    at: a steered deck holds one rule's variants (FR-058d), decks dealt to the
    rules by the games their variants lack."""
    if not by_rule:
        return build_coverage_decks(weights, candidates, count, rng=rng), None
    per_rule: dict[str, dict[str, float]] = {}
    for name, weight in weights.items():
        per_rule.setdefault(units[name].rule or "", {})[name] = weight
    rules = sorted(per_rule)
    drawn = rng.choices(rules, weights=[sum(per_rule[r].values()) for r in rules], k=count)
    decks: list[list[str]] = []
    aimed: list[str | None] = []
    for rule in drawn:
        deck = build_coverage_decks(per_rule[rule], candidates, 1, rng=rng)
        if deck:
            decks.append(deck[0])
            aimed.append(rule)
    return decks, aimed


def play_variant_rounds(
    units: dict[str, VariantGames],
    candidates: Mapping[str, DeckCandidate],
    supervisor: CollectorSupervisor,
    counter: VariantGameCounter,
    config: CollectVariantsConfig,
    *,
    by_rule: bool = False,
    rng: random.Random | None = None,
) -> bool:
    """Play rounds until every variant has ``--variant-games`` games or has
    retired (FR-058a); True when Ctrl-C ended them.

    Each round plays ``--decks-per-round`` matches from decks weighted by the
    games each variant lacks. A variant a round's games showed and that gained
    no game counts a stalled round, and retires after ``--no-progress-rounds``
    of them in a row; one written into decks no game showed retires after
    ``UNSEEN_RETIRE_FACTOR`` times as many.
    """
    decks_file = config.effect_records / f"variant-decks.{supervisor.run_id}.txt"
    rng = rng if rng is not None else random.Random(config.seed)
    target = config.variant_games
    round_number = 0
    while not is_complete(units, target):
        round_number += 1
        weights = missing_games(units, target)
        decks, rules = decks_of(
            weights, candidates, config.decks_per_round, rng, units, by_rule=by_rule,
        )
        if not decks:
            logger.error("No unfinished variant has a deck candidate; nothing to deck.")
            break
        previous = {name: unit.progress for name, unit in units.items()}
        written = {name for deck in decks for name in deck} & set(units)
        supervisor.play_round(
            decks, decks_file, label="variant", matches=config.decks_per_round, rules=rules,
        )
        shown = counter.update(units)
        retired = retire_stalled(
            units, previous, no_progress_rounds=config.no_progress_rounds,
            decked=written & shown, written=written,
        )
        logger.info("%s", round_line(round_number, units, target, len(retired)))
        if supervisor.interrupted:
            logger.warning(
                "Interrupted after round %d; stopping instead of starting another",
                round_number,
            )
            return True
    return False


def round_line(round_number: int, units: Mapping[str, VariantGames], target: int,
               retired: int) -> str:
    done = sum(1 for unit in units.values() if unit.satisfied(target))
    remaining = sum(1 for unit in units.values() if not unit.done(target))
    return (
        f"variant round {round_number}: {done} with {target} games, "
        f"{remaining} remaining, {retired} retired this round"
    )


def finish(
    config: CollectVariantsConfig, units: Mapping[str, VariantGames],
    counter: VariantGameCounter, *, interrupted: bool,
) -> int:
    """Report what the run's rounds collected; 130 when Ctrl-C ended them, 1
    when they wrote no record at all."""
    played = sum(1 for unit in units.values() if unit.progress)
    retired = sum(1 for unit in units.values() if unit.retired)
    if interrupted:
        logger.warning(
            "Interrupted after %d new effect records; the variant round did not finish",
            counter.records,
        )
        return 130
    if counter.records <= 0:
        logger.error(
            "The variant round played its matches and added no new effect "
            "records at all. The staged scripts most likely never reached "
            "Forge's card database under the names the decks file spells, so "
            "every deck materialized as basics only.",
        )
        return 1
    logger.info(
        "Collected %d new effect records; %d of %d variants played in at least "
        "one game, %d retired short of %d games",
        counter.records, played, len(units), retired, config.variant_games,
    )
    return 0
