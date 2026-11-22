"""``collect-variants --round steered``: variants aimed at the rules the corpus
is short of (root spec § 8.15, FR-058c, FR-058d).

The round reads ``--rule-counts``, the gen-3 counts of the corpus being
collected, and finds the short pairs: a rule the rules-to-fields map marks
``trained`` with a declared detector, and a polarity with fewer than
``--steer-floor`` records over every source. Each pair takes
``--steer-variants`` × its shortfall ÷ the summed shortfall variants, at least
one, made on its carriers: the cards the rule's scenarios name and the cards
``rule-counts`` lists as its carriers, none held out or multi-faced. A carrier
takes the ``script_edit`` of every scenario variant of the pair's rule and
polarity that names it, then near misses: one numeric shift or selector swap on
a line the rule's tags act through. A (carrier, edit) pair, the script it makes,
yields one variant in the whole round.

A steered deck holds one rule's variants, its rule in the deck file's rule
column; the worker stamps each game's records ``aimed_at`` with its two decks'
rules. The round iterates: generate for the short pairs not retired, play
until each variant has ``--variant-games`` games or retires, count this run's
shards again, and retire every rule whose short polarities gained fewer than
``--steer-min-gain`` records, until no short pair is left or after
``--steer-iterations`` iterations.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.collect_variants import (
    CollectVariantsConfig,
    GeneratedVariant,
    VariantGameCounter,
    VariantRounds,
    finish,
    play_variant_rounds,
    prepare_variants,
    swap_protection_quality,
    write_variant,
)
from effects.application.collector_common import collector_supervisor, record_damaged_shards
from effects.application.shard_ledger import ShardLedger, ShardPass
from effects.domain.card_names import fold_card_name
from effects.domain.provenance import ProvenanceKey, SourceTree, split_script_file
from effects.domain.rule_catalogue import RuleMapEntry, ScenarioCatalogue, ScriptEdit
from effects.domain.script_variants import (
    is_multi_face,
    perturb,
    script_card_name,
    variant_digest,
    variant_name,
)
from effects.domain.tally_merge import merge_into
from effects.domain.variant_rounds import (
    ShortPair,
    SteerScope,
    VariantGames,
    acting_lines,
    counted_carriers,
    detected_rules,
    gained,
    rules_to_retire,
    short_pairs,
    variant_shares,
)
from effects.infrastructure.variant_staging import unstage_variants

if TYPE_CHECKING:
    from effects.infrastructure.collector_connector import CollectorSupervisor

logger = logging.getLogger(__name__)

#: Near-miss draws per variant a pair still wants before it gives up: a
#: line's perturbations repeat, and a draw naming a used script is wasted.
NEAR_MISS_DRAWS = 4

#: ``key -> the sidecar script text of the line it names``, None where none.
ScriptTextOf = Callable[[ProvenanceKey], "str | None"]


@dataclass
class Carrier:
    """A card a steered variant is made from: its Forge source script and the
    lines the rule's tags act through on it (from the counts)."""

    card: str
    path: Path
    lines: list[str]
    keys: set[ProvenanceKey] = field(default_factory=set)


def _script_names(source: Path) -> list[str]:
    """Every ``Name:`` line of one Forge script; none where it cannot be read."""
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line[len("Name:"):].strip() for line in lines if line.startswith("Name:")]


def script_index(sources: list[Path]) -> dict[str, Path]:
    """``card name -> its Forge source script``, every ``Name:`` line of every
    script, the first script naming a card kept."""
    index: dict[str, Path] = {}
    for source in sources:
        for name in _script_names(source):
            index.setdefault(name, source)
    return index


@dataclass(frozen=True)
class CarrierSources:
    """Where a rule's carriers are looked for: the scenario catalogue, the
    card-to-script index, Forge's cards folder and the held-out cards."""

    catalogue: ScenarioCatalogue
    index: Mapping[str, Path]
    forge_cards_path: Path
    held_out: frozenset[str]


def _scenario_paths(rule: str, sources: CarrierSources) -> dict[Path, set[ProvenanceKey]]:
    """The scripts of the cards the rule's scenarios name, each with no keys yet."""
    paths: dict[Path, set[ProvenanceKey]] = {}
    for scenario in sources.catalogue.by_rule().get(rule, ()):
        for card in scenario.cards:
            if card in sources.index:
                paths.setdefault(sources.index[card], set())
    return paths


def _add_counted_paths(
    paths: dict[Path, set[ProvenanceKey]], rule: str, counts: Mapping, forge_cards_path: Path,
) -> None:
    """Add the cardsfolder scripts of the carriers the counts list for the rule,
    with the keys that listed them."""
    for carrier in counted_carriers(counts, rule):
        tree, relative = split_script_file(carrier.script_file)
        if tree != SourceTree.CARDSFOLDER:
            continue
        path = Path(forge_cards_path) / relative
        if path.is_file():
            paths.setdefault(path, set()).update(carrier.keys)


def _excluded(card: str | None, lines: list[str], held_out: frozenset[str]) -> bool:
    return card is None or is_multi_face(lines) or fold_card_name(card) in held_out


def _read_carrier(path: Path, held_out: frozenset[str]) -> Carrier | None:
    """The carrier a script makes; None for a multi-faced or held-out card."""
    lines = path.read_text(encoding="utf-8").splitlines()
    card = script_card_name(lines)
    if _excluded(card, lines, held_out):
        return None
    return Carrier(card, path, lines)


def rule_carriers(rule: str, counts: Mapping, sources: CarrierSources) -> dict[str, Carrier]:
    """``card -> Carrier`` of one rule (FR-058c): the cards its scenarios name
    and the cardsfolder carriers the counts list, read from Forge's source
    scripts; a held-out or multi-faced card, or one with no source script,
    carries nothing."""
    paths = _scenario_paths(rule, sources)
    _add_counted_paths(paths, rule, counts, sources.forge_cards_path)
    found: dict[str, Carrier] = {}
    for path, keys in paths.items():
        carrier = _read_carrier(path, sources.held_out)
        if carrier is not None:
            found.setdefault(carrier.card, carrier).keys.update(keys)
    return found


def pair_edits(pair: ShortPair, catalogue: ScenarioCatalogue) -> list[tuple[str, ScriptEdit]]:
    """``(scenario/variant label, script_edit)`` of every scenario variant of
    the pair's rule and polarity that carries one."""
    return [
        (f"{scenario.id}/{variant.label}", variant.script_edit)
        for scenario in catalogue.by_rule().get(pair.rule, ())
        for variant in scenario.variants
        if variant.polarity is pair.polarity and variant.script_edit is not None
    ]


def _applied(label: str, edit: ScriptEdit, carrier: Carrier) -> list[str] | None:
    """The carrier's lines with the scenario's ``script_edit`` applied, None
    (and a warning) where the edit does not apply."""
    try:
        return edit.applied(carrier.lines)
    except ValueError as exc:
        logger.warning("%s: %s", label, exc)
        return None


def near_miss_lines(carrier: Carrier, script_text_of: ScriptTextOf) -> list[int]:
    """The carrier's source lines its rule's tags act through."""
    found: set[int] = set()
    for key in carrier.keys:
        text = script_text_of(key)
        if text:
            found.update(acting_lines(carrier.lines, text))
    return sorted(found)


def _near_miss(lines: list[str], index: int, rng: random.Random) -> tuple[list[str], str] | None:
    """One numeric shift or selector swap on ``lines[index]``; None where the
    line has nothing to perturb."""
    line = lines[index]
    swapped = swap_protection_quality(line, rng)
    if swapped is not None:
        changed, description = swapped
    else:
        result = perturb(line, rng)
        if result is None:
            return None
        changed, perturbation = result
        description = perturbation.describe()
    edited = list(lines)
    edited[index] = changed
    return edited, f"near miss: {description}"


@dataclass
class SteeredGeneration:
    """What the steered round's generation reads, fixed for the round."""

    catalogue: ScenarioCatalogue
    index: Mapping[str, Path]
    forge_cards_path: Path
    held_out: frozenset[str]
    script_text_of: ScriptTextOf
    output_path: Path
    rng: random.Random
    #: Every variant name an earlier pair or iteration made: a (carrier, edit)
    #: yields at most one variant in the whole round.
    used: set[str] = field(default_factory=set)

    @property
    def sources(self) -> CarrierSources:
        return CarrierSources(self.catalogue, self.index, self.forge_cards_path, self.held_out)

    def carriers(self, rule: str, counts: Mapping) -> dict[str, Carrier]:
        """The carriers of one rule (FR-058c)."""
        return rule_carriers(rule, counts, self.sources)

    def variants(self, pairs: list[ShortPair], counts: Mapping, total: int,
                 ) -> list[GeneratedVariant]:
        """Each pair's share of ``total`` variants (FR-058c)."""
        shares = variant_shares(pairs, total)
        made: list[GeneratedVariant] = []
        for pair in pairs:
            carriers = self.carriers(pair.rule, counts)
            made += self.pair_variants(pair, carriers, shares[(pair.rule, pair.polarity)])
        return made

    def pair_variants(self, pair: ShortPair, carriers: Mapping[str, Carrier],
                      wanted: int) -> list[GeneratedVariant]:
        """Up to ``wanted`` new variants for one pair: its scenario variants'
        script edits on the carriers they name, then near misses."""
        made = self._edit_variants(pair, carriers, wanted)
        if len(made) < wanted:
            self._add_near_misses(pair, carriers, wanted, made)
        return made

    def _edit_variants(self, pair: ShortPair, carriers: Mapping[str, Carrier],
                       wanted: int) -> list[GeneratedVariant]:
        made: list[GeneratedVariant] = []
        for label, edit in pair_edits(pair, self.catalogue):
            if len(made) >= wanted:
                break
            carrier = carriers.get(edit.card)
            lines = None if carrier is None else _applied(label, edit, carrier)
            if lines is not None:
                made += self._write(pair, carrier, lines, f"script_edit {label}")
        return made

    def _add_near_misses(self, pair: ShortPair, carriers: Mapping[str, Carrier],
                         wanted: int, made: list[GeneratedVariant]) -> None:
        """Draw near misses from the carriers' acting lines into ``made``, until
        ``wanted`` or ``NEAR_MISS_DRAWS`` a variant wanted are spent."""
        pool = [
            (carrier, index) for carrier in carriers.values()
            for index in near_miss_lines(carrier, self.script_text_of)
        ]
        if not pool:
            return
        for draw in range(NEAR_MISS_DRAWS * wanted):
            if len(made) >= wanted:
                return
            carrier, index = pool[draw % len(pool)]
            result = _near_miss(carrier.lines, index, self.rng)
            if result is not None:
                made += self._write(pair, carrier, *result)

    def _write(self, pair: ShortPair, carrier: Carrier, lines: list[str],
               description: str) -> list[GeneratedVariant]:
        """The variant ``lines`` make of ``carrier``, unless an earlier pair or
        iteration made it, or it would be multi-faced."""
        name = variant_name(carrier.card, variant_digest(lines))
        if name in self.used or is_multi_face(lines):
            return []
        self.used.add(name)
        variant = write_variant(self.output_path, carrier.card, lines, description)
        return [replace(variant, rule=pair.rule)]


def _add_section(rules: dict[str, dict[str, dict[str, int]]], section: Mapping) -> None:
    """Add one document's ``rules`` section into ``rules``."""
    for rule, sides in section.items():
        for polarity, sources in sides.items():
            into = rules.setdefault(rule, {}).setdefault(polarity, {})
            for source, n in sources.items():
                into[source] = into.get(source, 0) + n


def combined_counts(*documents: Mapping) -> dict:
    """The ``rules`` sections of several ``rule-counts`` documents summed."""
    rules: dict[str, dict[str, dict[str, int]]] = {}
    for document in documents:
        _add_section(rules, document.get("rules", {}))
    return {"rules": rules}


@dataclass(frozen=True)
class RuleCountSetup:
    """What counting rules over shards reads: the rules catalogue, the sidecar
    roots the records' keys resolve in, and the counting workers."""

    rules_dir: Path
    sidecar_roots: dict
    workers: int | None


class RunRuleCounter(ShardLedger):
    """The rule counts of this run's own shards, each read once (FR-058d:
    the pairs are counted again over the corpus and the round's shards)."""

    def __init__(self, records_dir: Path, run_id: str, setup: RuleCountSetup) -> None:
        from effects.application.rule_counts import RuleCounts

        super().__init__()
        self._records_dir = Path(records_dir)
        self._prefix = f"{run_id}."
        self._rules_dir = setup.rules_dir
        self._roots = setup.sidecar_roots
        self._workers = setup.workers
        self.total = RuleCounts()

    def update(self) -> None:
        """Count the run's shards not counted yet, and say, as ``rule-counts``
        does, how many gen-3 games they left out as cut. A damaged shard is
        counted up to the damage: one must not stop the run."""
        from effects.application.rule_counts import cut_games_line

        self.count(self.fresh(self._records_dir, (self._prefix,)), self._rule_pass(),
                   partial(merge_into, self.total))
        if self.last_cut:
            logger.info("Steered rule counts: %s", cut_games_line(self.last_cut))

    def _rule_pass(self) -> ShardPass:
        """``rule-counts``' own shard count, salvaging a damaged shard."""
        from effects.application.rule_counts import CountContext, count_shard

        return ShardPass(
            partial(count_shard, salvage=True), self._workers, progress_every=50,
            label="Counted rules in", context=CountContext(self._rules_dir, self._roots),
        )

    def document(self) -> dict:
        return self.total.to_json()


def _steered_generation(config: CollectVariantsConfig, sources: list[Path],
                        catalogue: ScenarioCatalogue) -> SteeredGeneration:
    """The round's generation, reading each carrier's script text off the
    converted sidecars."""
    from effects.infrastructure.converted_trees import SourceTrees, process_sidecars

    converted = process_sidecars(SourceTrees.of(config.cards_folders).sidecar_roots())

    def script_text_of(key: ProvenanceKey) -> str | None:
        line = converted.line_or_none(key)
        return line.script_text if line is not None else None

    return SteeredGeneration(
        catalogue=catalogue, index=script_index(sources),
        forge_cards_path=config.forge_cards_path,
        held_out=config.deck_holdout.exclusions(),
        script_text_of=script_text_of, output_path=config.variant_scripts,
        rng=random.Random(config.seed),
    )


def _run_rule_counter(config: CollectVariantsConfig, run_id: str) -> RunRuleCounter:
    from effects.infrastructure.converted_trees import SourceTrees

    return RunRuleCounter(config.effect_records, run_id, RuleCountSetup(
        rules_dir=config.rules_dir,
        sidecar_roots=SourceTrees.of(config.cards_folders, config.variant_scripts).sidecar_roots(),
        workers=config.workers,
    ))


def collect_steered(config: CollectVariantsConfig, sources: list[Path]) -> int:
    """The steered round's iterations under one supervisor (one run id)."""
    from effects.infrastructure.rules_catalogue_io import load_heads, load_scenarios

    counts = json.loads(Path(config.rule_counts).read_text(encoding="utf-8"))
    catalogue = load_scenarios(config.rules_dir)
    heads = load_heads(config.rules_dir, catalogue.rules())
    detected = detected_rules(catalogue)
    generation = _steered_generation(config, sources, catalogue)
    units: dict[str, VariantGames] = {}
    try:
        with collector_supervisor(
            config, variant_scripts=config.variant_scripts,
        ) as supervisor:
            games = VariantGameCounter(config.effect_records, supervisor.run_id, config.workers)
            outcome = steer(SteeredRound(
                config=config, generation=generation, counts=counts, heads=heads,
                detected=detected, units=units, supervisor=supervisor, games=games,
                ruled=_run_rule_counter(config, supervisor.run_id),
            ))
            if outcome is not SteerOutcome.PLAYED:
                return outcome.exit_code
        return finish(config, units, games, interrupted=supervisor.interrupted)
    finally:
        unstage_variants()


class SteerOutcome(Enum):
    """How a steered round's iterations ended, as the exit code reads it."""

    PLAYED = "played"
    #: No short pair left: every unit was already done.
    NOTHING_SHORT = "nothing short"
    #: Short pairs, and no variant could be made for any of them.
    NO_VARIANT = "no variant"

    @property
    def exit_code(self) -> int:
        return 1 if self is SteerOutcome.NO_VARIANT else 0


@dataclass(frozen=True)
class SteeredRound:
    """What the steered round's iterations read and fill: the config, the
    generation, the corpus's rule counts, the rule map's heads and the rules
    with a declared detector, the variants' game units (filled as the round
    plays), the supervisor and the counters of this run's games and rules."""

    config: CollectVariantsConfig
    generation: SteeredGeneration
    counts: Mapping
    heads: Mapping[str, RuleMapEntry]
    detected: frozenset[str]
    units: dict[str, VariantGames]
    supervisor: CollectorSupervisor
    games: VariantGameCounter
    ruled: RunRuleCounter


@dataclass
class _Progress:
    """What the iterations so far did: the rules retired, whether a variant was
    played, whether the round stalled with short pairs and no variant."""

    retired: set[str] = field(default_factory=set)
    played: bool = False
    stalled: bool = False

    def outcome(self) -> SteerOutcome:
        if self.played:
            return SteerOutcome.PLAYED
        return SteerOutcome.NO_VARIANT if self.stalled else SteerOutcome.NOTHING_SHORT


def steer(steered: SteeredRound) -> SteerOutcome:
    """The iterations (FR-058d); what ended them when none had anything to play."""
    progress = _Progress()
    for iteration in range(1, steered.config.steer_iterations + 1):
        if not _iterate(steered, iteration, progress):
            break
    return progress.outcome()


def _iterate(steered: SteeredRound, iteration: int, progress: _Progress) -> bool:
    """One iteration; whether the next should follow."""
    config = steered.config
    before = combined_counts(steered.counts, steered.ruled.document())
    pairs = short_pairs(before, SteerScope(
        steered.heads, steered.detected, config.steer_floor, frozenset(progress.retired),
    ))
    if not pairs:
        logger.info("Steered iteration %d: no short pair left", iteration)
        return False
    variants = steered.generation.variants(pairs, steered.counts, config.steer_variants)
    log_variants(iteration, pairs, variants)
    if not variants:
        progress.stalled = not progress.played and _log_stall(iteration, len(pairs))
        return False
    progress.played = True
    interrupted = _play(steered, prepare_variants(config, variants))
    progress.retired |= _retire(steered, iteration, before, pairs)
    return not interrupted


def _play(steered: SteeredRound, variants: list[GeneratedVariant]) -> bool:
    """Play the prepared variants' rounds, then count this run's shards again;
    whether the run was interrupted."""
    fresh = {variant.name: variant.games_unit() for variant in variants}
    steered.units.update(fresh)
    interrupted = play_variant_rounds(VariantRounds(
        units=fresh, candidates={v.name: v.deck_candidate() for v in variants},
        counter=steered.games, config=steered.config, by_rule=True,
    ), steered.supervisor, rng=steered.generation.rng)
    steered.ruled.update()
    record_damaged_shards(
        steered.supervisor, sorted({*steered.games.damaged, *steered.ruled.damaged}))
    return interrupted


def _retire(steered: SteeredRound, iteration: int, before: dict,
            pairs: list[ShortPair]) -> set[str]:
    """The rules whose short polarities gained too few records this iteration."""
    after = combined_counts(steered.counts, steered.ruled.document())
    newly = rules_to_retire(
        gained(before, after, pairs), min_gain=steered.config.steer_min_gain)
    logger.info("Steered iteration %d: retired %d rule(s): %s", iteration, len(newly),
                ", ".join(sorted(newly)) or "none")
    return newly


def _log_stall(iteration: int, pairs: int) -> bool:
    """Name a round that had short pairs and no variant; always True."""
    logger.error(
        "Steered iteration %d: %d short pair(s) and no variant could be made for any "
        "of them", iteration, pairs,
    )
    return True


def log_variants(iteration: int, pairs, variants) -> None:
    """The iteration's short pairs and the variants made for them."""
    logger.info(
        "Steered iteration %d: %d short pair(s) over %d rule(s), %d new variant(s)",
        iteration, len(pairs), len({p.rule for p in pairs}), len(variants),
    )
    for variant in variants:
        logger.info("  %s, from %s, for %s: %s", variant.name, variant.source_card,
                    variant.rule, variant.perturbation)
