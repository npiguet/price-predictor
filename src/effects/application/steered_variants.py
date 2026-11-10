"""``collect-variants --round steered``: variants aimed at the rules the corpus
is short of (root spec § 8.15, FR-058c, FR-058d).

The round reads ``--rule-counts``, the gen-3 counts of the corpus being
collected, and finds the short pairs: a rule the rules-to-fields map marks
``trained`` with a declared detector, and a polarity with fewer than
``--steer-floor`` records over all three sources. Each pair takes
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
from dataclasses import dataclass, field
from pathlib import Path

from effects.application.collect_variants import (
    CollectVariantsConfig,
    GeneratedVariant,
    VariantGameCounter,
    finish,
    play_variant_rounds,
    prepare_variants,
    swap_protection_quality,
    write_variant,
)
from effects.application.collector_common import collector_supervisor
from effects.application.parallel_map import run_parallel
from effects.domain.card_names import fold_card_name
from effects.domain.provenance import ProvenanceKey, SourceTree, split_script_file
from effects.domain.rule_catalogue import ScenarioCatalogue, ScriptEdit
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


def script_index(sources: list[Path]) -> dict[str, Path]:
    """``card name -> its Forge source script``, every ``Name:`` line of every
    script, the first script naming a card kept."""
    index: dict[str, Path] = {}
    for source in sources:
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("Name:"):
                index.setdefault(line[len("Name:"):].strip(), source)
    return index


def rule_carriers(
    rule: str,
    catalogue: ScenarioCatalogue,
    counts: Mapping,
    *,
    index: Mapping[str, Path],
    forge_cards_path: Path,
    held_out: frozenset[str],
) -> dict[str, Carrier]:
    """``card -> Carrier`` of one rule (FR-058c): the cards its scenarios name
    and the cardsfolder carriers the counts list, read from Forge's source
    scripts; a held-out or multi-faced card, or one with no source script,
    carries nothing."""
    paths: dict[Path, set[ProvenanceKey]] = {}
    for scenario in catalogue.by_rule().get(rule, ()):
        for card in scenario.cards:
            if card in index:
                paths.setdefault(index[card], set())
    for carrier in counted_carriers(counts, rule):
        tree, relative = split_script_file(carrier.script_file)
        if tree != SourceTree.CARDSFOLDER:
            continue
        path = Path(forge_cards_path) / relative
        if path.is_file():
            paths.setdefault(path, set()).update(carrier.keys)
    found: dict[str, Carrier] = {}
    for path, keys in paths.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        card = script_card_name(lines)
        if card is None or is_multi_face(lines) or fold_card_name(card) in held_out:
            continue
        found.setdefault(card, Carrier(card, path, lines)).keys.update(keys)
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

    def variants(self, pairs: list[ShortPair], counts: Mapping, total: int,
                 ) -> list[GeneratedVariant]:
        """Each pair's share of ``total`` variants (FR-058c)."""
        shares = variant_shares(pairs, total)
        made: list[GeneratedVariant] = []
        for pair in pairs:
            carriers = rule_carriers(
                pair.rule, self.catalogue, counts, index=self.index,
                forge_cards_path=self.forge_cards_path, held_out=self.held_out,
            )
            made += self.pair_variants(pair, carriers, shares[(pair.rule, pair.polarity)])
        return made

    def pair_variants(self, pair: ShortPair, carriers: Mapping[str, Carrier],
                      wanted: int) -> list[GeneratedVariant]:
        """Up to ``wanted`` new variants for one pair: its scenario variants'
        script edits on the carriers they name, then near misses."""
        made: list[GeneratedVariant] = []
        for label, edit in pair_edits(pair, self.catalogue):
            if len(made) >= wanted:
                return made
            carrier = carriers.get(edit.card)
            if carrier is None:
                continue
            try:
                lines = edit.applied(carrier.lines)
            except ValueError as exc:
                logger.warning("%s: %s", label, exc)
                continue
            made += self._write(pair, carrier, lines, f"script_edit {label}")
        pool = [
            (carrier, index) for carrier in carriers.values()
            for index in near_miss_lines(carrier, self.script_text_of)
        ]
        draws = 0
        while pool and len(made) < wanted and draws < NEAR_MISS_DRAWS * wanted:
            carrier, index = pool[draws % len(pool)]
            draws += 1
            result = _near_miss(carrier.lines, index, self.rng)
            if result is not None:
                made += self._write(pair, carrier, *result)
        return made

    def _write(self, pair: ShortPair, carrier: Carrier, lines: list[str],
               description: str) -> list[GeneratedVariant]:
        """The variant ``lines`` make of ``carrier``, unless an earlier pair or
        iteration made it, or it would be multi-faced."""
        name = variant_name(carrier.card, variant_digest(lines))
        if name in self.used or is_multi_face(lines):
            return []
        self.used.add(name)
        return [write_variant(self.output_path, carrier.card, lines, description,
                              rule=pair.rule)]


def combined_counts(*documents: Mapping) -> dict:
    """The ``rules`` sections of several ``rule-counts`` documents summed."""
    rules: dict[str, dict[str, dict[str, int]]] = {}
    for document in documents:
        for rule, sides in document.get("rules", {}).items():
            for polarity, sources in sides.items():
                into = rules.setdefault(rule, {}).setdefault(polarity, {})
                for source, n in sources.items():
                    into[source] = into.get(source, 0) + n
    return {"rules": rules}


class RunRuleCounter:
    """The rule counts of this run's own shards, each read once (FR-058d:
    the pairs are counted again over the corpus and the round's shards)."""

    def __init__(self, records_dir: Path, run_id: str, *, rules_dir: Path,
                 sidecar_roots: dict, workers: int | None) -> None:
        from effects.application.rule_counts import RuleCounts

        self._games = VariantGameCounter(records_dir, run_id, workers)
        self._rules_dir = rules_dir
        self._roots = sidecar_roots
        self._workers = workers
        self._counted: set[Path] = set()
        self.total = RuleCounts()

    def update(self) -> None:
        from effects.application.rule_counts import CountContext, count_shard

        fresh = [shard for shard in self._games.fresh_shards() if shard not in self._counted]
        for found in run_parallel(
            fresh, count_shard, workers=self._workers, label="Counted rules in",
            progress_every=50, context=CountContext(self._rules_dir, self._roots),
        ):
            merge_into(self.total, found)
        self._counted.update(fresh)

    def document(self) -> dict:
        return self.total.to_json({})


def collect_steered(config: CollectVariantsConfig, sources: list[Path]) -> int:
    """The steered round's iterations under one supervisor (one run id)."""
    from effects.infrastructure.converted_trees import SourceTrees, process_sidecars
    from effects.infrastructure.rules_catalogue_io import load_heads, load_scenarios

    counts = json.loads(Path(config.rule_counts).read_text(encoding="utf-8"))
    catalogue = load_scenarios(config.rules_dir)
    heads = load_heads(config.rules_dir, catalogue.rules())
    detected = detected_rules(catalogue)
    converted = process_sidecars(SourceTrees.of(config.cards_folders).sidecar_roots())

    def script_text_of(key: ProvenanceKey) -> str | None:
        line = converted.line_or_none(key)
        return line.script_text if line is not None else None

    generation = SteeredGeneration(
        catalogue=catalogue, index=script_index(sources),
        forge_cards_path=config.forge_cards_path,
        held_out=frozenset(fold_card_name(n) for n in config.deck_holdout.exclusions()),
        script_text_of=script_text_of, output_path=config.variant_scripts,
        rng=random.Random(config.seed),
    )
    units: dict[str, VariantGames] = {}
    try:
        with collector_supervisor(
            workers=config.workers, effect_records=config.effect_records,
            caps=config.caps, variant_scripts=config.variant_scripts,
        ) as supervisor:
            games = VariantGameCounter(config.effect_records, supervisor.run_id, config.workers)
            ruled = RunRuleCounter(
                config.effect_records, supervisor.run_id, rules_dir=config.rules_dir,
                sidecar_roots=SourceTrees.of(
                    config.cards_folders, config.variant_scripts).sidecar_roots(),
                workers=config.workers,
            )
            if not steer(config, generation, counts, heads, detected, units, supervisor,
                         games, ruled):
                return 0
        return finish(config, units, games, interrupted=supervisor.interrupted)
    finally:
        unstage_variants()


def steer(config, generation: SteeredGeneration, counts, heads, detected,
          units: dict[str, VariantGames], supervisor, games, ruled) -> bool:
    """The iterations (FR-058d); False when no iteration had anything to play."""
    retired: set[str] = set()
    played = False
    for iteration in range(1, config.steer_iterations + 1):
        before = combined_counts(counts, ruled.document())
        pairs = short_pairs(before, heads, detected, floor=config.steer_floor, retired=retired)
        if not pairs:
            logger.info("Steered iteration %d: no short pair left", iteration)
            break
        variants = generation.variants(pairs, counts, config.steer_variants)
        logger.info(
            "Steered iteration %d: %d short pair(s) over %d rule(s), %d new variant(s)",
            iteration, len(pairs), len({p.rule for p in pairs}), len(variants),
        )
        for variant in variants:
            logger.info("  %s, from %s, for %s: %s", variant.name, variant.source_card,
                        variant.rule, variant.perturbation)
        if not variants:
            break
        variants = prepare_variants(config, variants)
        fresh = {variant.name: variant.games_unit() for variant in variants}
        units.update(fresh)
        played = True
        interrupted = play_variant_rounds(
            fresh, {v.name: v.deck_candidate() for v in variants}, supervisor, games,
            config, by_rule=True, rng=generation.rng,
        )
        ruled.update()
        after = combined_counts(counts, ruled.document())
        newly = rules_to_retire(gained(before, after, pairs), min_gain=config.steer_min_gain)
        retired |= newly
        logger.info("Steered iteration %d: retired %d rule(s): %s", iteration, len(newly),
                    ", ".join(sorted(newly)) or "none")
        if interrupted:
            break
    return played
