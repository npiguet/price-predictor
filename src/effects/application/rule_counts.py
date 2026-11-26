"""``rule-counts``: how many records show each rule, per side and per source (FR-008).

Every record of a records set is tagged by the catalogue's detectors and
counted under ``rule -> polarity -> source``. The source says where the record
came from, which decides what it can be trusted for:

- ``fork``: a probe's, an intervention's or a protection fork's record, which
  re-records a moment of its live game, whatever else it carries;
- ``aimed``: the record carries ``aimed_at`` (a directed fork, or a coverage or
  variant run aimed at rules), whatever its game;
- ``random_seat``: a natural record of a game with a random seat;
- ``forge_ai``: a natural record of a game the Forge AI played on both seats.

Per rule the counts also list its carriers (FR-058f), the cards the steered
variant round edits: the script file of every card that is a condition entity
of the rule's tags, with the lines the tags act through on it and the records
of each. A line acted through is an acting key of the record (the acting line)
on the carrier's script; where none is, the carrier's keyword line of the
keyword the tagging scenario's detector names, from the entity's lines or,
where the snapshot lists none, from the carrier's sidecar. A card entity with
no printed line names no script file and carries nothing. A legality record
(``playability``) judges every entity of a list, so its carrier is the
candidate's own card, with the candidate's line, and a judged card only
through the keyword line the rule names.

Shards are read once each, in parallel; a game never spans shards (a shard is
one worker JVM's lifetime), so whether a game has a random seat, and whether a
game ended (its game-end marker read), is decided at the end of its
shard. A game without a marker was cut and ``build-corpus`` drops it
whole: its records are read but count toward nothing (no tag, review,
carrier, ``records`` or ``not_countable``, as the build's ``rule_counts``),
and the counts say how many such games were left out (``cut_games``). Each worker builds the
detector index and its sidecar cache once. Prior art: the ``build-corpus``
survey (``corpus_survey``'s ``_ShardSurveyor``), which reads shards the same
way; it is not reused whole because its read context is built around the
corpus's holdout and caps.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from effects.application.parallel_map import run_parallel
from effects.domain.game_ends import GameEnds
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey, SidecarLookup
from effects.domain.records import EffectRecord, RecordKind, acting_keys
from effects.domain.rule_catalogue import NoDetector, Polarity
from effects.domain.rule_detectors import (
    DetectorIndex,
    RuleSource,
    RuleTag,
    counts_as_fork,
    tag_record,
)
from effects.domain.shard_generation import ShardGeneration
from effects.domain.tally_merge import NOT_MERGED, merge_into
from effects.infrastructure.shard_io import DecodeTally
from effects.infrastructure.sidecar_io import JoinGaps

logger = logging.getLogger(__name__)


@dataclass
class RuleCounts:
    """``(rule, polarity, source) -> records``, and what the counts cannot say."""

    counts: Counter = field(default_factory=Counter)
    #: Rules a declared detector needs a field for that the records lack (FR-007).
    not_countable: set = field(default_factory=set)
    #: ``(record_id, scenario)`` of every finding whose outcome disagrees with its side.
    review: set = field(default_factory=set)
    records: int = 0
    #: ``(rule, script file, key) -> records``: a carrier's own count under
    #: key None, each line its tags act through under the line's key.
    carriers: Counter = field(default_factory=Counter)
    #: The records' generation (``gen-3``), which the steered
    #: variant round checks (FR-058g); set once over the whole set.
    generation: str = field(default="", metadata=NOT_MERGED)
    #: What reading the shards met without stopping, handed back by each
    #: worker: values outside the closed lists, damaged last members.
    decode: DecodeTally = field(default_factory=DecodeTally)
    #: The keys the sidecar cache read as no line, by why.
    joins: JoinGaps = field(default_factory=JoinGaps)
    #: Shards a damaged member or line ended early, under ``salvage`` alone:
    #: each one's message, its records before the damage counted.
    damaged: list = field(default_factory=list)
    #: Games without a game-end marker, left out of every count.
    cut_games: int = 0

    def to_json(self, standing: CatalogueStanding | None = None) -> dict:
        """The counts file; ``standing`` the catalogue's say of the rules no
        count can show, none where a caller has nothing of the catalogue."""
        if standing is None:
            standing = CatalogueStanding()
        by_rule: dict[str, dict[str, dict[str, int]]] = {}
        for (rule, polarity, source), n in sorted(self.counts.items()):
            by_rule.setdefault(rule, {}).setdefault(polarity, {})[source] = n
        not_countable = self.not_countable - set(by_rule)
        return {
            "generation": self.generation,
            "records": self.records,
            "cut_games": self.cut_games,
            "rules": by_rule,
            "declared_zero": sorted(standing.declared - set(by_rule) - not_countable,
                                    key=rule_order),
            "not_countable": sorted(not_countable),
            "none": dict(sorted(standing.none.items())),
            "untaggable": dict(sorted(standing.untaggable.items())),
            "same_as": dict(sorted(standing.same_as.items())),
            "review": [{"record_id": r, "scenario": s} for r, s in sorted(self.review)],
            "carriers": self._carriers_json(),
        }

    def _carriers_json(self) -> dict[str, list[dict]]:
        """``rule -> [{script_file, records, lines: [{key, records}]}]``, the
        carriers most records name first, each line keyed as a sidecar keys it."""
        files: dict[str, dict[str, int]] = {}
        lines: dict[tuple[str, str], list[tuple[ProvenanceKey, int]]] = {}
        for (rule, script_file, key), n in self.carriers.items():
            if key is None:
                files.setdefault(rule, {})[script_file] = n
            else:
                lines.setdefault((rule, script_file), []).append((key, n))
        return {
            rule: [
                {"script_file": script_file, "records": n, "lines": [
                    {"key": key.sidecar_dict(), "records": m}
                    for key, m in sorted(
                        lines.get((rule, script_file), ()),
                        key=lambda item: (-item[1], item[0].trait_kind,
                                          item[0].face, item[0].index_within_kind,
                                          item[0].option or -1),
                    )
                ]}
                for script_file, n in sorted(by_file.items(), key=lambda i: (-i[1], i[0]))
            ]
            for rule, by_file in sorted(files.items())
        }


@dataclass(frozen=True)
class CatalogueStanding:
    """What the catalogue says of the rules no count can show.

    ``declared``: the rules a taggable declaration speaks for, so one the
    records never tag is listed as declared with zero records rather than
    omitted (D3); ``none``: rules every scenario of which declares no
    detector, with the first reason; ``untaggable``: per scenario, a
    declaration asking its record kind for an event that kind never lists
    (``DetectorIndex.untaggable``); ``same_as``: per scenario, the scenario
    it states the same fact as, under whose rules its own are counted.
    """

    declared: frozenset[str] = frozenset()
    none: dict[str, str] = field(default_factory=dict)
    untaggable: dict[str, dict] = field(default_factory=dict)
    same_as: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def of(cls, catalogue) -> CatalogueStanding:
        index = DetectorIndex.of(catalogue)
        return cls(
            declared=index.declared_rules,
            none=none_rules(catalogue),
            untaggable={sid: {"rules": list(index.untaggable_rules[sid]), "reason": reason}
                        for sid, reason in index.untaggable.items()},
            same_as={sid: {"rules": list(d.rules), "same_as": list(d.same_as),
                           "counted_under": list(d.counted_under)}
                     for sid, d in index.duplicates.items()},
        )


@dataclass(frozen=True)
class CountContext:
    """One worker's read side: the detector index and its sidecar cache, built once."""

    rules_dir: Path
    sidecar_roots: dict

    def __call__(self):
        from effects.infrastructure.converted_trees import process_sidecars
        from effects.infrastructure.rules_catalogue_io import load_scenarios

        catalogue = load_scenarios(self.rules_dir)
        return DetectorIndex.of(catalogue), process_sidecars(
            self.sidecar_roots), scenario_keywords(catalogue)


def scenario_keywords(catalogue) -> dict[str, str]:
    """``scenario id -> keyword`` its detector declares, in the package's
    spelling, for the scenarios whose detector names one."""
    found: dict[str, str] = {}
    for scenario in catalogue.scenarios:
        params = getattr(scenario.detector, "params", None) or {}
        if isinstance(params.get("keyword"), str):
            found[scenario.id] = normalize_keyword(params["keyword"])
    return found


#: ``(script file, key)`` pairs: a carrier's card under key None, each line
#: acted through on it under its key.
Carried = set[tuple[str, ProvenanceKey | None]]


def tag_carriers(
    record: EffectRecord, tag: RuleTag, keywords: dict[str, str],
    sidecars: SidecarLookup | None = None,
) -> Carried:
    """``(script file, key)`` of the carriers ``tag`` names in ``record``:
    each carrier's card with key None, and each line the tag acts through on
    it (see the module's docstring)."""
    keyword = _KeywordLines(keywords.get(tag.scenario), sidecars)
    if record.kind is RecordKind.PLAYABILITY:
        return _legality_carriers(record, tag, keyword)
    acting = acting_keys(record)
    found: Carried = set()
    for entity, script_file in _printed_cards(record, tag.condition):
        through = ([key for key in acting if key.script_file == script_file]
                   or keyword.keys(entity, script_file))
        found.add((script_file, None))
        found.update((script_file, key) for key in through)
    return found


def _legality_carriers(record: EffectRecord, tag: RuleTag, keyword: _KeywordLines) -> Carried:
    """A legality record's carriers: the candidate's own card with its line,
    and a judged card only through the keyword line the rule names."""
    found: Carried = {(key.script_file, line)
                      for key in acting_keys(record) for line in (None, key)}
    for entity, script_file in _printed_cards(record, tag.condition):
        through = keyword.keys(entity, script_file)
        if through:
            found.add((script_file, None))
            found.update((script_file, key) for key in through)
    return found


def _printed_cards(record: EffectRecord, entity_ids):
    """``(entity, script file)`` of each of ``entity_ids`` with a printed line."""
    for entity_id in entity_ids:
        entity = record.state.entity(entity_id)
        if entity is not None and entity.printed:
            yield entity, entity.printed[0].script_file


@dataclass(frozen=True)
class _KeywordLines:
    """The keyword a tagging scenario's detector names, looked for on a carrier."""

    keyword: str | None
    sidecars: SidecarLookup | None

    def keys(self, entity, script_file: str) -> list[ProvenanceKey]:
        """The keys of the entity's keyword lines of the keyword on its own
        script, else of its sidecar's; none where no keyword is named."""
        if self.keyword is None:
            return []
        return [
            key for line in entity.keyword_lines if line.keyword == self.keyword
            for key in line.keys if key.script_file == script_file
        ] or _sidecar_keyword_keys(self.sidecars, script_file, self.keyword)


def _sidecar_keyword_keys(
    sidecars: SidecarLookup | None, script_file: str, keyword: str,
) -> list[ProvenanceKey]:
    """The keys of ``script_file``'s keyword lines of ``keyword``, read off its
    sidecar; none where it has no sidecar."""
    if sidecars is None:
        return []
    try:
        sidecar = sidecars.get(script_file)
    except (KeyError, LookupError, OSError, ValueError):
        return []
    return [
        key for line in sidecar.lines
        if line.script_text and normalize_keyword(line.script_text) == keyword
        for key in line.provenance if key.trait_kind == "keyword"
    ]


def count_shard(built, shard: Path, *, salvage: bool = False) -> RuleCounts:
    """One shard's counts. Tags, review entries and carriers are held per game
    until the shard ends, when each game's source is known and a cut
    game (no game-end marker) is left out. ``salvage`` counts a damaged shard
    up to the damage and names it in ``damaged`` (a collection run must not
    stop on one); otherwise the damage raises, located."""
    from effects.infrastructure.shard_io import SHARD_DAMAGE, read_shard, take_decode_tally

    index, sidecars, keywords = built
    out = RuleCounts()
    seat_games: set[str] = set()
    ends = GameEnds()
    pending: list[_Pending] = []
    by_game = _GameReads()
    try:
        _count_records(read_shard(shard), by_game, index, sidecars, keywords, seat_games,
                       pending, ends)
    except SHARD_DAMAGE as exc:
        if not salvage:
            raise
        logger.warning("%s; its records before the damage are counted", exc)
        out.damaged.append(str(exc))
    cut = ends.cut
    out.cut_games = len(cut)
    for game, records in by_game.records.items():
        if game not in cut:
            out.records += records
            out.not_countable |= by_game.not_countable.get(game, set())
    for found in pending:
        if found.game_id in cut:
            continue
        out.review |= found.review
        out.carriers.update(found.carried)
        source = RuleSource.of(fork=found.fork, aimed=found.aimed,
                               random_seat_game=found.game_id in seat_games)
        for tag in found.tags:
            out.counts[(tag.rule, str(tag.polarity), str(source))] += 1
    out.decode = take_decode_tally()
    take_gaps = getattr(sidecars, "take_join_gaps", None)
    if take_gaps is not None:
        out.joins = take_gaps()
    return out


@dataclass
class _GameReads:
    """What each game of a shard read, held until the shard ends: its
    records and the rules its records could not count."""

    records: Counter = field(default_factory=Counter)
    not_countable: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Pending:
    """What one record adds, held until its game is known to have ended."""

    game_id: str
    fork: bool
    aimed: bool
    tags: tuple
    review: frozenset
    carried: frozenset


def _count_records(records, by_game: _GameReads, index, sidecars, keywords,
                   seat_games: set[str], pending: list, ends: GameEnds) -> None:
    for record in records:
        by_game.records[record.game_id] += 1
        ends.note(record)
        if record.random_seat:
            seat_games.add(record.game_id)
        result = tag_record(record, index, sidecars)
        if result.not_countable:
            by_game.not_countable.setdefault(record.game_id, set()).update(
                result.not_countable)
        review = frozenset((record.record_id, tag.scenario) for tag in result.review)
        carried = frozenset(
            (tag.rule, script_file, key)
            for tag in result.tags
            for script_file, key in tag_carriers(record, tag, keywords, sidecars)
        )
        if result.tags or review:
            pending.append(_Pending(record.game_id, counts_as_fork(record),
                                    record.aimed_at is not None,
                                    tuple(result.tags), review, carried))


def none_rules(catalogue) -> dict[str, str]:
    """Rules whose every scenario declares no detector, with the first reason
    given; a ``same_as`` scenario declares its rule's detector elsewhere."""
    reasons: dict[str, str] = {}
    declared: set[str] = set()
    for scenario in catalogue.scenarios:
        for rule in scenario.rules:
            if isinstance(scenario.detector, NoDetector):
                reasons.setdefault(rule, scenario.detector.reason)
            elif scenario.detector is not None:
                declared.add(rule)
    return {rule: reason for rule, reason in reasons.items() if rule not in declared}


@dataclass(frozen=True)
class RuleCountsConfig:
    records_dir: Path
    output: Path
    rules_dir: Path
    sidecar_roots: dict
    workers: int | None = None


def count(config: RuleCountsConfig) -> tuple[RuleCounts, CatalogueStanding]:
    """Count every shard under ``config.records_dir``; refuse a set that is not gen-3."""
    from effects.infrastructure.rules_catalogue_io import load_scenarios
    from effects.infrastructure.shard_io import NoShardsError, iter_shards, require_generation

    shards = iter_shards(config.records_dir)
    if not shards:
        raise NoShardsError(f"no shards under {config.records_dir}")
    require_generation(shards)
    logger.info("Counting %d gen-3 shard(s) under %s", len(shards), config.records_dir)
    total = RuleCounts(generation=str(ShardGeneration.GEN_3))
    for found in run_parallel(
        shards, count_shard, workers=config.workers, label="Counted",
        progress_every=50, context=CountContext(config.rules_dir, config.sidecar_roots),
    ):
        merge_into(total, found)
    return total, CatalogueStanding.of(load_scenarios(config.rules_dir))


def unit_of(rule: str) -> str:
    """A rule's catalogue unit: its section, or its second-level rule in 701 and 702."""
    section, _, rest = rule.partition(".")
    if section in ("701", "702"):
        return f"{section}.{''.join(c for c in rest if c.isdigit())}"
    return section


def report_lines(counts: dict) -> list[str]:
    """The printed table: one line per rule grouped by unit, then the rest."""
    def split(sides: dict, polarity: str) -> str:
        found = sides.get(polarity, {})
        return "/".join(str(found.get(str(s), 0)) for s in RuleSource)

    lines = [f"{'rule':12s} {'fires':>20s} {'does_not_fire':>20s}   "
             f"(each {' / '.join(str(s) for s in RuleSource)})"]
    unit = None
    for rule, sides in sorted(counts["rules"].items(), key=lambda item: rule_order(item[0])):
        if unit_of(rule) != unit:
            unit = unit_of(rule)
            lines.append(f"-- {unit}")
        lines.append(f"{rule:12s} {split(sides, Polarity.FIRES):>20s} "
                     f"{split(sides, Polarity.DOES_NOT_FIRE):>20s}")
    zero = counts.get("declared_zero", [])
    lines.append(f"rules declared with zero records: {len(zero)}")
    lines += [f"  {rule}" for rule in zero]
    untaggable = counts.get("untaggable", {})
    lines.append(f"scenarios no record can show (untaggable): {len(untaggable)}")
    lines += [f"  {sid} ({', '.join(found['rules'])}): {found['reason']}"
              for sid, found in untaggable.items()]
    same_as = counts.get("same_as", {})
    lines.append(f"scenarios counted under another (same_as): {len(same_as)}")
    lines += [f"  {sid} ({', '.join(found['rules'])}) -> {', '.join(found['same_as'])} "
              f"({', '.join(found['counted_under'])})" for sid, found in same_as.items()]
    lines.append(f"rules declared none: {len(counts['none'])}")
    lines += [f"  {rule}: {reason}" for rule, reason in counts["none"].items()]
    lines.append(f"rules not countable on these records: {len(counts['not_countable'])}")
    lines += [f"  {rule}" for rule in counts["not_countable"]]
    lines.append(f"records for review: {len(counts['review'])}")
    return lines


def rule_order(rule: str):
    import re

    m = re.match(r"(\d+)\.(\d+)([a-z]*)", rule)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else (0, 0, rule)


def cut_games_line(cut_games: int) -> str:
    """What a count says of the gen-3 games it left out as cut."""
    return (f"{cut_games} gen-3 game(s) without a game-end marker (cut) left out of "
            "every count, as build-corpus drops them")


def run(config: RuleCountsConfig) -> int:
    counts, standing = count(config)
    document = counts.to_json(standing)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    for line in report_lines(document):
        print(line)
    for line in [*counts.decode.report_lines(), *counts.joins.report_lines()]:
        print(line)
    if counts.cut_games:
        print(cut_games_line(counts.cut_games))
    print(f"{counts.records} records; counts written to {config.output}")
    return 0
