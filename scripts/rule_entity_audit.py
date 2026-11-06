"""Audit where rule tags' entities fall against where their rules' outputs train (feature 025).

A rule tag (``RuleTag``) names entities, and two consumers read outputs at
them: the hard rule mask trains an added copy's affected gate only at its
drawn rule's ``rule_entities``, and the control score's rule tests read each
of the rule's map outputs (``rule_map[rule].outputs``) at the tag's entities.
Each detector family chooses those entities its own way: the keyword's
carrier, the creature that died, an event's subjects, ``[ACT]``. This script
measures, before any fix, how often a rule's outputs are actually trained at
the entities its tags name, against the record's other entities and its
affected set.

Tagging is ``tag_record`` over the catalogue's detectors, run once per
detector family (one ``DetectorIndex`` per family, cut from the catalogue's)
so a rule declared in several families is reported per family. A record is
read once, deduplicated by ``record_id`` across the strata. "Supervised" is
the per-entity target training reads (``derive_targets``, unmasked): an
entity with targets, the field present (written or silent under its scope and
the record's sampling class) and, for a gated field, the entity affected.
"Changed" is supervised at a value other than the field's silent one (any
written value for a field with none). Verdict, global and created-objects
outputs are record-level and reported per record.

    python -u scripts/rule_entity_audit.py \\
        --corpus output/effects/corpus-gen3-pilot3 \\
        --output output/effects/reports/rule-entity-audit
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain.aux_head_targets import (
    created_objects_targets,
    global_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.default_paths import DEFAULT_CARDS_FOLDER_PATHS
from effects.domain.effect_head_layout import (
    CREATED_OBJECT_SLOTS,
    CREATED_SLOT_WIDTH,
    FIELDS_BY_NAME,
    GLOBAL_LAYOUT,
    VERDICT_LAYOUT,
    CreatedSlot,
    FieldSpec,
)
from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.records import EffectRecord
from effects.domain.rule_detectors import DetectorIndex, RuleTag, tag_record
from effects.domain.rule_mask import CREATED_OBJECTS_OUTPUT, GATE_OUTPUT
from effects.domain.stratum_placement import rule_test_eligible

logger = logging.getLogger("rule_entity_audit")

#: The curated strata read by default: every distinct record once.
CORPUS_STRATA = ("training", "validation/card-disjoint", "validation/game-disjoint")

ACT = "[ACT]"
#: The three entity sets each per-entity output is measured over.
TAGGED, OTHER, AFFECTED = "tagged", "other", "affected"
SETS = (TAGGED, OTHER, AFFECTED)

#: How an output is read: at entities (a per-entity field or the gate), at
#: the record (verdict, global, created objects), or by no shipped head.
ENTITY, GATE, RECORD, UNREAD = "entity", "gate", "record", "unread"


def output_level(output: str) -> str:
    if output == GATE_OUTPUT:
        return GATE
    if output in FIELDS_BY_NAME:
        return ENTITY
    if (output == CREATED_OBJECTS_OUTPUT or output in VERDICT_LAYOUT.by_name
            or output in GLOBAL_LAYOUT.by_name):
        return RECORD
    return UNREAD


def output_need(output: str) -> str:
    """The kind of entity an output is read at: a per-entity field's scope,
    ``entity`` for the gate, ``record`` for a record-level head."""
    level = output_level(output)
    if level == ENTITY:
        return f"entity:{FIELDS_BY_NAME[output].scope.value}"
    return {GATE: "entity:any", RECORD: "record", UNREAD: "unread"}[level]


def entity_kind(record: EffectRecord, ref: str) -> str:
    """What one tagged ref is: ``[ACT]``, a player, a card by zone, or a ref
    the snapshot does not hold."""
    if ref == ACT:
        return "act"
    if any(player.id == ref for player in record.state.players):
        return "player"
    entity = record.state.entity(ref)
    if entity is None:
        return "off-snapshot"
    return f"card:{(entity.zone or 'unknown').lower()}"


def universe(record: EffectRecord) -> tuple[str, ...]:
    """Every per-entity target the record has a slot for: its cards and players."""
    return (*(e.id for e in record.state.entities), *(p.id for p in record.state.players))


def _differs(spec: FieldSpec, value) -> bool:
    if spec.silent is None:
        return True
    if isinstance(value, list):
        return any(float(v) != spec.silent for v in value)
    return float(value) != spec.silent


@dataclass(frozen=True, slots=True)
class Reading:
    """One output at one entity: whether training reads a target there,
    whether that target moved, and whether a rule test reads the gated
    field's off-gate zero there (gate trained, entity not affected)."""

    supervised: bool
    changed: bool
    gate_zero: bool = False


def entity_reading(targets: RecordTargets, output: str, ref: str, on_board: bool) -> Reading:
    """``output`` (a per-entity field or the gate) at ``ref``, as training reads it."""
    entry = targets.get(ref) if on_board else None
    if output == GATE_OUTPUT:
        supervised = on_board and targets.gated
        return Reading(supervised, supervised and entry is not None and entry.affected)
    spec = FIELDS_BY_NAME[output]
    affected = entry is not None and entry.affected
    gate_zero = on_board and spec.gated and targets.gated and not affected
    if entry is None:
        return Reading(False, False, gate_zero)
    value = entry.supervised(spec)
    if value is None or (spec.gated and not affected):
        return Reading(False, False, gate_zero)
    return Reading(True, _differs(spec, value))


def record_reading(record: EffectRecord, output: str) -> Reading:
    """A record-level output: whether the record supervises it, and whether
    its target is non-zero (a categorical's class other than the first)."""
    if output == CREATED_OBJECTS_OUTPUT:
        if not supervises_created_objects(record):
            return Reading(False, False)
        target, _known = created_objects_targets(record)
        present = any(target[slot * CREATED_SLOT_WIDTH + CreatedSlot.PRESENT]
                      for slot in range(CREATED_OBJECT_SLOTS))
        return Reading(True, bool(present))
    layout, derive = ((VERDICT_LAYOUT, verdict_targets) if output in VERDICT_LAYOUT.by_name
                      else (GLOBAL_LAYOUT, global_targets))
    row = derive(record)
    if row is None:
        return Reading(False, False)
    values, mask = row
    if not mask[layout.index[output]]:
        return Reading(False, False)
    first, last = layout.target_slices[output]
    return Reading(True, any(float(v) != 0 for v in values[first:last]))


# ── tallies ─────────────────────────────────────────────────────────────


@dataclass
class SetTally:
    pairs: int = 0
    supervised: int = 0
    changed: int = 0
    gate_zero: int = 0

    def add(self, reading: Reading) -> None:
        self.pairs += 1
        self.supervised += reading.supervised
        self.changed += reading.changed
        self.gate_zero += reading.gate_zero


@dataclass
class OutputTally:
    sets: dict[str, SetTally] = field(default_factory=lambda: {s: SetTally() for s in SETS})
    #: Records whose output changed at some entity, at a tagged one, and at
    #: an untagged one.
    changed_anywhere: int = 0
    changed_at_tagged: int = 0
    changed_at_other: int = 0
    #: Record-level outputs.
    record_supervised: int = 0
    record_changed: int = 0


@dataclass
class RuleTally:
    """One (rule, family, polarity)."""

    records: int = 0
    empty: int = 0
    forks: int = 0
    eligible: int = 0
    eligible_empty: int = 0
    kinds: Counter = field(default_factory=Counter)
    outputs: dict[str, OutputTally] = field(default_factory=lambda: defaultdict(OutputTally))


Key = tuple[str, str, str]


@dataclass
class Audit:
    rule_outputs: Mapping[str, tuple[str, ...]]
    tallies: dict[Key, RuleTally] = field(default_factory=lambda: defaultdict(RuleTally))
    #: Added copies per drawn rule, and those carrying no ``rule_entities``.
    added_copies: Counter = field(default_factory=Counter)
    added_empty: Counter = field(default_factory=Counter)
    records: int = 0

    def note_copy(self, record: EffectRecord) -> None:
        curation = record.curation
        if curation is None or not curation.is_added:
            return
        self.added_copies[curation.drawn_rule] += 1
        if not curation.rule_entities:
            self.added_empty[curation.drawn_rule] += 1

    def add(self, record: EffectRecord, tags: Iterable[tuple[str, RuleTag]]) -> None:
        """One distinct record and its ``(family, tag)`` pairs."""
        self.records += 1
        targets: RecordTargets | None = None
        board = universe(record)
        on_board = frozenset(board)
        for family, tag in tags:
            outputs = self.rule_outputs.get(tag.rule)
            if outputs is None:
                continue
            if targets is None:
                targets = derive_targets(record)
            tally = self.tallies[(tag.rule, family, str(tag.polarity))]
            tagged = tuple(dict.fromkeys(tag.entities))
            eligible = rule_test_eligible(record, tag)
            tally.records += 1
            tally.empty += not tagged
            tally.forks += bool(record.fork)
            tally.eligible += eligible
            tally.eligible_empty += eligible and not tagged
            tally.kinds.update(entity_kind(record, ref) for ref in tagged)
            tagged_set = frozenset(tagged)
            others = [ref for ref in board if ref not in tagged_set]
            affected = [ref for ref in board
                        if (entry := targets.get(ref)) is not None and entry.affected]
            for output in outputs:
                level = output_level(output)
                counted = tally.outputs[output]
                if level == RECORD:
                    reading = record_reading(record, output)
                    counted.record_supervised += reading.supervised
                    counted.record_changed += reading.changed
                    continue
                if level == UNREAD:
                    continue
                hit = False
                for ref in tagged:
                    reading = entity_reading(targets, output, ref, ref in on_board)
                    counted.sets[TAGGED].add(reading)
                    hit |= reading.changed
                elsewhere = False
                for ref in others:
                    reading = entity_reading(targets, output, ref, True)
                    counted.sets[OTHER].add(reading)
                    elsewhere |= reading.changed
                for ref in affected:
                    counted.sets[AFFECTED].add(entity_reading(targets, output, ref, True))
                counted.changed_anywhere += hit or elsewhere
                counted.changed_at_tagged += hit
                counted.changed_at_other += elsewhere


# ── tagging, per family ─────────────────────────────────────────────────


def family_indexes(index: DetectorIndex) -> dict[str, DetectorIndex]:
    """The catalogue's index cut into one index per detector family."""
    cut: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for kind, found in index.by_kind.items():
        for entry in found:
            cut[entry[1].family][kind].append(entry)
    return {family: DetectorIndex({kind: tuple(v) for kind, v in by_kind.items()},
                                  index.declared_rules)
            for family, by_kind in sorted(cut.items())}


def family_tagger(indexes: Mapping[str, DetectorIndex], lines) -> Callable[
        [EffectRecord], list[tuple[str, RuleTag]]]:
    def tag(record: EffectRecord) -> list[tuple[str, RuleTag]]:
        return [(family, found) for family, index in indexes.items()
                for found in tag_record(record, index, lines).tags]
    return tag


# ── reading ─────────────────────────────────────────────────────────────


def corpus_dirs(corpus: Path) -> list[Path]:
    return [corpus / stratum for stratum in CORPUS_STRATA if (corpus / stratum).is_dir()]


def distinct_records(dirs: Iterable[Path], on_copy: Callable[[EffectRecord], None]
                     ) -> Iterator[EffectRecord]:
    """Every record under ``dirs`` once by ``record_id``; ``on_copy`` sees every copy."""
    from effects.infrastructure.shard_io import iter_shards, read_shard

    seen: set[str] = set()
    for directory in dirs:
        shards = iter_shards(directory)
        logger.info("Reading %d shard(s) under %s", len(shards), directory)
        for shard in shards:
            for record in read_shard(shard):
                on_copy(record)
                if record.record_id in seen:
                    continue
                seen.add(record.record_id)
                yield record


# ── reports ─────────────────────────────────────────────────────────────


def _share(n: int, d: int) -> str:
    return f"{n / d:.4f}" if d else ""


CSV_COLUMNS = (
    "rule", "family", "polarity", "output", "level", "need", "records", "empty_entity_share",
    "fork_records", "eligible_records", "eligible_empty_share",
    "tagged_pairs", "tagged_supervised", "tagged_changed", "tagged_gate_zero",
    "other_pairs", "other_supervised", "other_changed",
    "affected_pairs", "affected_supervised", "affected_changed",
    "records_changed_anywhere", "hit_rate", "other_hit_rate", "record_supervised",
    "record_changed",
)


def csv_rows(audit: Audit) -> Iterator[dict]:
    for (rule, family, polarity), tally in sorted(audit.tallies.items()):
        for output in audit.rule_outputs[rule] or ("",):
            counted = tally.outputs.get(output)
            row = {"rule": rule, "family": family, "polarity": polarity, "output": output,
                   "level": output_level(output) if output else "",
                   "need": output_need(output) if output else "",
                   "records": tally.records,
                   "empty_entity_share": _share(tally.empty, tally.records),
                   "fork_records": tally.forks, "eligible_records": tally.eligible,
                   "eligible_empty_share": _share(tally.eligible_empty, tally.eligible)}
            if counted is not None:
                for name in SETS:
                    held = counted.sets[name]
                    row[f"{name}_pairs"] = held.pairs
                    row[f"{name}_supervised"] = _share(held.supervised, held.pairs)
                    row[f"{name}_changed"] = _share(held.changed, held.pairs)
                row["tagged_gate_zero"] = _share(counted.sets[TAGGED].gate_zero,
                                                 counted.sets[TAGGED].pairs)
                row["records_changed_anywhere"] = counted.changed_anywhere
                row["hit_rate"] = _share(counted.changed_at_tagged, counted.changed_anywhere)
                row["other_hit_rate"] = _share(counted.changed_at_other, counted.changed_anywhere)
                if row["level"] == RECORD:
                    row["record_supervised"] = _share(counted.record_supervised, tally.records)
                    row["record_changed"] = _share(counted.record_changed, tally.records)
            yield row


def write_csv(path: Path, rows: Iterable[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True, slots=True)
class Miss:
    """One (rule, family, polarity)'s per-entity outputs, summarized."""

    rule: str
    family: str
    polarity: str
    records: int
    empty: float
    #: Tags a rule test may draw (``rule_test_eligible``).
    eligible: int
    #: Mean over per-entity outputs (gate excluded) of each set's supervised share.
    tagged: float
    other: float
    affected: float
    #: Mean hit rate over outputs that moved in some record; None when none did.
    hit: float | None
    outputs: str


def misses(audit: Audit) -> list[Miss]:
    found = []
    for (rule, family, polarity), tally in audit.tallies.items():
        fields = [o for o in audit.rule_outputs[rule] if output_level(o) == ENTITY]

        def mean(name: str, outputs=fields, tally=tally) -> float:
            shares = [tally.outputs[o].sets[name].supervised / tally.outputs[o].sets[name].pairs
                      for o in outputs if tally.outputs[o].sets[name].pairs]
            return sum(shares) / len(shares) if shares else float("nan")

        hits = [tally.outputs[o].changed_at_tagged / tally.outputs[o].changed_anywhere
                for o in fields if tally.outputs[o].changed_anywhere]
        found.append(Miss(
            rule, family, polarity, tally.records, tally.empty / tally.records, tally.eligible,
            mean(TAGGED), mean(OTHER), mean(AFFECTED),
            sum(hits) / len(hits) if hits else None,
            "  ".join(_output_detail(o, tally) for o in audit.rule_outputs[rule])))
    return found


def _output_detail(output: str, tally: RuleTally) -> str:
    """One output of one row: ``t`` supervised/changed at the tagged entities
    (``z`` the share a rule test reads as the gated field's off-gate zero),
    ``a`` the same over the affected set, ``h`` the share of records whose
    output moved that moved at a tagged entity and ``o`` at an untagged one;
    a record-level output's ``r`` supervised/non-zero per record."""
    level = output_level(output)
    counted = tally.outputs.get(output)
    if level == UNREAD or counted is None:
        return f"{output}[unread]"
    if level == RECORD:
        return (f"{output}[r{_pct(_ratio(counted.record_supervised, tally.records)).strip()}/"
                f"{_pct(_ratio(counted.record_changed, tally.records)).strip()}]")
    tagged, affected = counted.sets[TAGGED], counted.sets[AFFECTED]
    moved = counted.changed_anywhere
    return (f"{output}[t{_pct(_ratio(tagged.supervised, tagged.pairs)).strip()}/"
            f"{_pct(_ratio(tagged.changed, tagged.pairs)).strip()}"
            f" z{_pct(_ratio(tagged.gate_zero, tagged.pairs)).strip()}"
            f" a{_pct(_ratio(affected.supervised, affected.pairs)).strip()}/"
            f"{_pct(_ratio(affected.changed, affected.pairs)).strip()}"
            f" h{_pct(_ratio(counted.changed_at_tagged, moved)).strip()}"
            f" o{_pct(_ratio(counted.changed_at_other, moved)).strip()} n{moved}]")


def _ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def _pct(value: float | None) -> str:
    if value is None or value != value:
        return "   -"
    return f"{100 * value:4.0f}"


def summary_lines(audit: Audit, verdicts: Mapping[str, str]) -> list[str]:
    lines = [f"{audit.records} distinct records; {len(audit.tallies)} (rule, family, polarity) "
             "rows over trained rules.",
             "Shares in percent. tag/oth/aff = mean over the rule's per-entity outputs (gate "
             "excluded) of the share of (record, entity) pairs supervised in that set; hit = "
             "mean over those outputs of the share of records whose output changed somewhere "
             "that changed at a tagged entity. Ranked by hit, then tag, ascending (worst first).",
             "Per output: t supervised/changed at tagged pairs, z share of tagged pairs a rule "
             "test reads as the gated field's off-gate zero (gate trained, entity unaffected), "
             "a supervised/changed over the affected set, h/o share of the n records where the "
             "output changed that changed at a tagged / an untagged entity; record-level "
             "outputs r supervised/non-zero per record."]
    by_family: dict[str, list[Miss]] = defaultdict(list)
    for miss in misses(audit):
        by_family[miss.family].append(miss)
    kinds_by_family: dict[str, Counter] = defaultdict(Counter)
    needs_by_family: dict[str, Counter] = defaultdict(Counter)
    for (rule, family, _), tally in audit.tallies.items():
        kinds_by_family[family].update(tally.kinds)
        if tally.empty:
            kinds_by_family[family]["(empty tag)"] += tally.empty
        needs_by_family[family].update(output_need(o) for o in audit.rule_outputs[rule])
    families_of: dict[str, set] = defaultdict(set)
    for rule, family, _ in audit.tallies:
        families_of[rule].add(family)
    for family in sorted(by_family):
        rows = by_family[family]
        lines += ["", f"== {family}: {len({m.rule for m in rows})} rules, "
                      f"{sum(m.records for m in rows)} tags"]
        lines.append("  tagged kinds: " + ", ".join(
            f"{k} {n}" for k, n in kinds_by_family[family].most_common()))
        lines.append("  outputs need: " + ", ".join(
            f"{k} {n}" for k, n in needs_by_family[family].most_common()))
        lines.append(f"  {'rule':12s} {'pol':13s} {'recs':>6s} {'elig':>6s} {'empty':>5s} "
                     f"{'tag':>4s} {'oth':>4s} {'aff':>4s} {'hit':>4s}  outputs")
        rows.sort(key=lambda m: (m.hit if m.hit is not None else 2.0,
                                 m.tagged if m.tagged == m.tagged else 2.0, m.rule))
        for m in rows:
            lines.append(f"  {m.rule:12s} {m.polarity:13s} {m.records:6d} {m.eligible:6d} "
                         f"{_pct(m.empty):>5s} {_pct(m.tagged)} {_pct(m.other)} "
                         f"{_pct(m.affected)} {_pct(m.hit)}  {m.outputs}")
    several = sorted(rule for rule, found in families_of.items() if len(found) > 1)
    lines += ["", f"Rules tagged by several families ({len(several)}): "
              + ", ".join(f"{r} [{'/'.join(sorted(families_of[r]))}]" for r in several)]
    lines += ["", "Added copies with empty rule_entities, by drawn rule (as built):"]
    total = sum(audit.added_copies.values())
    empty = sum(audit.added_empty.values())
    lines.append(f"  {empty} of {total} added copies")
    for rule, n in audit.added_empty.most_common():
        lines.append(f"  {rule:12s} {n:5d} of {audit.added_copies[rule]:5d}  "
                     f"[{verdicts.get(rule, '?')}] {','.join(audit.rule_outputs.get(rule, ()))}")
    return lines


# ── entry point ─────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, default=Path("output/effects/corpus-gen3-pilot3"),
                        help="A curated corpus; its strata and manifest are read")
    parser.add_argument("--records-dir", type=Path, default=None,
                        help="A raw records directory instead of the corpus's strata "
                             "(the rule map still comes from --manifest)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="The manifest whose rule_map is read (default: the corpus's)")
    parser.add_argument("--rules-dir", type=Path, default=Path("rules"))
    parser.add_argument("--cards-folder", dest="cards_folders", type=Path, action="append")
    parser.add_argument("--output", type=Path,
                        default=Path("output/effects/reports/rule-entity-audit"))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    from effects.infrastructure.converted_trees import SourceTrees, process_sidecars
    from effects.infrastructure.rules_catalogue_io import load_scenarios

    started = time.monotonic()
    manifest = json.loads((args.manifest or args.corpus / "manifest.json").read_text("utf-8"))
    rule_map = manifest["rule_map"]
    verdicts = {rule: entry.get("verdict", "") for rule, entry in rule_map.items()}
    trained = {rule: tuple(entry.get("outputs", ())) for rule, entry in rule_map.items()
               if entry.get("verdict") == "trained"}
    variant = manifest.get("variant_scripts") or None
    roots = SourceTrees.of(args.cards_folders or DEFAULT_CARDS_FOLDER_PATHS,
                           variant).sidecar_roots()
    indexes = family_indexes(DetectorIndex.of(load_scenarios(args.rules_dir)))
    tag = family_tagger(indexes, process_sidecars(roots))
    audit = Audit(trained)
    dirs = [args.records_dir] if args.records_dir else corpus_dirs(args.corpus)
    for n, record in enumerate(distinct_records(dirs, audit.note_copy), 1):
        audit.add(record, tag(record))
        if n % 5000 == 0:
            logger.info("Audited %d records (%.0f s)", n, time.monotonic() - started)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "rule-outputs.csv", csv_rows(audit), CSV_COLUMNS)
    lines = summary_lines(audit, verdicts)
    lines.append(f"\nRun time {time.monotonic() - started:.0f} s.")
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
