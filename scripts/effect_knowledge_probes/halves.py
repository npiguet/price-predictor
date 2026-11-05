"""The frozen selection set's halves, as a probing run reads and scores them
(FR-070, FR-074; research E1, E5; contracts/selection-set.md).

Every board-dependent probe of a run reads the records of one validation
stratum's games in one half, selection or reporting, as the frozen selection
set (``effects.domain.selection_set``) names them, and reports each half on
its own (story 8 acceptance 6). :func:`read_half` reads a half's games once,
each record once — a validation stratum holds repeated rows of one record
(added copies, FR-068a), and every reader here takes its first — and keeps
what the run's probes ask of it:

- the frozen probe items, relabelled where they were frozen
  (``labels.record_items``) so a trigger keeps its cause and an item its fold
  group, the label the frozen one;
- the frozen grids' records with their entities;
- the gen-3 probes' qualifying records (``conditions``, ``anthem``,
  ``target_restrictions``), at most ``per_game`` a game and ``limit`` a
  probe, by smallest ``crc32`` of record id;
- the sweeps' candidates (``sweeps.SweepSelector``), the trigger joins of the
  interaction pairs and the observed outcomes of the line labels.

:func:`score_grids` scores each frozen grid as the control score does: each
cell's mean prediction against the rule's outcome there, ``1 − MAE / B``.
:func:`score_rule_tests` scores a half's rule tests through the control
score's own scorer (``effects.application.control_score.score_half``), which
reads the half's records itself.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from effect_knowledge_probes import common, readout

#: The two halves of a validation stratum, as the selection set names them.
SELECTION, REPORTING = "selection", "reporting"
HALVES: tuple[str, ...] = (SELECTION, REPORTING)

#: Qualifying records each gen-3 probe reads per half, and from one game.
PROBE_RECORDS = 1000
PROBE_PER_GAME = 3


def load_selection(context, directory=None):
    """The selection set frozen for ``context``'s corpus under ``directory``
    (default ``common.REPORTS``); exits naming the file when it is missing,
    frozen against another corpus or edited."""
    from effects.domain.selection_set import (
        SelectionSetError,
        load_selection_set,
        selection_set_path,
    )

    path = selection_set_path(directory or common.REPORTS, context.corpus_digest)
    try:
        return load_selection_set(path, context.corpus_digest)
    except SelectionSetError as error:
        raise SystemExit(str(error)) from None


def distinct(records: Iterable) -> Iterator:
    """Each record's first row: a validation stratum or the training stratum
    repeats a record's row for every copy curation wrote of it, and a probe
    reads a record once (FR-070)."""
    seen: set[str] = set()
    for record in records:
        if record.record_id not in seen:
            seen.add(record.record_id)
            yield record


def half_records(context, stratum: str, games, record_ids=None) -> Iterator:
    """Every distinct record of ``games`` in ``stratum``, narrowed to
    ``record_ids``, in shard path order."""
    from effects.infrastructure.shard_io import iter_shards

    shards = iter_shards(context.stratum_dir(stratum))
    return distinct(record for _, record in common.read_games(shards, set(games), record_ids))


@dataclass
class Capped:
    """At most ``per_game`` values a game and ``limit`` in all, those of the
    smallest ``crc32`` of record id, so a rebuild keeping a record keeps it."""

    limit: int = PROBE_RECORDS
    per_game: int = PROBE_PER_GAME
    _games: dict[str, list[tuple[int, object]]] = field(default_factory=lambda: defaultdict(list))

    def add(self, record, value) -> None:
        kept = self._games[record.game_id]
        kept.append((common.stable_hash(record.record_id), value))
        if len(kept) > self.per_game:
            kept.sort(key=lambda pair: pair[0])
            del kept[self.per_game:]

    def values(self) -> list:
        every = sorted((pair for kept in self._games.values() for pair in kept),
                       key=lambda pair: pair[0])
        return [value for _, value in every[: self.limit]]


@dataclass
class HalfRead:
    """What one half of one validation stratum gives a probing run."""

    stratum: str
    half: str
    games: int = 0
    #: ``record_id -> RecordItem list`` of the frozen probe types the run reads.
    probe_items: dict[str, list] = field(default_factory=dict)
    #: ``grid -> [(record, entities)]``.
    grids: dict[str, list[tuple[object, tuple[str, ...]]]] = field(default_factory=dict)
    conditions: list = field(default_factory=list)
    anthem: list = field(default_factory=list)
    targeting: list = field(default_factory=list)
    #: ``sweep -> [row]``, chosen by ``sweeps.SweepSelector``.
    sweep_rows: dict = field(default_factory=dict)
    #: The records every probe above reads, by id.
    records: dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.stratum}/{self.half}"

    def probe_records(self) -> list:
        return [self.records[rid] for rid in self.probe_items]


@dataclass
class ReadSettings:
    """What :func:`read_half` collects besides the frozen items."""

    probe_types: frozenset[str]
    gen3_probes: bool = True
    sweeps: bool = True
    limit: int = PROBE_RECORDS
    per_game: int = PROBE_PER_GAME


def read_half(context, selection, stratum: str, half: str, settings: ReadSettings, *,
              acting_text, keywords_of, observe: Callable[[object], None] | None = None
              ) -> HalfRead:
    """Read one half of ``stratum`` once and keep what the run's probes ask
    of it; ``observe`` sees every distinct record (the line labels' observed
    outcomes, the interaction pairs' triggers)."""
    from effect_knowledge_probes import anthem, conditions, labels, sweeps, target_restrictions

    sidecars = context.sidecars()
    games = selection.halves.get(stratum, {}).get(half, ())
    read = HalfRead(stratum, half, games=len(games))
    frozen_probes: dict[str, dict[tuple[str, str | None], float]] = defaultdict(dict)
    for probe_type, rows in selection.probes.get(stratum, {}).get(half, {}).items():
        if probe_type in settings.probe_types:
            for row in rows:
                frozen_probes[row.record_id][(probe_type, row.entity)] = row.label
    grid_rows: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for name, grid in selection.grids.get(stratum, {}).get(half, {}).items():
        for row in grid.records:
            grid_rows[row.record_id].append((name, row.entities))

    join = labels.TriggerJoin()
    selector = sweeps.SweepSelector(sidecars) if settings.sweeps else None
    caps = {name: Capped(settings.limit, settings.per_game)
            for name in ("conditions", "anthem", "targeting")}
    kept: dict[str, object] = {}
    for record in half_records(context, stratum, games):
        rid = record.record_id
        join.observe(record)
        if observe is not None:
            observe(record)
        if selector is not None:
            selector.observe(record)
        if rid in frozen_probes or rid in grid_rows:
            kept[rid] = record
        for name, entities in grid_rows.get(rid, ()):
            read.grids.setdefault(name, []).append((record, entities))
        if settings.gen3_probes:
            found = conditions.record_items(record, sidecars, keywords_of)
            if found:
                caps["conditions"].add(record, found)
            chosen = anthem.chosen(record, sidecars)
            if chosen is not None:
                caps["anthem"].add(record, chosen)
            decision = target_restrictions.decision_items(record, sidecars)
            if decision is not None:
                caps["targeting"].add(record, (record, *decision))

    joined = join.result()
    for rid, wanted in frozen_probes.items():
        record = kept.get(rid)
        if record is None:
            continue
        items = [item for item in labels.record_items(record, sidecars, joined, acting_text)
                 if (item.target, item.entity) in wanted]
        # The frozen label is the item's: the freeze labelled the same record.
        items = [_relabelled(item, wanted[(item.target, item.entity)]) for item in items]
        if items:
            read.probe_items[rid] = items
    read.conditions = [item for found in caps["conditions"].values() for item in found]
    read.anthem = caps["anthem"].values()
    read.targeting = caps["targeting"].values()
    read.records = kept
    if selector is not None:
        read.sweep_rows = selector.result()
        wanted = {row["record_id"] for rows in read.sweep_rows.values() for row in rows}
        missing = wanted - set(kept)
        if missing:
            for record in half_records(context, stratum, games, missing):
                kept[record.record_id] = record
    return read


def _relabelled(item, label: float):
    from dataclasses import replace

    return item if item.label == label else replace(item, label=label)


# ── the grids ───────────────────────────────────────────────────────────


def grid_predictions(probe_model, name: str, record, entities: Sequence[str]) -> list[float]:
    """The model's answer in each cell of frozen grid ``name`` for one of its
    records, in the frozen cells' order: the cells the control score reads
    (``effects.domain.grid_edits.GridEdits``), each edited text read through
    the probe batcher's memo under its own key."""
    from effects.domain.grid_edits import GridEdits

    batcher = probe_model.batcher
    cells = GridEdits(batcher).cells(name, record, entities)
    for cell in cells:
        for edit in cell.texts:
            batcher.override_text(edit.key, edit.text, edit.like, script=edit.script,
                                  role_spans=edit.role_spans)
    (entity, name_read), = {(cell.entity, cell.readout) for cell in cells}
    return common.predict_all(probe_model, [cell.record for cell in cells],
                              readout.at_card(name_read, entity))


def score_grids(probe_model, selection, read: HalfRead) -> dict[str, dict]:
    """``{grid: {"records", "score", "mae", "baseline"}}`` for each frozen grid
    of the half: ``score`` is ``1 − MAE / B`` (NaN where the rule's outcome is
    the same in every cell), ``mae`` the cell means' mean absolute error
    against the rule, ``baseline`` B."""
    from effects.domain.score_scales import grid_mean_error, grid_score

    frozen = selection.grids.get(read.stratum, {}).get(read.half, {})
    out: dict[str, dict] = {}
    for name in sorted(read.grids):
        outcomes = np.array([cell.outcome for cell in frozen[name].cells], dtype=np.float64)
        rows = [grid_predictions(probe_model, name, record, entities)
                for record, entities in read.grids[name]]
        means = np.nanmean(np.array(rows, dtype=np.float64), axis=0)
        out[name] = {
            "records": len(rows),
            "score": float(grid_score(means, outcomes)),
            "mae": float(np.mean(np.abs(means - outcomes))),
            "baseline": float(grid_mean_error(outcomes)),
        }
    return out


# ── the rule tests ──────────────────────────────────────────────────────


#: A stratum's rule tests where the half holds none.
NO_RULE_TESTS: dict = {"rules": {}, "left_out": [], "component": float("nan")}


def score_rule_tests(context, selection, probe_model, half: str) -> dict[str, dict]:
    """``{stratum: {"rules", "left_out", "component"}}`` for the frozen rule
    tests of ``half`` of every stratum the selection set halves, scored by the
    control score's own scorer (FR-073, FR-074): ``rules`` each rule's mean
    over the map's outputs a head reads of ``2·AUC − 1`` (``fires`` ranked
    above ``does_not_fire``, the category and direction read off the records'
    own targets), clipped below at −1; ``left_out`` the rules with no output
    a head reads or no finite reading of both polarities; ``component`` their
    mean, NaN with none. Each distinct record is read once, so no rule mask
    weighs anything here."""
    from effects.application.control_score import score_half
    from effects.application.freeze_selection_set import stratum_reader
    from effects.domain.selection_set import RULE_TESTS

    readers = {stratum: stratum_reader(context.stratum_dir(stratum))
               for stratum in selection.halves}
    scored = score_half(
        selection, readers, half=half, encoder=probe_model.encoder, model=probe_model.model,
        batcher=probe_model.batcher,
        rule_outputs={rule: tuple(entry.outputs)
                      for rule, entry in context.manifest.rule_map.items()},
        mask=None,
        stratum_weight=0.5, deviance_floor=0.0, components=(RULE_TESTS,))
    return {stratum: {"rules": dict(score.items.get(RULE_TESTS, {})),
                      "left_out": list(score.left_out.get(RULE_TESTS, ())),
                      "component": score.components[RULE_TESTS]}
            for stratum, score in scored.strata.items()}


# ── the gen-3 probes of FR-074 ──────────────────────────────────────────


def score_gen3_probes(probe_model, read: HalfRead) -> dict[str, dict]:
    """The turn and step conditions, the anthem and the target-legality
    probes over one half's qualifying records."""
    from effect_knowledge_probes import anthem, conditions, target_restrictions

    out: dict[str, dict] = {}
    if read.conditions:
        out["conditions"] = conditions.score(
            read.conditions, conditions.predict_items(probe_model, read.conditions))
    if read.anthem:
        rows = [row for record, found in read.anthem for row in anthem.items(record, found)]
        result = anthem.score(rows, anthem.predict_items(probe_model, rows))
        out["anthem"] = {**result, "records": len(read.anthem)}
    if read.targeting:
        records = [row[0] for row in read.targeting]
        readings = target_restrictions.read_items(probe_model, records)
        families = target_restrictions.breakdowns(read.targeting, readings)["family"]
        names = target_restrictions.READOUTS
        out["target_legality"] = {
            "records": len(records),
            "auc": {name: float(common.auc(
                [y for cell in families.values() for y in cell[name][0]],
                [p for cell in families.values() for p in cell[name][1]])) for name in names},
            "by_family": {key: {"items": len(cell[names[0]][0]),
                                **{name: float(common.auc(*cell[name])) for name in names}}
                          for key, cell in sorted(families.items())},
        }
    return out
