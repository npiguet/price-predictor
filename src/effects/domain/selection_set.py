"""The frozen selection set: what checkpoint selection and reporting read, per
stratum and half (FR-068a, FR-068b, FR-070–FR-073; research E1–E6;
contracts/selection-set.md).

``freeze_selection_set`` writes one per curated corpus, before any model
exists, to ``output/effects/reports/selection-set-<corpus digest>.json``;
training finds it there by the corpus digest, and every scorer reads its items
from it rather than choosing records again. It holds, for each validation
stratum and each half of it:

- ``probes``: the board-dependent probe items, by probe type, each a record,
  the entity it is read at (None at ``[ACT]``) and its label;
- ``grids``: each edit grid's records (one per game), the entities the grid
  edits, and its cells with the rule's mean outcome in each;
- ``rule_tests``: by rule and polarity, the rule-test records and the
  entities their tag names;
- ``deviance_samples``: the records the explained deviance reads, copies
  included, each copy named by its occurrence in the stratum;

beside each item's worst-case standard error (both rule-mask modes for an
output field) and the items left out, with why. Its ``digest`` is the sha256
of its canonical JSON with ``digest`` itself left out.

A copy is named ``(record_id, copy)``: ``copy`` counts the record's earlier
occurrences in the stratum read in shard path order, line by line, which is
the order every reader of the stratum reads it in. Rule tests, probes and
grids read each distinct record once (FR-068a) and name no copy.

Torch-free, and free of the probe scripts: ``train-effect-model`` reads it at
startup and the scripts read it through here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

from effects.domain.canonical_digest import canonical_sha256
from effects.domain.seeded_draw import keyed_hash

#: The balance ratio r (research E3): in a half, one polarity of a rule, one
#: probe type or one record-drawn grid cell holds at most r times the records
#: of the smallest it is balanced against.
BALANCE_RATIO = 4
#: Records per half the deviance sample holds at most (research E2).
DEVIANCE_SAMPLE_CAP = 16_384
#: The worst-case standard error each grid's records are sized to (T160): a
#: grid freezes the fewest of its records, by smallest keyed hash, that bring
#: its standard error to this, just under the item target of 0.1, since every
#: grid record is 64 or more forward records in every epoch (research E2, E4).
GRID_TARGET_ERROR = 0.09
#: Records a grid freezes at most, whatever its target asks: a grid whose rule
#: barely varies over its cells would otherwise ask for thousands, and is left
#: out at this many by its standard error instead.
GRID_RECORD_CAP = 512


def grid_records_needed(outcomes, *, target: float = GRID_TARGET_ERROR,
                        cap: int = GRID_RECORD_CAP) -> int:
    """How many records bring an edit grid's worst-case standard error
    ``1 / (2 √N B)`` (research E4) to ``target``, at most ``cap``; ``B`` is the
    error of predicting the grid's mean outcome in every cell. A grid whose
    outcome is the same in every cell (``B = 0``) needs none: no number of
    records bounds its error."""
    from effects.domain.score_scales import grid_mean_error

    baseline = grid_mean_error(outcomes)
    if not baseline > 0 or not target > 0:
        return 0
    return min(cap, max(1, math.ceil((1.0 / (2.0 * target * baseline)) ** 2)))

#: The four components of a stratum score, in the order an epoch line prints them.
COMPONENTS: tuple[str, ...] = ("probes", "grids", "rule_tests", "deviance")
PROBES, GRIDS, RULE_TESTS, DEVIANCE = COMPONENTS

#: A probe type scored by AUC, and one scored by R² (``labels.TargetSpec.kind``).
BINARY, AMOUNT = "binary", "amount"

#: Why a rule names no test in a half (FR-073).
UNTESTED = "untested: its detector finds no test record in this half"
#: Why a rule names no test in any half: its tagged records carry none of
#: the outputs its map entry names, so no output is left to read
#: (``CorpusManifest.effective_outputs``, FR-060a).
NO_READABLE_OUTPUT = "untested: its tagged records carry no output its map entry names"
#: Every reason a trained rule is left out untested.
UNTESTED_REASONS = frozenset({UNTESTED, NO_READABLE_OUTPUT})

T = TypeVar("T")


class SelectionSetError(ValueError):
    """A selection set that is missing, frozen against another corpus, or edited."""


# ── items ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    """One probe item: a record, the entity it is read at, and its label."""

    record_id: str
    game_id: str
    entity: str | None
    label: float

    def to_json(self) -> dict:
        return {"record_id": self.record_id, "game_id": self.game_id,
                "entity": self.entity, "label": self.label}

    @classmethod
    def from_json(cls, data: Mapping) -> ProbeRecord:
        return cls(data["record_id"], data["game_id"], data["entity"], float(data["label"]))


@dataclass(frozen=True, slots=True)
class TaggedRecord:
    """A rule-test record or a grid record: a record and the entities read on it."""

    record_id: str
    game_id: str
    entities: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {"record_id": self.record_id, "game_id": self.game_id,
                "entities": list(self.entities)}

    @classmethod
    def from_json(cls, data: Mapping) -> TaggedRecord:
        return cls(data["record_id"], data["game_id"], tuple(data["entities"]))


@dataclass(frozen=True, slots=True)
class GridCell:
    """One grid cell: its label and the rule's outcome there, averaged over the
    grid's records."""

    cell: str
    outcome: float

    def to_json(self) -> dict:
        return {"cell": self.cell, "outcome": self.outcome}

    @classmethod
    def from_json(cls, data: Mapping) -> GridCell:
        return cls(data["cell"], float(data["outcome"]))


@dataclass(frozen=True, slots=True)
class FrozenGrid:
    """An edit grid: every cell edits the same ``records``, so each cell holds
    them all (FR-068a's cell balance holds by construction)."""

    cells: tuple[GridCell, ...]
    records: tuple[TaggedRecord, ...]

    def to_json(self) -> dict:
        return {"cells": [c.to_json() for c in self.cells],
                "records": [r.to_json() for r in self.records]}

    @classmethod
    def from_json(cls, data: Mapping) -> FrozenGrid:
        return cls(tuple(GridCell.from_json(c) for c in data["cells"]),
                   tuple(TaggedRecord.from_json(r) for r in data["records"]))


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One copy of a record in a deviance sample."""

    record_id: str
    game_id: str
    copy: int

    def to_json(self) -> dict:
        return {"record_id": self.record_id, "game_id": self.game_id, "copy": self.copy}

    @classmethod
    def from_json(cls, data: Mapping) -> SampleRecord:
        return cls(data["record_id"], data["game_id"], int(data["copy"]))


@dataclass(frozen=True, slots=True)
class LeftOut:
    """An item a half's component leaves out, and why."""

    item: str
    reason: str

    def to_json(self) -> dict:
        return {"item": self.item, "reason": self.reason}

    @classmethod
    def from_json(cls, data: Mapping) -> LeftOut:
        return cls(data["item"], data["reason"])


def item_key(stratum: str, half: str, component: str, item: str) -> str:
    """An item's key in ``standard_errors``: ``stratum/half/component/item``."""
    return f"{stratum}/{half}/{component}/{item}"


#: The per-entity head, as ``loss_floor`` and the loss denominators key it.
PER_ENTITY_HEAD = "per_entity"


def deviance_item(head: str, term: str) -> str:
    """A deviance item's name: a per-entity field (the gate included) by its
    own name, any other head's term as ``<head>.<term>`` (``verdict.can_play``,
    ``global.day_night_change``, ``created_objects.count``), the terms the loss
    denominators divide by (``loss_floor.constant_predictor_losses``)."""
    return str(term) if head == PER_ENTITY_HEAD else f"{head}.{term}"


def deviance_head(item: str) -> str:
    """The head a deviance item belongs to: the part before its first ``.``,
    or the per-entity head for a bare field name (no per-entity field's name
    holds a ``.``)."""
    head, dot, _ = item.partition(".")
    return head if dot else PER_ENTITY_HEAD


def deviance_by_head(items: Mapping[str, float]) -> dict[str, tuple[float, int]]:
    """Scored deviance items grouped by head: ``head -> (mean, items)``, the
    heads in loss-denominator order (per-entity, verdict, created-objects,
    global), a head with no item absent."""
    grouped: dict[str, list[float]] = {}
    for item, value in items.items():
        grouped.setdefault(deviance_head(item), []).append(float(value))
    order = {head: i for i, head in enumerate(DEVIANCE_HEADS)}
    return {head: (sum(values) / len(values), len(values))
            for head, values in sorted(grouped.items(), key=lambda kv: order.get(kv[0], 9))}


#: Every head whose terms are deviance items, in ``loss_floor.FLOOR_HEADS``'s order.
DEVIANCE_HEADS: tuple[str, ...] = (PER_ENTITY_HEAD, "verdict", "created_objects", "global")


def deviance_items(losses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """``head -> term -> X`` flattened to ``item -> X`` by :func:`deviance_item`."""
    return {deviance_item(head, term): value
            for head, terms in losses.items() for term, value in terms.items()}


#: A standard error as written: a number, None for an infinite one (no JSON
#: number is), or ``{mode: ...}`` for an output field under each rule mask.
StandardError = float | None | dict[str, float | None]


def error_value(value: float) -> float | None:
    """A standard error as JSON holds it: None for an infinite one."""
    return None if math.isinf(value) else float(value)


def error_of(value: float | None) -> float:
    """A written standard error read back; None is infinite."""
    return math.inf if value is None else float(value)


# ── the set ──────────────────────────────────────────────────────────────

#: ``stratum -> half -> name -> X``.
ByHalf = dict[str, dict[str, dict[str, T]]]


def _by_half(data: Mapping, read: Callable[[Any], T]) -> ByHalf:
    return {stratum: {half: {name: read(value) for name, value in items.items()}
                      for half, items in halves.items()}
            for stratum, halves in data.items()}


def _write_by_half(data: Mapping, write: Callable[[T], Any]) -> dict:
    return {stratum: {half: {name: write(value) for name, value in sorted(items.items())}
                      for half, items in sorted(halves.items())}
            for stratum, halves in sorted(data.items())}


@dataclass(frozen=True)
class SelectionSet:
    """Everything checkpoint selection and reporting read, frozen with its digest.

    ``probe_kinds`` says how each probe type is scored (``binary`` by AUC,
    ``amount`` by R²); every other field is data-model.md's.
    """

    corpus_digest: str
    r: int
    #: ``stratum -> half -> game ids``, the manifest's, sorted.
    halves: dict[str, dict[str, tuple[str, ...]]]
    probes: ByHalf[tuple[ProbeRecord, ...]]
    probe_kinds: dict[str, str]
    grids: ByHalf[FrozenGrid]
    #: ``stratum -> half -> rule -> polarity -> records``.
    rule_tests: ByHalf[dict[str, tuple[TaggedRecord, ...]]]
    #: ``stratum -> half -> records``.
    deviance_samples: dict[str, dict[str, tuple[SampleRecord, ...]]]
    standard_errors: dict[str, StandardError]
    #: ``stratum -> half -> component -> items left out``.
    left_out: ByHalf[tuple[LeftOut, ...]]
    #: The thresholds the freeze ran under (research E5).
    settings: dict[str, Any] = field(default_factory=dict)
    digest: str = ""

    def to_json(self) -> dict:
        """The set as written; ``digest`` only once it has one."""
        out = {
            "corpus_digest": self.corpus_digest,
            "r": self.r,
            "halves": {s: {h: list(g) for h, g in sorted(by.items())}
                       for s, by in sorted(self.halves.items())},
            "probes": _write_by_half(self.probes, lambda rows: [r.to_json() for r in rows]),
            "probe_kinds": dict(sorted(self.probe_kinds.items())),
            "grids": _write_by_half(self.grids, FrozenGrid.to_json),
            "rule_tests": _write_by_half(self.rule_tests, lambda sides: {
                polarity: [r.to_json() for r in rows] for polarity, rows in sorted(sides.items())
            }),
            "deviance_samples": {s: {h: [r.to_json() for r in rows]
                                     for h, rows in sorted(by.items())}
                                 for s, by in sorted(self.deviance_samples.items())},
            "standard_errors": dict(sorted(self.standard_errors.items())),
            "left_out": _write_by_half(self.left_out, lambda rows: [r.to_json() for r in rows]),
            "settings": dict(sorted(self.settings.items())),
        }
        if self.digest:
            out["digest"] = self.digest
        return out

    @classmethod
    def from_json(cls, data: Mapping) -> SelectionSet:
        return cls(
            corpus_digest=data["corpus_digest"],
            r=int(data["r"]),
            halves={s: {h: tuple(g) for h, g in by.items()} for s, by in data["halves"].items()},
            probes=_by_half(data["probes"],
                            lambda rows: tuple(ProbeRecord.from_json(r) for r in rows)),
            probe_kinds=dict(data["probe_kinds"]),
            grids=_by_half(data["grids"], FrozenGrid.from_json),
            rule_tests=_by_half(data["rule_tests"], lambda sides: {
                polarity: tuple(TaggedRecord.from_json(r) for r in rows)
                for polarity, rows in sides.items()
            }),
            deviance_samples={s: {h: tuple(SampleRecord.from_json(r) for r in rows)
                                  for h, rows in by.items()}
                              for s, by in data["deviance_samples"].items()},
            standard_errors=dict(data["standard_errors"]),
            left_out=_by_half(data["left_out"],
                              lambda rows: tuple(LeftOut.from_json(r) for r in rows)),
            settings=dict(data.get("settings", {})),
            digest=data.get("digest", ""),
        )

    def computed_digest(self) -> str:
        """sha256 of the canonical JSON (sorted keys, no whitespace), ``digest`` excluded."""
        return canonical_digest(self.to_json())

    def sealed(self) -> SelectionSet:
        """The set with its digest computed over everything else."""
        return replace(self, digest=self.computed_digest())

    def record_ids(self, stratum: str, half: str) -> frozenset[str]:
        """Every record one half's probes, grids and rule tests read, once each."""
        ids = {r.record_id for rows in self.probes.get(stratum, {}).get(half, {}).values()
               for r in rows}
        ids |= {r.record_id for grid in self.grids.get(stratum, {}).get(half, {}).values()
                for r in grid.records}
        ids |= {r.record_id for sides in self.rule_tests.get(stratum, {}).get(half, {}).values()
                for rows in sides.values() for r in rows}
        return frozenset(ids)


def canonical_digest(payload: Mapping) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace), ``digest`` excluded."""
    return canonical_sha256({k: v for k, v in payload.items() if k != "digest"})


# ── the file ─────────────────────────────────────────────────────────────


def selection_set_path(reports_dir: Path, corpus_digest: str) -> Path:
    """Where the set frozen against the corpus of ``corpus_digest`` lives."""
    return Path(reports_dir) / f"selection-set-{corpus_digest}.json"


def write_selection_set(path: Path, selection: SelectionSet) -> Path:
    """Write ``selection`` sealed, as indented JSON with sorted keys."""
    sealed = selection if selection.digest else selection.sealed()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(sealed.to_json(), handle, indent=1, sort_keys=True)
        handle.write("\n")
    return path


def load_selection_set(path: Path, corpus_digest: str) -> SelectionSet:
    """The set frozen for this corpus; refuses a missing, mismatched or edited one.

    The digest is checked over what :class:`SelectionSet` reads back, so a
    field it would drop fails here rather than vanishing from the next set
    written.
    """
    path = Path(path)
    if not path.exists():
        raise SelectionSetError(
            f"no selection set at {path}: freeze one with "
            "`scripts/effect_knowledge_probes/run.py --freeze-probe-set --corpus DIR` "
            "(once per curated corpus)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("corpus_digest") != corpus_digest:
        raise SelectionSetError(
            f"{path} was frozen against corpus digest {data.get('corpus_digest')}, "
            f"not this corpus's {corpus_digest}")
    selection = SelectionSet.from_json(data)
    if data.get("digest") != selection.computed_digest():
        raise SelectionSetError(f"{path} does not match its own digest; it was edited")
    return selection


# ── the seeded balance ───────────────────────────────────────────────────


def balanced(
    groups: Mapping[Hashable, Sequence[T]], r: int, *, seed: int, salt: str,
    key: Callable[[T], str],
) -> dict[Hashable, list[T]]:
    """Each group cut to at most ``r`` times the smallest non-empty group, by a
    seeded draw: the items of smallest ``keyed_hash(salt|key(item), seed)``
    stay, in their original order. An empty group bounds nothing."""
    sizes = [len(items) for items in groups.values() if items]
    if not sizes:
        return {name: list(items) for name, items in groups.items()}
    limit = r * min(sizes)
    out: dict[Hashable, list[T]] = {}
    for name, items in groups.items():
        items = list(items)
        if len(items) <= limit:
            out[name] = items
            continue
        kept = set(sorted(range(len(items)), key=lambda i: (
            keyed_hash(f"{salt}|{key(items[i])}", seed), i))[:limit])
        out[name] = [item for i, item in enumerate(items) if i in kept]
    return out
