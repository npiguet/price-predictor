"""Whether every provenance key the window names joins its own sidecar."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import NamedTuple

from effects.application.corpus_validation.findings import (
    Finding,
    Thresholds,
    keep_example,
    share_of,
)
from effects.domain.provenance import KeyResolution, ProvenanceKey, SidecarLookup, TraitKind
from effects.domain.records import (
    EffectRecord,
    key_sites,
)


class JoinOutcome(Enum):
    """How one provenance key meets the sidecar of the card it names."""

    JOINED = auto()
    UNJOINABLE = auto()
    #: Outside every ``(face, trait_kind)`` range its sidecar declares: a
    #: trait Forge attached at runtime, neither a join nor a mismatch.
    RUNTIME_ONLY = auto()
    #: In a tree this run was not given (such as ``variant-scripts`` without
    #: ``--variant-scripts``): unchecked, never a mismatch.
    UNCHECKED = auto()


class _Join(NamedTuple):
    outcome: JoinOutcome
    #: Why an unjoinable key does not join.
    why: str = ""


def _join_result(sidecars: SidecarLookup, key: ProvenanceKey) -> _Join:
    """How ``key`` joins.

    ``resolution_of`` rather than ``row_for``, which answers ``None`` both to a
    dropped line (a join) and to a key outside every range (not one).
    """
    try:
        resolution = sidecars.resolution_of(key)
    except KeyError as exc:
        return _Join(JoinOutcome.UNJOINABLE, f"{key.label}: {exc}")
    if resolution is KeyResolution.UNCONFIGURED_TREE:
        return _Join(JoinOutcome.UNCHECKED)
    if resolution is KeyResolution.UNCONVERTED:
        return _Join(
            JoinOutcome.UNJOINABLE, f"{key.script_file}: no sidecar beside the converted card",
        )
    if resolution is KeyResolution.RUNTIME_ONLY:
        return _Join(JoinOutcome.RUNTIME_ONLY)
    return _Join(JoinOutcome.JOINED)


_KEYWORD_JOIN = "keyword provenance keys join their sidecar"


_OTHER_JOIN = "non-keyword provenance keys join their sidecar"


_OUTSIDE_JOIN = "provenance keys outside every declared range"


class KeyJoins:
    """Every provenance key the window names resolves in its own sidecar.

    A key joined to the wrong printed line is invisible downstream. ``keyword``
    keys are judged: their ordinal is derived rather than positional, and a
    converter and collector numbering keywords differently is the known
    failure. Every other trait kind is watched, its healthy rate unmeasured. A
    third, watched finding counts keys outside every declared range: not a
    mismatch, but a sidecar declaring no keyword line puts every keyword key
    of its face there, so the keyword finding names that count too.
    """

    def __init__(self, sidecars: SidecarLookup | None) -> None:
        self.sidecars = sidecars
        #: Every distinct printed-line key the window named, joined once.
        self.keys: set[ProvenanceKey] = set()

    def add(self, record: EffectRecord) -> None:
        self.keys.update(key_sites(record))

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        if self.sidecars is None:
            unchecked = Finding.watch(
                name=_KEYWORD_JOIN,
                measured=(
                    f"{len(self.keys)} distinct keys, none checked: no converted "
                    "tree was readable (pass --cards-folder)"
                ),
            )
            yield unchecked
            yield replace(unchecked, name=_OTHER_JOIN)
            yield replace(unchecked, name=_OUTSIDE_JOIN)
            return
        tally = _JoinTally()
        ordered = sorted(
            self.keys,
            key=lambda k: (k.script_file, k.trait_kind, k.index_within_kind),
        )
        for key in ordered:
            tally.add(key, _join_result(self.sidecars, key))
        yield tally.keyword_finding()
        yield tally.other_finding()
        yield tally.outside_finding()


@dataclass(slots=True)
class _TraitTally:
    """One partition of a window's checked keys, by trait kind."""

    checked: Counter[str] = field(default_factory=Counter)
    unjoinable: Counter[str] = field(default_factory=Counter)
    runtime_only: Counter[str] = field(default_factory=Counter)
    #: Why the first few unjoinable keys failed.
    examples: list[str] = field(default_factory=list)

    def add(self, key: ProvenanceKey, join: _Join) -> None:
        self.checked[key.trait_kind] += 1
        if join.outcome is JoinOutcome.RUNTIME_ONLY:
            self.runtime_only[key.trait_kind] += 1
        elif join.outcome is JoinOutcome.UNJOINABLE:
            self.unjoinable[key.trait_kind] += 1
            keep_example(self.examples, join.why)


@dataclass(slots=True)
class _JoinTally:
    """One join of a window's distinct keys: the judged ``keyword`` keys apart
    from the watched rest."""

    keyword: _TraitTally = field(default_factory=_TraitTally)
    other: _TraitTally = field(default_factory=_TraitTally)
    unchecked_by_tree: Counter[str] = field(default_factory=Counter)

    def add(self, key: ProvenanceKey, join: _Join) -> None:
        if join.outcome is JoinOutcome.UNCHECKED:
            self.unchecked_by_tree[key.tree] += 1
        else:
            (self.keyword if key.trait_kind == TraitKind.KEYWORD else self.other).add(key, join)

    def keyword_finding(self) -> Finding:
        broken = self.keyword.unjoinable.total()
        trees = tuple(
            f"{count} keys in {tree}, a tree this run was not given"
            for tree, count in sorted(self.unchecked_by_tree.items())
        )
        return Finding(
            name=_KEYWORD_JOIN,
            ok=broken == 0,
            measured=(
                f"{broken}/{self.keyword.checked.total()} keyword keys fail to "
                f"join, {self.keyword.runtime_only.total()} outside every "
                f"declared range, {self.unchecked_by_tree.total()} unchecked"
            ),
            detail=trees + tuple(self.keyword.examples),
        )

    def other_finding(self) -> Finding:
        other = self.other
        other_checked = other.checked.total()
        other_broken = other.unjoinable.total()
        other_rate = share_of(other_broken, other_checked)
        return Finding.watch(
            name=_OTHER_JOIN,
            measured=(
                f"{other_broken}/{other_checked} keys of every other trait "
                f"kind fail to join ({other_rate:.1%}, watched, no ceiling)"
            ),
            detail=tuple(
                f"{kind}: {other.unjoinable[kind]}/{count} unjoinable"
                for kind, count in sorted(other.checked.items())
            ) + tuple(other.examples),
        )

    def outside_finding(self) -> Finding:
        checked = self.keyword.checked + self.other.checked
        runtime_only = self.keyword.runtime_only + self.other.runtime_only
        total_checked = checked.total()
        outside = runtime_only.total()
        outside_rate = share_of(outside, total_checked)
        return Finding.watch(
            name=_OUTSIDE_JOIN,
            measured=(
                f"{outside}/{total_checked} keys resolve outside every "
                f"(face, trait_kind) range their sidecar declares "
                f"({outside_rate:.1%}, watched, no ceiling)"
            ),
            detail=tuple(
                f"{kind}: {runtime_only[kind]}/{checked[kind]} outside "
                "every declared range"
                for kind in sorted(runtime_only)
            ),
        )
