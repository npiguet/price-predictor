"""Invariants a shard directory must hold, measured rather than asserted.

Meant for the first few minutes of a collection pass: a collector that writes a
literal, reuses an id or varies its snapshot depth produces records that parse
and pass every contract test, so each invariant prints the number it measured,
pass or fail.

**Measuring and judging are separate.** A :class:`Finding` with no threshold is
watched: it prints ``[WATCH]`` and never fails a run, and passing a floor makes
it a verdict without changing what is measured. A number stays watched until
its healthy value is known, because an invariant that fails on every good run
teaches an operator to skip the whole report.

One streaming pass, holding every ``record_id``, one digest per record and every
distinct provenance key, which is why ``--limit`` exists.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from effects.application.corpus_validation.envelope_checks import (
    ActingLines,
    Duplicates,
    GameBoundaries,
    Links,
    MirrorTurns,
    Modes,
    ProbeForks,
    RecordIds,
    Tiers,
)
from effects.application.corpus_validation.event_checks import (
    Attribution,
    CauseBearing,
    DuplicateEvents,
    EventTypes,
    ZoneChanges,
    fork_events,
    resolution_events,
)
from effects.application.corpus_validation.findings import (
    Check,
    Findings,
    Thresholds,
)
from effects.application.corpus_validation.key_joins import KeyJoins
from effects.application.corpus_validation.payload_checks import (
    CostChannels,
    Outcomes,
    Rewrites,
    Triggers,
)
from effects.application.corpus_validation.presence import FieldPresence
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord


@dataclass(frozen=True, slots=True)
class DirectoryReport:
    """What :func:`validate_directory` read and found."""

    shards: int
    findings: Findings


def validate_directory(
    root: Path, *,
    limit: int = 0,
    thresholds: Thresholds = Thresholds(),
    sidecars: SidecarLookup | None = None,
) -> DirectoryReport:
    """Every invariant over the first ``limit`` records under ``root`` (0
    reading all), the shards listed once; raises ``NoShardsError`` on a
    directory holding none."""
    from effects.infrastructure.shard_io import read_window, require_shards

    shards = require_shards(root)
    presence = FieldPresence()
    findings = validate_corpus(
        read_window(root, limit, presence.observe, shards=shards),
        thresholds, sidecars, presence,
    )
    return DirectoryReport(len(shards), findings)


def validate_corpus(
    records: Iterable[EffectRecord],
    thresholds: Thresholds = Thresholds(),
    sidecars: SidecarLookup | None = None,
    presence: FieldPresence | None = None,
) -> Findings:
    """Every invariant, measured over one pass of ``records``.

    ``sidecars`` of ``None`` reports the key-join checks as unchecked rather
    than as holding. ``presence`` is the per-shard envelope tally
    ``shard_io.read_window`` fills from the raw lines (FR-033), since a parsed
    record cannot say whether a field was on the wire.
    """
    checks = _checks(sidecars)
    for record in records:
        for check in checks:
            check.add(record)
    findings = Findings(
        finding for check in checks for finding in check.findings(thresholds)
    )
    if presence is not None:
        findings.append(presence.finding())
    return findings


def _checks(sidecars: SidecarLookup | None) -> tuple[Check, ...]:
    """Every invariant, in the order the report prints them."""
    record_ids = RecordIds()
    return (
        record_ids,
        GameBoundaries(),
        Links(),
        ActingLines(),
        KeyJoins(sidecars),
        Modes(),
        Outcomes(),
        CostChannels(),
        Tiers(),
        Duplicates(),
        DuplicateEvents(),
        Triggers(),
        Rewrites(),
        ZoneChanges(),
        Attribution(
            name="resolution events name what produced them",
            noun="resolution events",
            events_of=resolution_events,
            floor_of=lambda limits: limits.min_attributed_rate,
        ),
        Attribution(
            name="fork records' events name what produced them",
            noun="events on fork records",
            events_of=fork_events,
            floor_of=lambda limits: limits.min_fork_attributed_rate,
        ),
        CauseBearing(),
        MirrorTurns(record_ids),
        ProbeForks(),
        EventTypes(),
    )
