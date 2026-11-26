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

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from effects.application.corpus_validation.envelope_checks import (
    ActingLines,
    Duplicates,
    GameBoundaries,
    GameEndMarkers,
    KindPresence,
    Links,
    LossyForks,
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
    EmptyMana,
    EventTypes,
    ZoneChanges,
    fork_events,
    resolution_events,
)
from effects.application.corpus_validation.findings import (
    EXAMPLES,
    Check,
    Finding,
    Findings,
    Thresholds,
)
from effects.application.corpus_validation.key_joins import KeyJoins
from effects.application.corpus_validation.payload_checks import (
    Activations,
    Rewrites,
    Triggers,
)
from effects.application.corpus_validation.ports import RecordRenderer, ShardSource
from effects.application.corpus_validation.presence import FieldPresence
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DirectoryReport:
    """What :func:`validate_directory` read and found."""

    shards: int
    findings: Findings


def validate_directory(
    root: Path, *,
    source: ShardSource,
    record_renderer: RecordRenderer,
    limit: int = 0,
    thresholds: Thresholds = Thresholds(),
    sidecars: SidecarLookup | None = None,
) -> DirectoryReport:
    """Every invariant over the first ``limit`` records under ``root`` (0
    reading all), the shards listed once; raises ``NoShardsError`` on a
    directory holding none.

    A shard whose read a damaged member or an undecodable line ends is a
    failed finding naming it and the line, its records before the damage
    measured; the other shards are read on (V5 #20). What the read met
    without stopping, values outside the wire's closed lists and damaged last
    members (``shard_io.DecodeTally``), is reported as watched figures.
    ``source`` and ``record_renderer`` are the ports the command passes
    (:mod:`effects.application.corpus_validation.ports`).
    """
    shards = source.require_shards(root)
    source.require_generation(shard.path(root) for shard in shards)
    presence = FieldPresence()
    damage = ShardDamage(source)
    findings = validate_corpus(
        damage.read(root, shards, limit, presence.observe), thresholds, sidecars, presence,
        playability_rate=playability_rate_of(root), window_cut=bool(limit),
        record_renderer=record_renderer,
    )
    findings.extend(damage.findings())
    return DirectoryReport(len(shards), findings)


class ShardDamage:
    """The shards a damaged member or an undecodable line cut short, and the
    read's decode tally, as findings."""

    def __init__(self, source: ShardSource) -> None:
        self.source = source
        self.read_shards = 0
        self.cut: list[str] = []

    def read(self, root: Path, shards, limit: int, observe) -> Iterator[EffectRecord]:
        """Every record of ``shards`` up to ``limit``, a shard's damage noted
        and its read ended, the next shard read on."""
        seen = 0
        for shard in shards:
            self.read_shards += 1
            try:
                for record in self.source.read_window(
                        root, limit - seen if limit else 0, observe, shards=(shard,)):
                    yield record
                    seen += 1
            except self.source.damage as error:
                logger.error("%s", error)
                self.cut.append(str(error))
            if limit and seen >= limit:
                return

    def findings(self) -> list[Finding]:
        found = [Finding(
            name="every shard line decodes",
            ok=not self.cut,
            measured=(f"{self.read_shards - len(self.cut)}/{self.read_shards} shards read "
                      f"whole; {len(self.cut)} ended at a line or member that does not "
                      "decode, their records before it measured"),
            detail=tuple(self.cut[:EXAMPLES]),
        )]
        tally = self.source.decode_tally
        found.append(Finding.watch(
            f"values outside the wire's closed lists: {sum(tally.off_list.values())}",
            "read as written and counted (shard_io.DecodeTally)",
            tuple(f"{member} {value}: {count}"
                  for (member, value), count in sorted(tally.off_list.items())),
        ))
        found.append(Finding.watch(
            f"shards whose damaged last gzip member ended the read: "
            f"{len(tally.damaged_tails)}",
            "their records before it read",
            tuple(tally.damaged_tails[:EXAMPLES]),
        ))
        return found


def playability_rate_of(root: Path) -> float | None:
    """The ``--playability-rate`` the run under ``root`` collected at, from
    its ``run.json`` (each collector's invocation records its resolved
    ``caps``): the smallest any invocation recorded; None without a
    manifest or a recorded rate."""
    import json

    try:
        document = json.loads((Path(root) / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rates = [entry["caps"]["playability_rate"]
             for entry in document.get("invocations") or ()
             if isinstance(entry, dict) and isinstance(entry.get("caps"), dict)
             and isinstance(entry["caps"].get("playability_rate"), (int, float))]
    return min(rates) if rates else None


def validate_corpus(
    records: Iterable[EffectRecord],
    thresholds: Thresholds = Thresholds(),
    sidecars: SidecarLookup | None = None,
    presence: FieldPresence | None = None,
    *,
    record_renderer: RecordRenderer,
    playability_rate: float | None = None,
    window_cut: bool = False,
) -> Findings:
    """Every invariant, measured over one pass of ``records``.

    ``sidecars`` of ``None`` reports the key-join checks as unchecked rather
    than as holding. ``presence`` is the per-shard member tally
    ``shard_io.read_window`` fills from the raw lines (FR-033, root § 8.13),
    since a parsed record cannot say whether a field was on the wire.
    ``window_cut`` says ``--limit`` ended the read, so a record the window
    does not hold may exist. ``record_renderer`` renders a record as the
    writer does, for the exact-duplicate check.
    """
    checks = _checks(sidecars, record_renderer, playability_rate, window_cut)
    for record in records:
        for check in checks:
            check.add(record)
    findings = Findings(
        finding for check in checks for finding in check.findings(thresholds)
    )
    if presence is not None:
        findings.extend(presence.findings())
    return findings


def _checks(sidecars: SidecarLookup | None, record_renderer: RecordRenderer,
            playability_rate: float | None = None, window_cut: bool = False) -> tuple[Check, ...]:
    """Every invariant, in the order the report prints them."""
    record_ids = RecordIds()
    return (
        record_ids,
        GameBoundaries(),
        GameEndMarkers(),
        KindPresence(playability_rate),
        Links(),
        ActingLines(),
        KeyJoins(sidecars),
        Modes(),
        Activations(),
        Tiers(),
        Duplicates(record_renderer),
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
        MirrorTurns(record_ids, window_cut),
        ProbeForks(),
        LossyForks(),
        EmptyMana(),
        EventTypes(),
    )
