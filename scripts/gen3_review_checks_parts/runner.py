"""Run every check over a records directory in one pass."""

from __future__ import annotations

import sys
from pathlib import Path

from effects.infrastructure.shard_io import iter_record_dicts, iter_shards

from .convoke import ConvokeCandidates
from .counter_checks import CounterHalves
from .event_checks import (
    DirectedForks,
    ManaAttribution,
    PhaseSkipped,
    TriggerSubjects,
)
from .forced_checks import ForcedHalves, ForcedUnattached
from .game_checks import LethalCombat, MarkedGames, RescueRecords
from .half_checks import EquipHalves, FizzledHalves
from .key_checks import AmbiguousKeys
from .model import Check, Line
from .protection import ProtectionForks
from .sidecars import Sidecars, keys_declared_by_two_lines
from .snapshot_checks import CopyLines, LibraryCards
from .tree_checks import tree_lines


def build_checks(sidecars: Sidecars) -> list[Check]:
    return [
        AmbiguousKeys(sidecars),
        EquipHalves(),
        CounterHalves(sidecars),
        FizzledHalves(),
        ForcedHalves(),
        ForcedUnattached(),
        ProtectionForks(),
        TriggerSubjects(),
        PhaseSkipped(),
        MarkedGames(),
        RescueRecords(),
        LethalCombat(),
        DirectedForks(),
        LibraryCards(),
        CopyLines(),
        ManaAttribution(),
        ConvokeCandidates(sidecars),
    ]


def feed_shard(shard: Path, checks: list[Check]) -> None:
    """Feed one shard's records to every check; a damaged shard is read up to the damage."""
    try:
        for record in iter_record_dicts(shard):
            for check in checks:
                check.feed(record)
    except ValueError as error:
        print(f"  stopped at damage in {shard.name}: {error}", file=sys.stderr, flush=True)


def run_tree_checks(converted: Path) -> list[Line]:
    """The checks that need only the converted tree at ``converted``."""
    return [keys_declared_by_two_lines(converted), *tree_lines(converted)]


def run_checks(shards: list[Path], converted: Path) -> list[Line]:
    """Every check's lines over ``shards``, the converted tree at ``converted``."""
    checks = build_checks(Sidecars(converted))
    for number, shard in enumerate(shards, 1):
        print(f"[{number}/{len(shards)}] {shard.name}", file=sys.stderr, flush=True)
        feed_shard(shard, checks)
    lines = run_tree_checks(converted)
    for check in checks:
        lines.extend(check.lines())
    return lines


def shards_of(records_dir: Path) -> list[Path]:
    return iter_shards(records_dir)
