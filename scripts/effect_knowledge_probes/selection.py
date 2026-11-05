"""The probe suite's side of the selection-set freeze: the readers
``effects.application.freeze_selection_set`` is handed (research E1).

The package cannot import the scripts, so the freeze takes its probe labels
and its grids through ports, and these are the suite's: the rung-3 probe
items ``labels.record_items`` gives, and the damage, combat and evasion grids
as edit grids, which live in the package (``effects.domain.grid_edits``)
because the control score edits them in every training run.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from effect_knowledge_probes import common, labels
from effects.domain.grid_edits import (  # noqa: F401  (the suite's names for them)
    VALUES,
    CombatGrid,
    DamageGrid,
    EvasionGrid,
    grid_sources,
)


@dataclass
class ProbeItems:
    """``ProbeItemSource`` over ``labels.record_items``: the board-dependent
    items of the ``act`` and ``entity`` targets, triggers joined to their
    causes within the shard's records."""

    sidecars: object
    acting_text: object

    def items(self, records: Sequence):
        from effects.application.freeze_selection_set import ProbeItem

        join = labels.TriggerJoin()
        for record in records:
            join.observe(record)
        joined = join.result()
        for record in records:
            for item in labels.record_items(record, self.sidecars, joined, self.acting_text):
                spec = labels.TARGETS_BY_NAME[item.target]
                if spec.scope not in ("act", "entity"):
                    continue
                yield ProbeItem(item.target, spec.kind, item.record_id, item.entity,
                                float(item.label))


#: The halves a grid script reads by default: a selected checkpoint is
#: reported on the half selection never read (FR-070).
REPORTING = ("reporting",)


def frozen_grid_records(context, prefix: str, *, halves: Sequence[str] = REPORTING,
                        reports_dir=None) -> list[tuple[object, tuple[str, ...]]]:
    """``(record, entities)`` of every frozen grid named ``prefix``…, in the
    named halves of both validation strata, each record once, read from the
    selection set frozen for the checkpoint's corpus rather than by scanning
    shards. Refuses a corpus without one."""
    from effect_knowledge_probes.halves import load_selection

    frozen = load_selection(context, reports_dir)
    out: list[tuple[object, tuple[str, ...]]] = []
    for stratum, by_half in sorted(frozen.grids.items()):
        wanted: dict[str, tuple[str, ...]] = {}
        games: set[str] = set()
        for half in halves:
            for name, grid in sorted(by_half.get(half, {}).items()):
                if name.startswith(prefix):
                    for row in grid.records:
                        wanted.setdefault(row.record_id, row.entities)
                        games.add(row.game_id)
        if not wanted:
            continue
        seen: set[str] = set()
        shards = iter_stratum_shards(context, stratum)
        for _, record in common.read_games(shards, games, set(wanted)):
            if record.record_id not in seen:
                seen.add(record.record_id)
                out.append((record, wanted[record.record_id]))
    return out


def iter_stratum_shards(context, stratum: str):
    from effects.infrastructure.shard_io import iter_shards

    return iter_shards(context.stratum_dir(stratum))
