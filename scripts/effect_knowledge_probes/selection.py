"""The probe suite's side of the selection-set freeze: the readers
``effects.application.freeze_selection_set`` is handed (research E1).

The package cannot import the scripts, so the freeze takes its probe labels
and its grids through ports, and these are the suite's: the rung-3 probe
items ``labels.record_items`` gives, and the damage, combat and evasion grids
as edit grids, each record qualifying as its grid script qualifies it and
each cell's rule outcome computed as the script computes it.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from effect_knowledge_probes import combat_grid, common, damage_grid, evasion_grid, labels

#: Damage and toughness, and power and toughness, of the damage and combat grids.
VALUES = tuple(range(1, 9))


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


def _grid_cells() -> tuple[str, ...]:
    return tuple(f"{a}x{b}" for a in VALUES for b in VALUES)


@dataclass
class _BlockMemo:
    """``combat_grid.grid_block`` of the last record asked, shared by every
    combat grid of one kind of block (``blocks``, ``combat_grid.BLOCKS``): the
    freeze asks all fourteen of one record in turn."""

    keywords_of: object
    blocks: str = "one"
    _last: tuple = (None, None)

    def __call__(self, record):
        if self._last[0] != record.record_id:
            self._last = (record.record_id,
                          combat_grid.grid_block(record, self.keywords_of, self.blocks))
        return self._last[1]


@dataclass
class CombatGrid:
    """One combat grid: a variant, and whose death it reads (``blocker``: the
    attacker's power against the blocker's toughness; ``attacker``: the
    reverse), over one-on-one blocks or over combats of several
    (``blocks.blocks``). Entities ``(attacker, blocker)`` of the pair edited."""

    variant: str
    which: str
    blocks: _BlockMemo
    cells: tuple[str, ...] = field(default_factory=_grid_cells)

    @property
    def name(self) -> str:
        return f"{combat_grid.GRID_PREFIX[self.blocks.blocks]}{self.variant}/{self.which}"

    def qualify(self, record):
        from effects.application.freeze_selection_set import GridQualification

        pair = self.blocks(record)
        if pair is None:
            return None
        attacker, blocker = pair
        _, rule = combat_grid.grid_records(record, attacker, blocker, self.which, self.variant)
        return GridQualification((attacker, blocker), tuple(float(v) for v in rule.ravel()))


@dataclass
class EvasionGrid:
    """The evasion grid: attacker keyword rows by blocker trait columns, read
    at the blocker. Entities ``(attacker, blocker)``."""

    keywords_of: object
    name: str = "evasion"
    cells: tuple[str, ...] = tuple(
        f"{evasion_grid.label(row)}|{evasion_grid.label(column)}"
        for row in evasion_grid.ATTACKER_KEYWORDS for column in evasion_grid.BLOCKER_TRAITS)

    def qualify(self, record):
        from effects.application.freeze_selection_set import GridQualification

        pair = evasion_grid.qualifying_pair(record, self.keywords_of)
        if pair is None:
            return None
        attacker, blocker = pair
        outcomes = tuple(
            float(evasion_grid.edited(record, attacker, blocker, row, column)[1])
            for row in evasion_grid.ATTACKER_KEYWORDS for column in evasion_grid.BLOCKER_TRAITS)
        return GridQualification((attacker, blocker), outcomes)


@dataclass
class DamageGrid:
    """The damage grid: a single-target damage resolution with a literal
    amount, its amount by its target's toughness, read at the target
    (``damage_grid``). Entities ``(target,)``."""

    sidecars: object
    keywords_of: object
    name: str = "damage"
    cells: tuple[str, ...] = field(default_factory=_grid_cells)

    def qualify(self, record):
        from effects.application.freeze_selection_set import GridQualification
        from effects.domain.rule_outcomes import damage_lethal

        if record.fork or record.interventional:
            return None
        found = damage_grid.damage_target(self.sidecars, self.keywords_of)(record)
        if found is None:
            return None
        _, entity_id = found
        damage = record.state.entity(entity_id).damage
        outcomes = tuple(float(damage_lethal(toughness, damage, amount))
                         for amount in VALUES for toughness in VALUES)
        return GridQualification((entity_id,), outcomes)


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


def grid_sources(sidecars, keywords_of) -> list:
    """Every grid the selection set freezes: damage, the combat grid's
    variants each read at either creature, over one-on-one blocks and over
    combats of several blocked attackers, and evasion."""
    grids: list = [DamageGrid(sidecars, keywords_of)]
    for kind in combat_grid.BLOCKS:
        blocks = _BlockMemo(keywords_of, kind)
        grids += [CombatGrid(variant, which, blocks)
                  for variant in combat_grid.VARIANTS for which in ("blocker", "attacker")]
    return [*grids, EvasionGrid(keywords_of)]
