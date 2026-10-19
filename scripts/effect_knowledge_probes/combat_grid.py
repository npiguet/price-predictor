"""Predicted deaths in a one-on-one block, power 1..8 x toughness 1..8, averaged over records.

    python scripts/effect_knowledge_probes/combat_grid.py CHECKPOINT [--source SOURCE]
        [--variant VARIANT ...] [--records 40] [--shards 200]

The combat counterpart of ``damage_grid.py``. A record qualifies when exactly one
attacker is blocked, by exactly one blocker that blocks nothing else, neither creature
has a damage-step keyword or damage from earlier in the turn, and the real outcome obeys
the rule. Each grid edits two numbers on the board and reads one creature's predicted
death:

- ``blocker``: the attacker's power against the blocker's toughness, reading the blocker;
- ``attacker``: the blocker's power against the attacker's toughness, reading the attacker.

A variant grants one creature a keyword through the until-end-of-turn channel, which
changes the rule's answer in a known way: deathtouch kills on any damage, indestructible
never dies, first strike kills before the other creature deals damage, double strike
deals damage in both steps. Each grid is scored against the rule applied to every edited
record, so the score is right whatever the other creature's real power and toughness are.

The block pairing is unambiguous only with one blocked attacker: a card slot says that a
creature is blocking, not which attacker it blocks.

A combat record's snapshot is taken at the step's first damage event, so the creature
damaged first already shows that damage as marked. A creature may therefore show marked
damage equal to the other creature's power, which is this step's own damage; any other
marked damage disqualifies the record. Every edited record clears both creatures' marked
damage, so the grid asks about the board before the damage step.

``--source validation`` (the default) reads the card-disjoint and game-disjoint strata,
``--source training`` the training shards.
"""
from __future__ import annotations

import argparse
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "effect_knowledge_probes.run", Path(__file__).resolve().parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
common = run.sibling("common")
sweeps = run.sibling("sweeps")

CORPUS = Path("output/effects/corpus-gen2")
VALUES = range(1, 9)
#: Keywords that change who dies in a damage step; a qualifying pair carries none.
COMBAT_KEYWORDS = frozenset({
    "first_strike", "double_strike", "deathtouch", "indestructible", "wither", "infect",
    "trample", "lifelink", "protection", "banding", "shadow",
})
#: ``variant -> (creature granted the keyword, keyword)``; ``none`` grants nothing.
VARIANTS: dict[str, tuple[str, str] | None] = {
    "none": None,
    "attacker-deathtouch": ("attacker", "deathtouch"),
    "blocker-indestructible": ("blocker", "indestructible"),
    "attacker-first-strike": ("attacker", "first_strike"),
    "blocker-first-strike": ("blocker", "first_strike"),
    "attacker-double-strike": ("attacker", "double_strike"),
}


# ── the rule ────────────────────────────────────────────────────────────


def combat_deaths(
    attacker: tuple[int, int, int], blocker: tuple[int, int, int],
    attacker_keywords=frozenset(), blocker_keywords=frozenset(),
) -> tuple[bool, bool]:
    """``(attacker dies, blocker dies)`` when one blocker blocks one attacker.

    Each creature is ``(power, toughness, damage already marked)``. Rules 510 and
    702.7, 702.4, 702.2 and 702.12: a first or double striker deals damage in the
    first-strike step, a creature without first strike or with double strike in the
    regular step, and a creature destroyed after the first step deals no damage in
    the second. Lethal damage is damage at least the toughness, or any damage from a
    deathtouch source; indestructible survives both.
    """
    power = {"a": attacker[0], "b": blocker[0]}
    toughness = {"a": attacker[1], "b": blocker[1]}
    marked = {"a": attacker[2], "b": blocker[2]}
    keywords = {"a": set(attacker_keywords), "b": set(blocker_keywords)}
    dealt_lethal = {"a": False, "b": False}
    dead = {"a": False, "b": False}
    other = {"a": "b", "b": "a"}

    def strikes(side: str, first_step: bool) -> bool:
        k = keywords[side]
        if first_step:
            return bool(k & {"first_strike", "double_strike"})
        return "first_strike" not in k or "double_strike" in k

    for first_step in (True, False):
        hits = [s for s in ("a", "b") if not dead[s] and strikes(s, first_step)
                and power[s] > 0]
        for side in hits:
            target = other[side]
            marked[target] += power[side]
            if "deathtouch" in keywords[side]:
                dealt_lethal[target] = True
        for side in ("a", "b"):
            if dead[side] or "indestructible" in keywords[side]:
                continue
            if marked[side] >= toughness[side] or dealt_lethal[side]:
                dead[side] = True
    return dead["a"], dead["b"]


# ── the edits ───────────────────────────────────────────────────────────


def _edit_entity(record, entity_id: str, edit):
    entities = tuple(edit(e) if e.id == entity_id else e for e in record.state.entities)
    return replace(record, state=replace(record.state, entities=entities))


def set_power(record, entity_id: str, power: int):
    """``record`` with one entity's total power set; base power absorbs the change."""
    from effects.domain.state_snapshot import PowerToughness

    def edit(entity):
        pt = entity.pt or PowerToughness(base=(0, 0))
        base_p = power - pt.boosts[0] - pt.counters[0]
        return replace(entity, pt=replace(pt, base=(base_p, pt.base[1])))

    return _edit_entity(record, entity_id, edit)


def grant_keyword(record, entity_id: str, keyword: str):
    """``record`` with one entity granted ``keyword`` until end of turn."""
    def edit(entity):
        granted = entity.granted_temporary
        if keyword in granted.keywords:
            return entity
        return replace(entity, granted_temporary=replace(
            granted, keywords=(*granted.keywords, keyword)))

    return _edit_entity(record, entity_id, edit)


def clear_damage(record, *entity_ids: str):
    """``record`` with the named entities' marked damage cleared."""
    out = record
    for entity_id in entity_ids:
        out = _edit_entity(out, entity_id, lambda entity: replace(entity, damage=0))
    return out


def side_of(record, entity_id: str):
    """``(power, toughness, marked damage)`` of one entity, and its granted keywords."""
    entity = record.state.entity(entity_id)
    power, toughness = entity.pt.total
    return (power, toughness, entity.damage), frozenset(entity.granted_temporary.keywords)


# ── which records qualify ───────────────────────────────────────────────


def single_block(record, keywords_of) -> tuple[str, str] | None:
    """``(attacker, blocker)`` when the record is a qualifying one-on-one block.

    ``keywords_of(entity)`` is the entity's keywords from both channels
    (``KeywordResolver.keywords_of``), so a printed keyword disqualifies as well as a
    granted one.
    """
    from effects.domain.effect_model import ZONE_OUTCOMES
    from effects.domain.effect_targets import derive_targets
    from effects.domain.records import CombatPayload, RecordKind

    if record.kind is not RecordKind.COMBAT or record.fork or record.interventional:
        return None
    payload = record.payload
    if not isinstance(payload, CombatPayload) or getattr(payload, "probed_keyword", None):
        return None
    blocked = [(attacker, blockers) for attacker, blockers in payload.blocks.items()
               if blockers]
    if len(blocked) != 1 or len(blocked[0][1]) != 1:
        return None
    attacker_id, (blocker_id,) = blocked[0]
    attacker = record.state.entity(attacker_id)
    blocker = record.state.entity(blocker_id)
    for entity in (attacker, blocker):
        if (entity is None or entity.pt is None or entity.zone != "battlefield"
                or "creature" not in entity.types):
            return None
        if keywords_of(entity) & COMBAT_KEYWORDS:
            return None
    if blocker.combat is None or tuple(blocker.combat.blocking) != (attacker_id,):
        return None
    if attacker.pt.total[0] <= 0 or blocker.pt.total[0] < 0:
        return None
    # Marked damage is either none or this step's own, which the snapshot taken at
    # the first damage event already shows; damage from earlier in the turn would
    # make the cleared board in the grid a different board.
    if attacker.damage not in (0, blocker.pt.total[0]):
        return None
    if blocker.damage not in (0, attacker.pt.total[0]):
        return None
    # The real outcome must obey the rule on the board before the step, so
    # prevention, regeneration or a replacement the board does not show cannot
    # be what the grid reads.
    targets = derive_targets(record)
    died = ZONE_OUTCOMES.index("died")
    before = clear_damage(record, attacker_id, blocker_id)
    expected = combat_deaths(side_of(before, attacker_id)[0], side_of(before, blocker_id)[0])
    for entity_id, should_die in zip((attacker_id, blocker_id), expected):
        entry = targets.get(entity_id)
        observed = entry is not None and entry.fields.get("zone_outcome") == died
        if observed != should_die:
            return None
    return attacker_id, blocker_id


def qualifying_records(directories, count: int, shards: int, keywords_of):
    """The first ``count`` qualifying ``(record, attacker, blocker)`` triples, one per game.

    One per game because a game repeats its creatures from combat to combat, and forty
    blocks by one large attacker would read one board forty times.
    """
    from effects.infrastructure.record_io import iter_shards, read_shard

    chosen = []
    games: set[str] = set()
    for directory in directories:
        for shard in iter_shards(directory)[:shards]:
            for record in read_shard(shard):
                if record.game_id in games:
                    continue
                pair = single_block(record, keywords_of)
                if pair is not None:
                    games.add(record.game_id)
                    chosen.append((record, *pair))
                    if len(chosen) >= count:
                        return chosen
    return chosen


# ── the grids ───────────────────────────────────────────────────────────


def grid_records(record, attacker: str, blocker: str, which: str, variant: str):
    """The 8 x 8 edited records of one grid, row by row, and the rule's answer for each."""
    record = clear_damage(record, attacker, blocker)
    granted = VARIANTS[variant]
    if granted is not None:
        record = grant_keyword(record, attacker if granted[0] == "attacker" else blocker,
                               granted[1])
    striker, struck = (attacker, blocker) if which == "blocker" else (blocker, attacker)
    rows, rules = [], []
    for power in VALUES:
        row, rule = [], []
        for toughness in VALUES:
            edited = sweeps.set_toughness(set_power(record, striker, power), struck, toughness)
            row.append(edited)
            a, a_kw = side_of(edited, attacker)
            b, b_kw = side_of(edited, blocker)
            attacker_dies, blocker_dies = combat_deaths(a, b, a_kw, b_kw)
            rule.append(blocker_dies if which == "blocker" else attacker_dies)
        rows.append(row)
        rules.append(rule)
    return rows, np.array(rules, dtype=float)


def score(grid: np.ndarray, rule: np.ndarray) -> str:
    """Mean absolute error against the rule, and the cells either side of the step."""
    line = f"error against the rule {np.abs(grid - rule).mean():.3f}"
    if np.array_equal(rule, np.array([[1.0 if p >= t else 0.0 for t in VALUES]
                                      for p in VALUES])):
        by1 = np.mean([grid[p - 1, p - 2] for p in range(2, 9)])
        exact = np.mean([grid[p - 1, p - 1] for p in range(1, 9)])
        short = np.mean([grid[p - 1, p] for p in range(1, 8)])
        line += (f" | lethal by 1 {by1:.2f} | exactly lethal {exact:.2f}"
                 f" | short by 1 {short:.2f} | gap {by1 - short:.2f}")
    return line


def main() -> None:
    from effects.domain.damage_step_keywords import KeywordResolver

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source", choices=("validation", "training"), default="validation")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), nargs="+", default=["all"])
    parser.add_argument("--records", type=int, default=40)
    parser.add_argument("--shards", type=int, default=200)
    args = parser.parse_args()
    variants = list(VARIANTS) if "all" in args.variant else args.variant

    sidecars = common.sidecar_cache(common.DEFAULT_CARDS_FOLDERS)
    resolver = KeywordResolver(sidecars)
    directories = ([CORPUS / "training"] if args.source == "training" else
                   [CORPUS / "validation" / "card-disjoint",
                    CORPUS / "validation" / "game-disjoint"])
    triples = qualifying_records(directories, args.records, args.shards, resolver.keywords_of)
    checkpoint = common.load_checkpoint(args.checkpoint)
    probe_model = common.load_probe_model(
        args.checkpoint, list(common.DEFAULT_CARDS_FOLDERS), [r for r, _, _ in triples],
        checkpoint=checkpoint)
    print(f"{args.checkpoint.parent.name} ({checkpoint.epoch}), "
          f"{len(triples)} {args.source} one-on-one blocks", flush=True)
    for variant in variants:
        for which in ("blocker", "attacker"):
            grid = np.zeros((8, 8))
            rule = np.zeros((8, 8))
            for record, attacker, blocker in triples:
                rows, rules = grid_records(record, attacker, blocker, which, variant)
                read_id = blocker if which == "blocker" else attacker
                for i, row in enumerate(rows):
                    grid[i] += sweeps._predict(
                        probe_model, row,
                        lambda o, v, s, r, e=read_id: sweeps._death(o, s, r, e))
                rule += rules
            grid /= max(len(triples), 1)
            rule /= max(len(triples), 1)
            striker, struck = (("attacker", "blocker") if which == "blocker"
                               else ("blocker", "attacker"))
            print(f"\n{variant}, {struck}'s death: {score(grid, rule)}")
            print(f"| {striker} power \\ {struck} toughness | "
                  + " | ".join(str(t) for t in VALUES) + " |")
            print("|---|" + "---:|" * 8)
            for i, power in enumerate(VALUES):
                print(f"| {power} | " + " | ".join(f"{grid[i, j]:.2f}" for j in range(8))
                      + " |", flush=True)


if __name__ == "__main__":
    main()
