"""Predicted deaths in a one-on-one block, power 1..8 x toughness 1..8, averaged over records.

    python scripts/effect_knowledge_probes/combat_grid.py CHECKPOINT [--source SOURCE]
        [--variant VARIANT ...] [--records 40] [--shards 200]

The combat counterpart of ``damage_grid.py``. A record qualifies when exactly one
attacker is blocked, by exactly one blocker that blocks nothing else, the record covers
the regular damage step, neither creature has a damage-step keyword or damage from
earlier in the turn, and the real outcome obeys the rule. Each grid edits two numbers on
the board and reads one creature's predicted death:

- ``blocker``: the attacker's power against the blocker's toughness, reading the blocker;
- ``attacker``: the blocker's power against the attacker's toughness, reading the attacker.

A variant grants one creature a keyword through the until-end-of-turn channel, which
changes the rule's answer in a known way: deathtouch kills on any damage, indestructible
never dies. A record covers one damage step, and the model predicts that step alone, so
a first or double striker is read in the first-strike step, which the variant sets: there
it deals its damage and the creature it fights deals none. A double striker's second hit
is read in the regular step, with the blocker already marked with the first. Each grid
is scored against the rule for that step applied to every edited record, so the score is
right whatever the other creature's real power and toughness are.

The block pairing is unambiguous only with one blocked attacker: a card slot says that a
creature is blocking, not which attacker it blocks.

A combat record's snapshot is taken at the step's first damage event, so the creature
damaged first already shows that damage as marked. A creature may therefore show marked
damage equal to the other creature's power, which is this step's own damage; any other
marked damage disqualifies the record. Every edited record clears both creatures' marked
damage, so the grid asks about the board before the damage step, the double striker's
first hit apart. It also sets both creatures' damage assignment to what the edited board
assigns, because the input carries each creature's assignment beside its power.

``--source validation`` (the default) reads the card-disjoint and game-disjoint strata,
``--source training`` the training shards.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_knowledge_probes import common, record_walk, sweeps  # noqa: E402
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402

VALUES = range(1, 9)
#: Keywords that change who dies in a damage step; a qualifying pair carries none.
COMBAT_KEYWORDS = frozenset({
    "first_strike", "double_strike", "deathtouch", "indestructible", "wither", "infect",
    "trample", "lifelink", "protection", "banding", "shadow",
})
#: ``variant -> (creature granted the keyword or None, keyword, damage step, blocker
#: carries the attacker's first-step damage)``. A record covers one damage step, so a
#: first or double striker is read in the first-strike step, where it deals its damage
#: and the creature it fights deals none, and a double striker's second hit in the
#: regular step, on a blocker already marked with the first.
VARIANTS: dict[str, tuple[str | None, str | None, str, bool]] = {
    "none": (None, None, "regular", False),
    "attacker-deathtouch": ("attacker", "deathtouch", "regular", False),
    "blocker-indestructible": ("blocker", "indestructible", "regular", False),
    "attacker-first-strike": ("attacker", "first_strike", "first_strike", False),
    "blocker-first-strike": ("blocker", "first_strike", "first_strike", False),
    "attacker-double-strike": ("attacker", "double_strike", "first_strike", False),
    "attacker-double-strike-second-hit": ("attacker", "double_strike", "regular", True),
}


# ── the rule ────────────────────────────────────────────────────────────


def strikes_in_step(keywords, step: str) -> bool:
    """Whether a creature with ``keywords`` deals its combat damage in ``step``.

    Rules 510, 702.7 and 702.4: a first or double striker in the first-strike step,
    a creature without first strike or with double strike in the regular step.
    """
    if step == "first_strike":
        return bool(keywords & {"first_strike", "double_strike"})
    return "first_strike" not in keywords or "double_strike" in keywords


def step_deaths(
    attacker: tuple[int, int, int], blocker: tuple[int, int, int],
    attacker_keywords=frozenset(), blocker_keywords=frozenset(), step: str = "regular",
) -> tuple[bool, bool]:
    """``(attacker dies, blocker dies)`` in one damage step of a one-on-one block.

    Each creature is ``(power, toughness, damage already marked)``. Rules 510 and
    702.7, 702.4, 702.2 and 702.12: a first or double striker deals damage in the
    first-strike step, a creature without first strike or with double strike in the
    regular step, both creatures striking in a step at once. Lethal damage is marked
    damage at least the toughness, or any damage from a deathtouch source;
    indestructible survives both.
    """
    power = {"a": attacker[0], "b": blocker[0]}
    toughness = {"a": attacker[1], "b": blocker[1]}
    marked = {"a": attacker[2], "b": blocker[2]}
    keywords = {"a": set(attacker_keywords), "b": set(blocker_keywords)}
    deathtouched = {"a": False, "b": False}
    other = {"a": "b", "b": "a"}

    for side in [s for s in ("a", "b") if strikes_in_step(keywords[s], step) and power[s] > 0]:
        marked[other[side]] += power[side]
        if "deathtouch" in keywords[side]:
            deathtouched[other[side]] = True
    dead = {side: "indestructible" not in keywords[side]
            and (marked[side] >= toughness[side] or deathtouched[side])
            for side in ("a", "b")}
    return dead["a"], dead["b"]


# ── the edits ───────────────────────────────────────────────────────────


def assign_step_damage(record, attacker: str, blocker: str):
    """``record`` with both combatants' damage assignment matching their edited board.

    A combat record's input carries each creature's assignment, the targets it assigns
    to and the damage it assigns, so an edit to a creature's power or to the step has to
    move its assignment with it, or the board and the assignment disagree. In a
    one-on-one block a creature that strikes in this step assigns all its power to the
    other; one that does not strike, or has no power, assigns nothing.
    """
    step = record.state.global_.combat_substep
    choices = {source: dict(targets)
               for source, targets in record.payload.assignment_choices.items()
               if source not in (attacker, blocker)}
    for source, target in ((attacker, blocker), (blocker, attacker)):
        (power, _, _), keywords = side_of(record, source)
        if power > 0 and strikes_in_step(keywords, step):
            choices[source] = {target: power}
    return replace(record, payload=replace(record.payload, assignment_choices=choices))


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
    from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome
    from effects.domain.effect_targets import derive_targets
    from effects.domain.records import CombatPayload, RecordKind
    from effects.domain.state_snapshot import CombatSubstep

    if record.kind is not RecordKind.COMBAT or record.fork or record.interventional:
        return None
    if record.state.global_.combat_substep != CombatSubstep.REGULAR:
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
        if (entity is None or entity.pt is None or not entity.on_battlefield
                or not entity.is_creature):
            return None
        if keywords_of(entity) & COMBAT_KEYWORDS:
            return None
    if blocker.combat is None or tuple(blocker.combat.blocking) != (attacker_id,):
        return None
    if attacker.power <= 0 or blocker.power < 0:
        return None
    # Marked damage is either none or this step's own, which the snapshot taken at
    # the first damage event already shows; damage from earlier in the turn would
    # make the cleared board in the grid a different board.
    if attacker.damage not in (0, blocker.power):
        return None
    if blocker.damage not in (0, attacker.power):
        return None
    # The real outcome must obey the rule on the board before the step, so
    # prevention, regeneration or a replacement the board does not show cannot
    # be what the grid reads.
    targets = derive_targets(record)
    died = ZONE_OUTCOMES.index(ZoneOutcome.DIED)
    before = BoardEdit(record).clear_damage(attacker_id, blocker_id).record
    expected = step_deaths(side_of(before, attacker_id)[0], side_of(before, blocker_id)[0])
    for entity_id, should_die in zip((attacker_id, blocker_id), expected):
        entry = targets.get(entity_id)
        observed = entry is not None and entry.fields.get(HeadField.ZONE_OUTCOME) == died
        if observed != should_die:
            return None
    return attacker_id, blocker_id


def qualifying_records(directories, count: int, shards: int, keywords_of):
    """The first ``count`` qualifying ``(record, attacker, blocker)`` triples, one per game.

    One per game because a game repeats its creatures from combat to combat, and forty
    blocks by one large attacker would read one board forty times.
    """
    def choose(record):
        pair = single_block(record, keywords_of)
        return None if pair is None else (record, *pair)

    return record_walk.qualifying(directories, choose, count, per_game=1, shards=shards)


# ── the grids ───────────────────────────────────────────────────────────


def grid_records(record, attacker: str, blocker: str, which: str, variant: str):
    """The 8 x 8 edited records of one grid, row by row, and the rule's answer for each."""
    holder, keyword, step, carried = VARIANTS[variant]
    base = BoardEdit(record).clear_damage(attacker, blocker).set_step(step)
    if keyword is not None:
        base = base.grant_keyword(attacker if holder == "attacker" else blocker, keyword)
    striker, struck = (attacker, blocker) if which == "blocker" else (blocker, attacker)
    rows, rules = [], []
    for power in VALUES:
        row, rule = [], []
        for toughness in VALUES:
            cell = base.set_power(striker, power).set_toughness(struck, toughness)
            if carried:
                cell = cell.set_marked_damage(blocker, side_of(cell.record, attacker)[0][0])
            edited = assign_step_damage(cell.record, attacker, blocker)
            row.append(edited)
            a, a_kw = side_of(edited, attacker)
            b, b_kw = side_of(edited, blocker)
            attacker_dies, blocker_dies = step_deaths(a, b, a_kw, b_kw, step)
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

    context = common.probe_context(args.checkpoint)
    resolver = KeywordResolver(context.sidecars())
    directories = ([context.training_dir] if args.source == "training"
                   else context.validation_dirs())
    triples = qualifying_records(directories, args.records, args.shards, resolver.keywords_of)
    probe_model = context.model([r for r, _, _ in triples])
    print(f"{args.checkpoint.parent.name} ({context.checkpoint.epoch}), "
          f"{len(triples)} {args.source} one-on-one blocks", flush=True)
    for variant in variants:
        for which in ("blocker", "attacker"):
            grid = np.zeros((8, 8))
            rule = np.zeros((8, 8))
            for record, attacker, blocker in triples:
                rows, rules = grid_records(record, attacker, blocker, which, variant)
                read_id = blocker if which == "blocker" else attacker
                for i, row in enumerate(rows):
                    grid[i] += common.predict(
                        probe_model, row,
                        lambda o, v, s, r, e=read_id: sweeps.death(o, s, r, e))
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
