"""Predicted deaths in a one-on-one block, power 1..8 x toughness 1..8, averaged over records.

    python scripts/effect_knowledge_probes/combat_grid.py CHECKPOINT [--source SOURCE]
        [--blocks one|several] [--variant VARIANT ...] [--half HALF] [--records 40]
        [--shards 200]

The combat counterpart of ``damage_grid.py``. A record qualifies when exactly one
attacker is blocked, by exactly one blocker that blocks nothing else, the record covers
the regular damage step, neither creature has a damage-step keyword or damage from
earlier in the turn, and the real outcome obeys the rule. Each grid edits two numbers on
the board and reads one creature's predicted death:

- ``blocker``: the attacker's power against the blocker's toughness, reading the blocker;
- ``attacker``: the blocker's power against the attacker's toughness, reading the attacker.

A variant grants one creature a keyword as an effect until end of turn grants it, a
granted keyword line plus the keyword's kept flag (``BoardEdit.grant_keyword``), which
changes the rule's answer in a known way: deathtouch kills on any damage, indestructible
never dies. A record covers one damage step, and the model predicts that step alone, so
a first or double striker is read in the first-strike step, which the variant sets: there
it deals its damage and the creature it fights deals none. A double striker's second hit
is read in the regular step, with the blocker already marked with the first. Each grid
is scored against the rule for that step applied to every edited record, so the score is
right whatever the other creature's real power and toughness are.

``--blocks several`` reads combats with several blocked attackers instead (FR-074): every
blocked attacker blocked by one creature blocking it alone, every pair qualifying as the
one-on-one block does, and the grid edits one pair, the attacker first in id order, with
the other pairs left as they fought. A gen-3 board states which attacker each creature
blocks, as a ``blocks`` relation between their slots, so the model can tell the pairs
apart; the score asks whether it reads the pair the grid edits apart from the rest. Both
kinds are scored through the same rule, ``rule_outcomes.step_deaths``, on the edited pair.

A combat record's snapshot is taken at the step's first damage event, so the creature
damaged first already shows that damage as marked. A creature may therefore show marked
damage equal to the other creature's power, which is this step's own damage; any other
marked damage disqualifies the record. Every edited record clears both creatures' marked
damage, so the grid asks about the board before the damage step, the double striker's
first hit apart. It also sets both creatures' damage assignment to what the edited board
assigns, because the input carries each creature's assignment beside its power.

``--source validation`` (the default) reads the blocks the corpus's selection set froze
for the combat grids in ``--half`` (``reporting`` by default, ``selection`` or ``both``) of
both validation strata, one a game (the several-blocker grids are frozen as
``combat-several/...``); ``--source training`` scans the first ``--shards`` training shards
for ``--records`` blocks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import print_table  # noqa: E402

from effect_knowledge_probes import common, readout, record_walk  # noqa: E402
from effects.domain.grid_edits import (  # noqa: E402, F401  (the suite's names for them)
    BLOCKS,
    COMBAT_KEYWORDS,
    GRID_PREFIX,
    VALUES,
    VARIANTS,
    assign_step_damage,
    grid_block,
    grid_records,
    side_of,
)
from effects.domain.rule_outcomes import step_deaths, strikes_in_step  # noqa: E402, F401


def single_block(record, keywords_of) -> tuple[str, str] | None:
    """``(attacker, blocker)`` when the record is a qualifying one-on-one block.

    ``keywords_of(entity)`` is the entity's keywords from both channels
    (``KeywordResolver.keywords_of``), so a printed keyword disqualifies as well as a
    granted one.
    """
    return grid_block(record, keywords_of, "one")


def several_blocks(record, keywords_of) -> tuple[str, str] | None:
    """``(attacker, blocker)`` of the pair a several-blocker grid edits: a combat
    with two or more blocked attackers, each pair qualifying as a one-on-one
    block does; the pair is the one whose attacker comes first in id order."""
    return grid_block(record, keywords_of, "several")


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

    def options(parser) -> None:
        parser.add_argument("--source", choices=("validation", "training"),
                            default="validation")
        parser.add_argument("--blocks", choices=BLOCKS, default="one",
                            help="one blocked attacker, or several")
        parser.add_argument("--variant", choices=(*VARIANTS, "all"), nargs="+",
                            default=["all"])
        parser.add_argument("--half", choices=("reporting", "selection", "both"),
                            default="reporting")
        parser.add_argument("--probe-set-dir", type=Path, default=common.REPORTS,
                            help="where the corpus's selection set is")

    args, context = common.grid_run(__doc__, records=40, options=options)
    variants = list(VARIANTS) if "all" in args.variant else args.variant
    resolver = KeywordResolver(context.sidecars())
    if args.source == "training":
        # One block a game: a game repeats its creatures from combat to combat, and
        # forty blocks by one large attacker would read one board forty times.
        triples = record_walk.qualifying(
            [context.training_dir],
            record_walk.with_record(
                lambda r: grid_block(r, resolver.keywords_of, args.blocks)),
            args.records, shards=args.shards)
    else:
        from effect_knowledge_probes.selection import frozen_grid_records

        halves = ("selection", "reporting") if args.half == "both" else (args.half,)
        triples = [(record, *entities) for record, entities in frozen_grid_records(
            context, GRID_PREFIX[args.blocks], halves=halves,
            reports_dir=args.probe_set_dir)]
    probe_model = context.model([r for r, _, _ in triples])
    kind = "one-on-one blocks" if args.blocks == "one" else "combats of several blocks"
    common.heading(args, context, f"{len(triples)} {args.source} {kind}")
    for variant in variants:
        for which in ("blocker", "attacker"):
            grid = np.zeros((8, 8))
            rule = np.zeros((8, 8))
            for record, attacker, blocker in triples:
                rows, rules = grid_records(record, attacker, blocker, which, variant)
                dies = readout.at_card("dies", blocker if which == "blocker" else attacker)
                cells = common.predict_all(probe_model, [cell for row in rows for cell in row],
                                           dies)
                grid += np.reshape(cells, grid.shape)
                rule += rules
            grid /= max(len(triples), 1)
            rule /= max(len(triples), 1)
            striker, struck = (("attacker", "blocker") if which == "blocker"
                               else ("blocker", "attacker"))
            print(f"\n{variant}, {struck}'s death: {score(grid, rule)}")
            print_table([f"{striker} power \\ {struck} toughness", *VALUES],
                        [[power, *(f"{v:.2f}" for v in row)] for power, row in zip(VALUES, grid)],
                        "l" + "r" * 8)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
