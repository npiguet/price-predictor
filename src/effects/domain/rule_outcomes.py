"""What the rules of the game decide, as pure functions of a board.

Each function here states one rule's outcome from the facts it depends on, so
that the code labelling counterfactual copies, the probe grids that edit a
board and score the model against the rule, and the rule-test detectors all
read one statement of the rule rather than three. Rules covered:

- 704.5f-g, a creature with toughness 0 or less, or with lethal damage marked,
  is put into its owner's graveyard (:func:`lethal`);
- 510, 702.2, 702.4, 702.7 and 702.12, which creatures die in one damage step of
  a one-on-one block (:func:`step_deaths`);
- 702.9, 702.13, 702.16, 702.28, 702.31 and 702.36, whether one creature may
  block another as far as evasion and protection decide (:func:`may_block`).

Torch-free.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from enum import StrEnum

from effects.domain.state_snapshot import EntityState

# ── 704.5f-g ─────────────────────────────────────────────────────────────


class EffectKind(StrEnum):
    """What one amount does to a creature it hits."""

    DAMAGE = "damage"  # marks damage on it
    SHRINK = "shrink"  # lowers its toughness, -X/-X style


def damage_lethal(toughness: int, damage: int, amount: int) -> bool:
    """Whether ``amount`` more damage kills a creature (704.5f-g)."""
    return toughness <= 0 or damage + amount >= toughness


def shrink_lethal(toughness: int, damage: int, amount: int) -> bool:
    """Whether losing ``amount`` toughness kills a creature (704.5f-g)."""
    left = toughness - amount
    return left <= 0 or (damage > 0 and damage >= left)


_LETHAL = {EffectKind.DAMAGE: damage_lethal, EffectKind.SHRINK: shrink_lethal}


def lethal(kind: EffectKind, amount: int, entity: EntityState) -> bool:
    """Whether ``amount`` kills ``entity`` under rule 704.5f-g."""
    return _LETHAL[kind](entity.toughness, entity.damage, amount)


# ── combat damage ────────────────────────────────────────────────────────


def strikes_in_step(keywords: Collection[str], step: str) -> bool:
    """Whether a creature with ``keywords`` deals its combat damage in ``step``.

    Rules 510, 702.7 and 702.4: a first or double striker in the first-strike step,
    a creature without first strike or with double strike in the regular step.
    """
    keywords = set(keywords)
    if step == "first_strike":
        return bool(keywords & {"first_strike", "double_strike"})
    return "first_strike" not in keywords or "double_strike" in keywords


def step_deaths(
    attacker: tuple[int, int, int], blocker: tuple[int, int, int],
    attacker_keywords: Iterable[str] = frozenset(),
    blocker_keywords: Iterable[str] = frozenset(), step: str = "regular",
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


# ── blocking ─────────────────────────────────────────────────────────────

COLOUR_WORDS = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}


def may_block(attacker_keywords: Iterable[str], attacker_colours: Iterable[str],
              blocker_keywords: Iterable[str], blocker_colours: Iterable[str],
              blocker_types: Collection[str]) -> bool:
    """Whether one creature may block another, as far as these keywords decide.

    Rules 702.9 (flying, reach), 702.31 (horsemanship), 702.36 (fear), 702.13
    (intimidate), 702.28 (shadow), 702.16 (protection) and "can't be blocked".
    """
    a, b = set(attacker_keywords), set(blocker_keywords)
    attacker_colours, blocker_colours = set(attacker_colours), set(blocker_colours)
    if "unblockable" in a:
        return False
    if "flying" in a and not b & {"flying", "reach"}:
        return False
    if "horsemanship" in a and "horsemanship" not in b:
        return False
    if "fear" in a and "artifact" not in blocker_types and "B" not in blocker_colours:
        return False
    if ("intimidate" in a and "artifact" not in blocker_types
            and not attacker_colours & blocker_colours):
        return False
    if ("shadow" in a) != ("shadow" in b):
        return False
    for keyword in a:
        if keyword.startswith("protection_from_"):
            colour = keyword.removeprefix("protection_from_")
            if any(COLOUR_WORDS.get(c) == colour for c in blocker_colours):
                return False
    return True
