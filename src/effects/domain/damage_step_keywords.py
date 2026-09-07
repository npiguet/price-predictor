"""Gate 2's table: the eight damage-step keywords, one row each.

Gate 2 asks a narrow question with a clean answer: when a keyword is removed
from a combat participant's model input, does the prediction move the way the
rules say it should? Each row below carries the three things that makes
checkable — which records qualify, which output fields the keyword touches, and
which way they move when it is present.

The perturbation is **model-side**, not a game fork: the keyword is removed from
the entity's ability token where it is printed or attachment-granted, and from
the overlay's temporarily-granted-keywords channel where it is not. That is what
lets gate 2 run at stage one, before any fork machinery exists — and its
per-keyword verdict is what decides whether that machinery gets built at all.

These are the eight keywords whose whole effect lands inside a damage step, so a
combat record contains both the cause and the consequence. Evasion keywords are
deliberately absent: their signal is in playability records, which stage one
does not collect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Subject(StrEnum):
    """Whose output field the keyword moves."""

    #: The creature carrying the keyword.
    CARRIER = "carrier"
    #: The creature it is fighting.
    OPPONENT = "opponent"
    #: The player who controls the carrier.
    CONTROLLER = "controller"
    #: The player being attacked.
    DEFENDING_PLAYER = "defending_player"


@dataclass(frozen=True, slots=True)
class KeywordEffect:
    """One field the keyword moves, and which way."""

    field: str
    subject: Subject
    #: Sign of the change when the keyword is **present** versus absent.
    direction: int
    #: For a categorical field, the class whose probability should move.
    category: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError(
                f"direction must be -1 or 1, got {self.direction} for "
                f"{self.field}"
            )


@dataclass(frozen=True, slots=True)
class DamageStepKeyword:
    """One row of gate 2's table."""

    keyword: str
    #: What makes a combat record a qualifying observation, in words. The
    #: evaluator's predicate is named here so the table and the code cannot
    #: drift into testing different populations.
    qualifies_when: str
    effects: tuple[KeywordEffect, ...]

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(effect.field for effect in self.effects)


DAMAGE_STEP_KEYWORDS: tuple[DamageStepKeyword, ...] = (
    DamageStepKeyword(
        keyword="first_strike",
        qualifies_when=(
            "the carrier is in combat against a creature with neither first "
            "strike nor double strike"
        ),
        effects=(
            # It kills before being killed, so it takes less damage back.
            KeywordEffect("damage_taken", Subject.CARRIER, direction=-1),
            KeywordEffect(
                "zone_outcome", Subject.CARRIER, direction=-1, category="died",
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="double_strike",
        qualifies_when="the carrier is in combat and deals damage",
        effects=(
            # Damage in both steps: twice as much reaches the other side.
            KeywordEffect("damage_taken", Subject.OPPONENT, direction=1),
            KeywordEffect(
                "zone_outcome", Subject.OPPONENT, direction=1, category="died",
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="deathtouch",
        qualifies_when=(
            "the carrier deals combat damage to a creature that would survive "
            "that much damage normally"
        ),
        effects=(
            KeywordEffect(
                "zone_outcome", Subject.OPPONENT, direction=1, category="died",
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="lifelink",
        qualifies_when="the carrier deals combat damage",
        effects=(
            KeywordEffect("life_delta", Subject.CONTROLLER, direction=1),
        ),
    ),
    DamageStepKeyword(
        keyword="trample",
        qualifies_when=(
            "the carrier is blocked by a creature with less toughness than the "
            "carrier's power"
        ),
        effects=(
            # The excess runs over: the defender loses more life.
            KeywordEffect("life_delta", Subject.DEFENDING_PLAYER, direction=-1),
        ),
    ),
    DamageStepKeyword(
        keyword="indestructible",
        qualifies_when=(
            "the carrier takes combat damage at least equal to its toughness"
        ),
        effects=(
            KeywordEffect(
                "zone_outcome", Subject.CARRIER, direction=-1, category="died",
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="wither",
        qualifies_when="the carrier deals combat damage to a creature",
        effects=(
            # Damage arrives as counters instead of as damage.
            KeywordEffect("counters_delta_m1m1", Subject.OPPONENT, direction=1),
            KeywordEffect("damage_taken", Subject.OPPONENT, direction=-1),
        ),
    ),
    DamageStepKeyword(
        keyword="infect",
        qualifies_when="the carrier deals combat damage to a creature or player",
        effects=(
            KeywordEffect("counters_delta_m1m1", Subject.OPPONENT, direction=1),
            KeywordEffect("damage_taken", Subject.OPPONENT, direction=-1),
            KeywordEffect(
                "poison_delta", Subject.DEFENDING_PLAYER, direction=1,
            ),
        ),
    ),
)

KEYWORDS_BY_NAME: dict[str, DamageStepKeyword] = {
    row.keyword: row for row in DAMAGE_STEP_KEYWORDS
}

#: Gate 2's thresholds (FR-119). Both are contract.
MIN_QUALIFYING_RECORDS = 200
MIN_DIRECTION_AGREEMENT = 0.70
