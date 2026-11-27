"""The right move of each gate-2 effect, read from the combat record's own events.

Gate 2 strips one keyword from one carrier and reads how far each effect's
prediction moves, keyword present minus stripped. The move the keyword's rule
gives, ``t``, has two sides: the record as it happened, read from its events,
and the twin without the keyword, by the rule.

- An amount effect moves by the damage ``D`` the carrier dealt in the step,
  after prevention: the ``amount`` of its ``damage_dealt`` events whose source
  is the carrier (``event_schema.damage_source``), at the subject, or, for
  lifelink's controller, at everything the carrier hit (702.15b). The sign is
  the effect's direction. The first striker's hit (702.7b), trample's excess
  (702.19b), infect's poison (702.90b) and wither's and infect's -1/-1
  counters (702.80a, 702.90a) exist only with the keyword; the damage wither
  and infect deal a creature is marked only without it; the player infect
  hits keeps the life it would otherwise lose.
- A zone-outcome effect moves by 1 where the record shows the outcome the
  keyword makes (the opponent died, the indestructible carrier lived) and by 0
  where it does not: the record contradicts the rule (a regeneration, a
  replacement), so nothing shows the twin to differ.

A prevented hit deals no ``damage_dealt``, so its right move is 0.
Torch-free: the move is read on the host, beside gate 2's forward.
"""

from __future__ import annotations

from collections.abc import Iterable

from effects.domain.damage_step_keywords import CombatParticipant, KeywordEffect, Subject
from effects.domain.effect_head_layout import HeadField
from effects.domain.event_schema import Event, EventType, damage_source
from effects.domain.rule_detectors.keyword_outcomes import died


def right_move(
    effect: KeywordEffect, participant: CombatParticipant, subject: str,
    events: Iterable[Event],
) -> float:
    """``t`` for ``effect`` at ``subject``: the record's value with the keyword
    minus the rule's without it, on the field gate 2 reads."""
    events = tuple(events)
    if effect.field == HeadField.ZONE_OUTCOME:
        return effect.direction * _outcome_shown(effect, subject, events)
    hit = None if effect.subject is Subject.CONTROLLER else subject
    return effect.direction * damage_dealt_by(participant.carrier.id, events, hit)


def damage_dealt_by(source: str, events: Iterable[Event], subject: str | None = None) -> float:
    """The damage ``source`` dealt, after prevention, to ``subject`` or, with
    none named, to everything it hit."""
    return sum(
        float((event.params or {}).get("amount") or 0) for event in events
        if _is_damage_from(event, source) and (subject is None or subject in event.subjects)
    )


def _is_damage_from(event: Event, source: str) -> bool:
    return event.type is EventType.DAMAGE_DEALT and damage_source(event) == source


def _outcome_shown(effect: KeywordEffect, subject: str, events: tuple[Event, ...]) -> float:
    """1 where the record shows the death a positive effect makes or the
    survival a negative one keeps, else 0."""
    return 1.0 if died(events, subject) == (effect.direction > 0) else 0.0
