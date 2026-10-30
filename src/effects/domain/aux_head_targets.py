"""What the ``[ACT]`` and ``[GLOBAL]`` heads train on (FR-060a).

The per-entity targets (:mod:`effects.domain.effect_targets`) say what
happened to each entity; these say what the acting line's verdict was and
which objects it created, one row per record rather than per entity.
"""

from __future__ import annotations

from dataclasses import dataclass

from effects.domain.effect_head_input import CARD_TYPES, vocabulary_slot
from effects.domain.effect_head_layout import (
    CREATED_OBJECT_SLOTS,
    CREATED_OBJECTS_WIDTH,
    CREATED_SLOT_WIDTH,
    VERDICT_BITS,
    VERDICT_WIDTH,
    CreatedSlot,
)
from effects.domain.effect_targets import overlay_index
from effects.domain.event_schema import EventType
from effects.domain.records import (
    ActivationPayload,
    EffectRecord,
    PlayabilityDecisionPayload,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
    events_of,
)
from effects.domain.state_snapshot import COLORS


def verdict_targets(
    record: EffectRecord,
) -> tuple[list[float], list[bool]] | None:
    """The verdict head's targets at ``[ACT]`` and which of them this record sets.

    A ``decision`` record sets its candidate's verdict bits, a cost half the
    mana it paid by colour, a ``trigger`` record whether it fired; every other
    column is masked out, since a cost half is no evidence of playability.
    """
    target = [0.0] * VERDICT_WIDTH
    mask = [False] * VERDICT_WIDTH
    bits = len(VERDICT_BITS)
    match record.payload:
        case PlayabilityDecisionPayload(candidates=(candidate, *_)):
            for index, value in enumerate(
                (candidate.can_play, candidate.affordable, candidate.has_legal_target)
            ):
                target[index] = 1.0 if value else 0.0
                mask[index] = True
        case ActivationPayload(costs=costs):
            paid = costs.mana_by_color
            for offset, color in enumerate(COLORS):
                target[bits + offset] = float(paid.get(color, 0))
                mask[bits + offset] = True
        case TriggerPayload(fired=fired):
            target[-1] = 1.0 if fired else 0.0
            mask[-1] = True
        case _:
            return None
    return target, mask


def supervises_created_objects(record: EffectRecord) -> bool:
    """Effect halves and in-place rewrites carry created-objects targets; any
    other rewrite ran another ability, whose effect half supervises it."""
    match record.payload:
        case ResolutionPayload():
            return True
        case RewritePayload(rewritten=rewritten):
            return rewritten
    return False


def created_objects_targets(record: EffectRecord) -> list[float]:
    """The created-objects head's targets at ``[GLOBAL]`` (FR-078).

    Scripted groups first by id, then characteristics-only groups by
    descending count; the flag past the last slot marks more than K groups.
    """
    groups: dict[tuple, TokenGroup] = {}
    for event in events_of(record):
        if event.type is not EventType.TOKEN_CREATED:
            continue
        found = TokenGroup.of(event.params)
        group = groups.setdefault(found.key, found)
        group.count += int(event.params.get("count", 1) or 1)

    ordered = sorted(
        (g for g in groups.values() if g.scripted), key=lambda g: g.id,
    ) + sorted(
        (g for g in groups.values() if not g.scripted), key=lambda g: -g.count,
    )
    target = [0.0] * CREATED_OBJECTS_WIDTH
    for slot, group in enumerate(ordered[:CREATED_OBJECT_SLOTS]):
        group.write(target, slot * CREATED_SLOT_WIDTH)
    target[-1] = 1.0 if len(ordered) > CREATED_OBJECT_SLOTS else 0.0
    return target


@dataclass
class TokenGroup:
    """Tokens one record created that the head counts as one group (FR-078):
    by token-script id where the event names one, else by characteristics."""

    scripted: bool
    id: str
    power: object
    toughness: object
    types: str
    keywords: tuple
    count: int = 0

    @classmethod
    def of(cls, params: dict) -> TokenGroup:
        script = params.get("token_script_id") or None
        traits = params.get("characteristics") or {}
        keywords = traits.get("keywords") or ()
        if isinstance(keywords, str):
            keywords = (keywords,)
        return cls(
            scripted=script is not None, id=str(script or ""),
            power=traits.get("power"), toughness=traits.get("toughness"),
            types=str(traits.get("types", "")), keywords=tuple(keywords),
        )

    @property
    def key(self) -> tuple:
        if self.scripted:
            return ("script", self.id)
        return ("traits", self.types, self.power, self.toughness,
                tuple(sorted(map(str, self.keywords))))

    def write(self, target: list[float], base: int) -> None:
        """This group's slot, from ``base`` on."""
        target[base + CreatedSlot.PRESENT] = 1.0
        target[base + CreatedSlot.SCRIPTED] = 1.0 if self.scripted else 0.0
        target[base + CreatedSlot.COUNT] = float(self.count)
        target[base + CreatedSlot.POWER] = max(_number(self.power), 0.0)
        target[base + CreatedSlot.TOUGHNESS] = max(_number(self.toughness), 0.0)
        for card_type in self.types.split(" - ", 1)[0].split():
            index = vocabulary_slot(card_type, CARD_TYPES)
            if index is not None:
                target[base + CreatedSlot.TYPES + index] = 1.0
        for keyword in self.keywords:
            index = overlay_index(str(keyword))
            if index is not None:
                target[base + CreatedSlot.KEYWORDS + index] = 1.0


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
