"""Labels one entity's several lasting-change events muddle (finding 10.7).

Each event applies on its own (:mod:`effects.domain.effect_targets`); these
read the record's events of one entity together and take back what they cannot
say as a whole.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from effects.domain.effect_head_layout import HeadField
from effects.domain.event_schema import Event
from effects.domain.input_vocabularies import vocabulary_slot
from effects.domain.keyword_names import changed_keyword
from effects.domain.output_vocabularies import KEYWORDS_CHANGED

_ENCHANT = vocabulary_slot(changed_keyword("enchant"), KEYWORDS_CHANGED)


def mixed_pt_durations(pt_changes: Iterable[Event], targets: Mapping) -> None:
    """``pt_duration`` is unknown on an entity whose ``pt_change`` events last
    for different times: its deltas are summed, one duration cannot label them."""
    _forget_mixed(pt_changes, targets, HeadField.PT_DURATION)


def mixed_type_color_durations(
    type_and_color_changes: Iterable[Event], targets: Mapping,
) -> None:
    """``type_color_duration`` is unknown on an entity whose ``type_change`` and
    ``color_change`` events together last for different times: one field labels
    both kinds, and their gains and losses are merged."""
    _forget_mixed(type_and_color_changes, targets, HeadField.TYPE_COLOR_DURATION)


def _forget_mixed(events: Iterable[Event], targets: Mapping, name: str) -> None:
    durations: dict[str, set[str | None]] = defaultdict(set)
    for event in events:
        for subject in event.subjects:
            durations[subject].add(event.duration)
    for subject in _mixed(durations):
        targets[subject].forget(name)


def _mixed(durations: Mapping[str, set]) -> list[str]:
    return [subject for subject, lasting in durations.items() if len(lasting) > 1]


def cancel_enchant_swaps(entries: Iterable) -> None:
    """An entity that gains one enchant restriction and loses another (Animate
    Dead) changes no ``enchant`` column: both spellings fold to one."""
    for entry in entries:
        gained = entry.fields.get(HeadField.KEYWORDS_GAINED)
        lost = entry.fields.get(HeadField.KEYWORDS_LOST)
        if _swapped(gained, lost):
            gained[_ENCHANT] = lost[_ENCHANT] = 0.0


def _swapped(gained: list[float] | None, lost: list[float] | None) -> bool:
    if _ENCHANT is None:
        return False
    return _set_at(gained) and _set_at(lost)


def _set_at(flags: list[float] | None) -> bool:
    return bool(flags) and bool(flags[_ENCHANT])
