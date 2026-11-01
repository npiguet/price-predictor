"""Detector family ``outcome_event``: a resolution producing an event of one kind.

Many rules are stated by what a resolution does: a permanent spell enters the
battlefield (608.3a), an instant goes to the graveyard (608.2n), a token or a
delayed trigger is created. A record shows such a rule on its ``fires`` side
when its effect half holds the event. The near miss is a resolution of the
same kind of ability that did not produce it, so it is counted only when
``api`` names that kind: without it, every other resolution in the corpus
would count as a near miss.

Parameters:

- ``event`` (required): the event type (``zone_change``, ``token_created`` …);
- ``from_zone``, ``to_zone``: a ``zone_change``'s zones;
- ``min_count``: how many such events the record must hold, 1 by default;
- ``api``: the acting line's script API type (``ChangeZone``, ``Token`` …),
  which restricts both sides to resolutions of that API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register


def acting_api(record: EffectRecord, lines: SidecarLookup | None) -> str | None:
    """The acting line's script API type, or None where no line is readable."""
    if lines is None or not record.ability:
        return None
    line = lines.first_line(record.ability)
    return line.script_api_type if line is not None else None


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    id: str = "outcome_event"
    params: frozenset[str] = frozenset({"event", "from_zone", "to_zone", "min_count", "api"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        event = EventType(params["event"])
        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
            return None
        api = params.get("api")
        if api is not None and acting_api(record, lines) != api:
            return None
        wanted = [
            e for e in record.payload.listed_events()
            if e.type is event and all(
                (e.params or {}).get(name) == params[name]
                for name in ("from_zone", "to_zone") if name in params)]
        if len(wanted) >= int(params.get("min_count", 1)):
            return FamilyMatch(Polarity.FIRES, True,
                               tuple(dict.fromkeys(s for e in wanted for s in e.subjects)))
        if api is None:
            return None
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False, ())


FAMILY = register(OutcomeEvent())
