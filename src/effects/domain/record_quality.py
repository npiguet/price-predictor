"""Which records a curated corpus refuses on sight (FR-148).

Three shapes the first corpus held that no model should learn from. A
resolution record with no acting ability is an outcome with no cause. An
event attributed to ``unresolved`` was produced by a clause the collector
could not find on the acting chain, which in practice marks whole-game event
streams that landed on one record. And a record carrying more events than any
single ability resolves is the same dump seen from the other side: one such
record in the first corpus held 278 events, 88 card draws and the game's
``player_won``.

Pure over the record, so build-corpus applies it in a worker without a
sidecar, and a test can state each rule in one line.
"""

from __future__ import annotations

from effects.domain.effect_targets import events_of
from effects.domain.records import EffectRecord, RecordKind

NO_ABILITY = "no-ability"
UNATTRIBUTED = "unattributed-events"
EVENT_FLOOD = "event-flood"
QUALITY_REASONS: tuple[str, ...] = (NO_ABILITY, UNATTRIBUTED, EVENT_FLOOD)

#: The ``attributed_to`` sentinel the collector writes when the producing
#: clause was sought and nothing on the chain claimed the event.
UNRESOLVED = "unresolved"

#: More events than this on one record is a game, not an ability. The
#: heaviest legitimate resolutions in the first corpus (board wipes over
#: two full boards) sat under forty.
MAX_EVENTS_PER_RECORD = 64


def quality_defect(
    record: EffectRecord, *, max_events: int = MAX_EVENTS_PER_RECORD,
) -> str | None:
    """The first reason this record is refused, or None when it is clean.

    Ordered so the most specific reason wins: a record with no ability is
    reported as that, whatever its events look like.
    """
    if record.kind is RecordKind.RESOLUTION and not record.ability:
        return NO_ABILITY
    events = events_of(record)
    if any(event.attributed_to == UNRESOLVED for event in events):
        return UNATTRIBUTED
    if len(events) > max_events:
        return EVENT_FLOOD
    return None
