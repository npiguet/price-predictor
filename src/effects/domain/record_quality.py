"""Which records a curated corpus refuses on sight (FR-148).

Three shapes the first corpus held that no model should learn from. A
resolution record with no acting ability is an outcome with no cause. An
event attributed to ``unresolved`` was produced by a clause the collector
could not find on the acting chain, which in practice marks whole-game event
streams that landed on one record. And a record carrying more events than any
single ability resolves is the same dump seen from the other side: one such
record in the first corpus held 278 events, 88 card draws and the game's
``player_won``.

**Combat records are exempt from the unattributed rule.** Nothing is
resolving in a damage step, so there is no acting chain for an event to be
attributed to and the collector stamps every combat-damage event
``unresolved`` by design; what caused the damage lives in the event's
``cause`` instead. Applied to combat the rule refused 98.9% of the combat
records in the real corpus — the whole sampling class, for being exactly
what it is supposed to be.

Pure over the record, so build-corpus applies it in a worker without a
sidecar, and a test can state each rule in one line.
"""

from __future__ import annotations

from effects.domain.effect_targets import events_of
from effects.domain.event_schema import ATTRIBUTION_UNRESOLVED
from effects.domain.records import EffectRecord, RecordKind

NO_ABILITY = "no-ability"
UNATTRIBUTED = "unattributed-events"
EVENT_FLOOD = "event-flood"
QUALITY_REASONS: tuple[str, ...] = (NO_ABILITY, UNATTRIBUTED, EVENT_FLOOD)

#: The ``attributed_to`` sentinel the collector writes when the producing
#: clause was sought and nothing on the chain claimed the event. Aliased from
#: the schema rather than re-declared, so the rule and the writer cannot drift
#: apart on the spelling of the one string the whole check turns on.
UNRESOLVED = ATTRIBUTION_UNRESOLVED

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

    The unattributed rule skips ``combat``: a damage step resolves nothing,
    so the collector has no chain to attribute an event to and stamps every
    one ``unresolved`` (the cause is in ``cause``). Judged by that rule the
    class is 98.9% defective, which is a statement about the rule.

    A non-positive ``max_events`` means **no event-flood rule at all**, the
    way ``CapHeap`` reads a non-positive cap and ``--text-cap 0`` reads zero.
    Taken as a literal ceiling instead, zero refuses every record carrying a
    single event -- which is nearly all of them -- so ``build-corpus``, which
    lists ``--max-events-per-record 0`` among the settings zero is real for,
    would write an almost empty corpus and still exit 0. The other two rules
    are unconditional: neither has a number to turn off.
    """
    if record.kind is RecordKind.RESOLUTION and not record.ability:
        return NO_ABILITY
    events = events_of(record)
    if record.kind is not RecordKind.COMBAT and any(
        event.attributed_to == UNRESOLVED for event in events
    ):
        return UNATTRIBUTED
    if max_events > 0 and len(events) > max_events:
        return EVENT_FLOOD
    return None
