"""Which records a curated corpus refuses on sight (FR-148): three rules.

Three shapes the first corpus held that no model should learn from. A
resolution record with no acting ability is an outcome with no cause. A
resolution record whose every acting key maps to no rendered line is the same
thing one step further in: it *has* a key, so the first rule passes it, and
the key names a trait the converter rendered nothing for. And a record
carrying more events than any single ability resolves is a whole game that
landed on one record: one such record in the first corpus held 278 events, 88
card draws and the game's ``player_won``.

**An unattributed event is watched, not refused.** ``attributed_to`` names
the sub-ability clause that produced an event, and the collector writes
``unresolved`` when it walked the acting chain and no clause claimed it. That
is a failure to attribute, not a failure to belong: the bracket collector
records the events that happened inside the ability's *own* resolution, so
the outcome is that ability's whether or not a clause can be named for it. On
the real corpus 12.3% of resolution records with an acting ability have every
event stamped ``unresolved`` and not one of them exceeds ten events, and
another 1% carry a single unattributed side effect — a state-based-action
``zone_change`` death, a ``choice_made``, a ``tapped``. Refusing them threw
away an eighth of the legitimate resolution outcomes, the "died" zone outcome
among them. Combat is the same case at full strength: a damage step resolves
nothing, so there is no chain at all and every combat-damage event is
``unresolved`` by design with its cause in ``cause`` — 98.9% of the class.
The whole-game dumps the rule was reaching for are caught by the no-ability
and event-flood rules instead. ``has_unattributed_events`` keeps the number
available, and ``build-corpus`` reports it as ``unattributed_records``: a
rising share is a statement about the collector, not about the corpus.

``quality_defect`` is pure over the record, so build-corpus applies it in a
worker holding nothing else. ``acting_text_defect`` needs the sidecars' answer
and takes it as a plain callable, which keeps the rule here beside the other
two while the files it reads stay in infrastructure. Either way a test states
one rule in one line.
"""

from __future__ import annotations

from collections.abc import Callable

from effects.domain.effect_targets import events_of
from effects.domain.event_schema import ATTRIBUTION_UNRESOLVED
from effects.domain.provenance import KeyResolution, ProvenanceKey
from effects.domain.records import EffectRecord, RecordKind

NO_ABILITY = "no-ability"
EVENT_FLOOD = "event-flood"
NO_ACTING_TEXT = "no-acting-text"
QUALITY_REASONS: tuple[str, ...] = (NO_ABILITY, EVENT_FLOOD, NO_ACTING_TEXT)

#: The ``attributed_to`` sentinel the collector writes when the producing
#: clause was sought and nothing on the chain claimed the event. Aliased from
#: the schema rather than re-declared, so the watch statistic and the writer
#: cannot drift apart on the spelling of the one string it turns on.
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

    A non-positive ``max_events`` means **no event-flood rule at all**, the
    way ``CapHeap`` reads a non-positive cap and ``--text-cap 0`` reads zero.
    Taken as a literal ceiling instead, zero refuses every record carrying a
    single event -- which is nearly all of them -- so ``build-corpus``, which
    lists ``--max-events-per-record 0`` among the settings zero is real for,
    would write an almost empty corpus and still exit 0. The no-ability rule
    is unconditional: it has no number to turn off.
    """
    if record.kind is RecordKind.RESOLUTION and not record.ability:
        return NO_ABILITY
    events = events_of(record)
    if max_events > 0 and len(events) > max_events:
        return EVENT_FLOOD
    return None


def acting_text_defect(
    record: EffectRecord, resolve: Callable[[ProvenanceKey], KeyResolution],
) -> str | None:
    """``NO_ACTING_TEXT`` when every acting key maps to no rendered line.

    A resolution record whose acting key is Forge's implicit permanent spell
    passes ``quality_defect`` — it has a key — and reaches the head with an
    empty acting slot. Three quarters of the resolution class was that record
    before this rule, weighted like the rarest text in the corpus.

    A ``RUNTIME_ONLY`` key keeps the record: level up, bestow and scavenge, a
    trigger another card's static granted, and a disguise creature's face-down
    face are all real abilities whose text the join does not reach yet, and
    refusing them would be a loss rather than a cleanup. A key inside a range
    the sidecar declared and in neither of its lists raises out of ``resolve``,
    as the contract requires.

    Separate from ``quality_defect`` rather than another branch of it, because
    this one needs the join and that one needs nothing: a caller with no
    sidecars — ``validate-corpus``, a test — still gets the other two rules.

    Args:
        resolve: ``ProvenanceKey -> KeyResolution``, the sidecar cache's answer.
    """
    if record.kind is not RecordKind.RESOLUTION or not record.ability:
        return None
    outcomes = {resolve(key) for key in record.ability}
    if KeyResolution.LINE in outcomes or KeyResolution.RUNTIME_ONLY in outcomes:
        return None
    return NO_ACTING_TEXT


def has_unattributed_events(record: EffectRecord) -> bool:
    """Whether any of this record's events names no producing clause.

    The watch statistic the withdrawn refusal became (see the module
    docstring). Every kind, including ``combat``, because the number is a
    reading of the collector's attribution and a count taken over only the
    kinds some rule happened to reach would move whenever the rules did.
    """
    return any(event.attributed_to == UNRESOLVED for event in events_of(record))
