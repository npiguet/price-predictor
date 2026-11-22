"""A compiled ``outcome_event`` declaration: what the family reads off its
parameters once, and the matcher that reads one record.

The matcher reads a record in three stages: the record tests (the record
kind, the acting line, the source and target filters, in the order the
family has always read them, each a named predicate with its declared value
bound); the events the declaration wants among the record's; and the side,
unqualified or qualified, that those events put the record on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import SidecarLine, SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind, ResolutionOutcome
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch
from effects.domain.rule_detectors.acting_line import (
    acting_line,
    apis_of,
    card_has_keyword,
    is_keyword_line,
    root_params_of,
    script_params_match,
    source_entities,
    source_is_token_of_kind,
)
from effects.domain.rule_detectors.emittable_events import absence_readable
from effects.domain.rule_detectors.entity_facts import has_face_state
from effects.domain.rule_detectors.outcome_entities import (
    condition_entities,
    near_outcome,
    outcome_entities,
    would_fall_on,
)
from effects.domain.rule_detectors.outcome_events import (
    RECORD_QUALIFIERS,
    EventSpec,
    declared_events,
    event_filters,
    first_strike_victims,
    meets_filters,
    source_dies_verdict,
    token_kind_of,
)
from effects.domain.rule_detectors.outcome_subjects import (
    SUBJECT_QUALIFIERS,
    subject_qualifiers,
    subjects_meet,
)
from effects.domain.script_text import api_of, segments
from effects.domain.script_text import params as params_of

#: Qualifiers read off the event itself against the record's other events.
EVENT_QUALIFIERS = frozenset({"source_dies"})
QUALIFIERS = frozenset(SUBJECT_QUALIFIERS) | RECORD_QUALIFIERS | EVENT_QUALIFIERS
#: The parameters that read the acting line or its source.
LINE_PARAMS = frozenset({"api", "keyword", "source_keyword", "script_param", "chain_param",
                         "source_token_kind", "source_type", "acting_face",
                         "target_face_state"})
#: Record filters read off the acting source and its targets (D4).
SOURCE_FILTERS = frozenset({"source_type", "acting_face", "target_face_state",
                            "half_outcome"})
#: The family's own parameters; every other one is an event filter.
OWN_PARAMS = frozenset({"event", "min_count", "api", "api_in_chain", "keyword",
                        "source_keyword", "script_param", "chain_param", "absent",
                        "token_kind", "source_token_kind"}) | QUALIFIERS | SOURCE_FILTERS
#: The parameters naming the kind of ability, which scope a near miss.
SCOPES = ("api", "keyword", "source_keyword")
#: The record kinds whose payload lists the events a rule's outcome is.
_EVENT_KINDS = frozenset({RecordKind.RESOLUTION, RecordKind.COMBAT, RecordKind.TURN_ACTION})
_GAME_END_EVENTS = frozenset({EventType.PLAYER_WON, EventType.PLAYER_LOST})

#: A record test with its declared value bound: whether the declaration
#: reads the record at all.
RecordTest = Callable[[EffectRecord, "SidecarLookup | None"], bool]
#: An acting-line test with its declared value bound.
LineTest = Callable[["SidecarLine | None"], bool]


def is_scoped(params: Mapping[str, Any]) -> bool:
    """Whether the declaration names the kind of ability, so a near miss is counted."""
    return any(params.get(name) is not None for name in SCOPES)


# ── the acting line ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _KindWanted:
    """``api`` (with ``api_in_chain``) and ``keyword``: a line either names counts."""

    apis: tuple[str, ...] | None
    in_chain: bool
    keyword: str | None


def _acts_through(line: SidecarLine | None, apis: tuple[str, ...], in_chain: bool) -> bool:
    """Whether ``line``'s root API is one of ``apis``, or under ``in_chain``
    any segment's of its chain."""
    if line is None:
        return False
    if line.script_api_type in apis:
        return True
    return in_chain and bool(line.script_text) and any(
        api_of(params_of(segment)) in apis for segment in segments(line.script_text))


def _names_kind(wanted: _KindWanted, line: SidecarLine | None) -> bool:
    by_api = wanted.apis is not None and _acts_through(line, wanted.apis, wanted.in_chain)
    by_keyword = wanted.keyword is not None and is_keyword_line(line, wanted.keyword)
    return by_api or by_keyword


def _script_param(wanted: Any, line: SidecarLine | None) -> bool:
    return script_params_match(root_params_of(line), wanted)


def _chain_param(wanted: Any, line: SidecarLine | None) -> bool:
    """Whether a segment of ``line``'s chain holds the parameter(s) ``wanted``."""
    if line is None or not line.script_text:
        return False
    return any(script_params_match(params_of(segment), wanted)
               for segment in segments(line.script_text))


def _kind_of(params: Mapping[str, Any]) -> LineTest:
    api = params.get("api")
    return partial(_names_kind, _KindWanted(
        None if api is None else apis_of(api), bool(params.get("api_in_chain")),
        params.get("keyword")))


def _names_api_or_keyword(params: Mapping[str, Any]) -> bool:
    return params.get("api") is not None or params.get("keyword") is not None


def _given(name: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda params: name in params


def _named(name: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda params: params.get(name) is not None


#: The acting-line tests in reading order: when a row is switched on, and
#: how it binds its declared values.
_LINE_TESTS = (
    (_names_api_or_keyword, _kind_of),
    (_given("script_param"), lambda params: partial(_script_param, params["script_param"])),
    (_given("chain_param"), lambda params: partial(_chain_param, params["chain_param"])),
)


def _line_fits(tests: tuple[LineTest, ...], record: EffectRecord,
               lines: SidecarLookup | None) -> bool:
    line = acting_line(record, lines)
    return all(test(line) for test in tests)


# ── the record ──────────────────────────────────────────────────────────


def _reads_record(event: EventType, record: EffectRecord, lines: SidecarLookup | None) -> bool:
    """Whether a declaration of ``event`` reads ``record``: an effect half, a
    combat or a turn action; the game-end record only for a game-end event,
    since the events the collector rescued into it are no resolution's."""
    if record.kind not in _EVENT_KINDS:
        return False
    if record.kind is RecordKind.RESOLUTION and record.moment is not Moment.RESOLUTION:
        return False
    return event in _GAME_END_EVENTS or not record.is_game_end_record


def _source_keyword(wanted: str, record: EffectRecord, lines: SidecarLookup | None) -> bool:
    return card_has_keyword(record.ability or (), lines, wanted)


def _source_token_kind(wanted: str, record: EffectRecord, lines: SidecarLookup | None) -> bool:
    return source_is_token_of_kind(record.state, wanted, record.ability)


def _source_type(wanted: str, record: EffectRecord, lines: SidecarLookup | None) -> bool:
    """The source card, as the snapshot shows it, has this card type."""
    return any(wanted in {t.lower() for t in e.types}
               for e in source_entities(record.state, record.ability or ()))


def _acting_face(alternative: bool, record: EffectRecord, lines: SidecarLookup | None) -> bool:
    """``front`` for an acting key on its card's first face, ``alternative``
    for another: an Adventure, a back face, a split half."""
    keys = record.ability or ()
    return bool(keys) and (keys[0].face != 0) == alternative


def _half_outcome(wanted: ResolutionOutcome, record: EffectRecord,
                  lines: SidecarLookup | None) -> bool:
    """The effect half's own outcome: a fizzled or declined spell's half holds
    its card's move off the stack."""
    return getattr(record.payload, "outcome", None) == wanted


def _target_face_state(wanted: str, record: EffectRecord, lines: SidecarLookup | None) -> bool:
    """A target of the acting ability is in this gen-3 face state."""
    targets = [record.state.entity(t) for t in record.state.refs.as_cast]
    return any(t is not None and has_face_state(t, wanted) for t in targets)


def _line_tests_of(params: Mapping[str, Any]) -> tuple[LineTest, ...]:
    return tuple(bind(params) for switched_on, bind in _LINE_TESTS if switched_on(params))


def _acting_line_of(params: Mapping[str, Any]) -> RecordTest:
    return partial(_line_fits, _line_tests_of(params))


#: The record tests after the record kind's, in reading order: when a row is
#: switched on, and how it binds its declared values.
_RECORD_TESTS = (
    (lambda params: bool(_line_tests_of(params)), _acting_line_of),
    (_named("source_keyword"), lambda params: partial(_source_keyword, params["source_keyword"])),
    (_named("source_token_kind"),
     lambda params: partial(_source_token_kind, params["source_token_kind"])),
    (_given("source_type"), lambda params: partial(_source_type, params["source_type"].lower())),
    (_given("acting_face"),
     lambda params: partial(_acting_face, params["acting_face"] == "alternative")),
    (_given("half_outcome"),
     lambda params: partial(_half_outcome, ResolutionOutcome(params["half_outcome"]))),
    (_given("target_face_state"),
     lambda params: partial(_target_face_state, params["target_face_state"])),
)


def record_tests(params: Mapping[str, Any]) -> tuple[RecordTest, ...]:
    """The record tests ``params`` switches on, the record kind's first."""
    event = EventType(params["event"])
    return (partial(_reads_record, event), *(
        bind(params) for switched_on, bind in _RECORD_TESTS if switched_on(params)))


# ── the events ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EventWanted:
    """The events a declaration counts: of its type, meeting its equalities,
    of its token kind (lower-cased) where it names one."""

    type: EventType
    filters: Mapping[str, Any]
    token_kind: str | None

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> EventWanted:
        event = EventType(params["event"])
        token_kind = params.get("token_kind")
        return cls(event, event_filters("outcome_event", event, params, OWN_PARAMS),
                   None if token_kind is None else token_kind.lower())

    def matches(self, event: Event) -> bool:
        if event.type is not self.type or not meets_filters(event, self.filters):
            return False
        return self.token_kind is None or self.token_kind in token_kind_of(event)


@dataclass(frozen=True, slots=True)
class Found:
    """What a record holds of a declaration's events."""

    listed: tuple[Event, ...]
    wanted: list[Event]
    #: Whether an event's absence can be read: a hook-derived event a hookless
    #: record lacks may have happened, so no side reads its absence (V5 #15).
    absence: bool


@dataclass(frozen=True, slots=True)
class Unqualified:
    """The side of a declaration without qualifiers: whether the events happened."""

    min_count: int
    absent: bool
    scoped: bool

    def __call__(self, record: EffectRecord, found: Found) -> FamilyMatch | None:
        happened = len(found.wanted) >= self.min_count
        if self.absent:
            return self._absence_side(record, found, happened)
        if happened:
            return FamilyMatch(Polarity.FIRES, True, condition_entities(record),
                               outcome_entities(record, found.wanted))
        if not self.scoped or not found.absence:
            return None
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False, condition_entities(record),
                           would_fall_on(record))

    @staticmethod
    def _absence_side(record: EffectRecord, found: Found, happened: bool) -> FamilyMatch | None:
        """The rule's outcome is the event's absence, so that is what is observed."""
        if happened:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, condition_entities(record),
                               outcome_entities(record, found.wanted))
        if not found.absence:
            return None
        return FamilyMatch(Polarity.FIRES, True, condition_entities(record),
                           would_fall_on(record))


@dataclass(frozen=True, slots=True)
class Qualified:
    """The side of a declaration with qualifiers: among the events that happened."""

    min_count: int
    scoped: bool
    with_event: EventSpec | None
    without_event: EventSpec | None
    #: The subject qualifiers, name -> declared value (``subject_qualifiers``).
    subjects: Mapping[str, Any]
    source_dies: bool | None

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> Qualified:
        with_event, without_event = (
            EventSpec.of(params[name]) if name in params else None
            for name in ("with_event", "without_event"))
        return cls(int(params.get("min_count", 1)), is_scoped(params), with_event,
                   without_event, subject_qualifiers(params), params.get("source_dies"))

    def __call__(self, record: EffectRecord, found: Found) -> FamilyMatch | None:
        record_ok = self._holds_with(found.listed) and self._lacks_without(found.listed)
        read = self._read(record, found)
        meeting = [e for e, verdict in read if verdict and record_ok]
        if len(meeting) >= self.min_count:
            return FamilyMatch(Polarity.FIRES, True, condition_entities(record),
                               outcome_entities(record, meeting))
        if not found.absence:
            return None
        return self._near_side(record, found, [e for e, _ in read if e not in meeting])

    def _holds_with(self, listed: tuple[Event, ...]) -> bool:
        return self.with_event is None or self.with_event.held_in(listed)

    def _lacks_without(self, listed: tuple[Event, ...]) -> bool:
        return self.without_event is None or not self.without_event.held_in(listed)

    def _read(self, record: EffectRecord, found: Found) -> list[tuple[Event, bool]]:
        """Each wanted event whose qualifiers can be read, with their verdict."""
        read = [(e, self._verdict(record, e, found.listed)) for e in found.wanted]
        return [(e, verdict) for e, verdict in read if verdict is not None]

    def _verdict(self, record: EffectRecord, event: Event,
                 listed: tuple[Event, ...]) -> bool | None:
        """The event's subject and source verdicts together; None where
        either cannot be read."""
        subject = subjects_meet(event.subjects, record.state, self.subjects)
        source = source_dies_verdict(event, listed, self.source_dies)
        return None if subject is None or source is None else subject and source

    def _near_side(self, record: EffectRecord, found: Found,
                   missed: list[Event]) -> FamilyMatch | None:
        if self.source_dies is True:
            # The contrast 510.2 draws: a first striker's victim, dying before
            # it could deal its damage, not every step whose damage dealers
            # all survive.
            victims = first_strike_victims(record, found.wanted, found.listed)
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, condition_entities(record),
                               victims) if victims else None
        if not self.scoped:
            return None
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False, condition_entities(record),
                           near_outcome(record, missed))


def _side_of(params: Mapping[str, Any]) -> Unqualified | Qualified:
    if QUALIFIERS & set(params):
        return Qualified.of(params)
    return Unqualified(int(params.get("min_count", 1)), bool(params.get("absent")),
                       is_scoped(params))


@dataclass(frozen=True, slots=True, eq=False)
class OutcomeMatcher:
    """A compiled ``outcome_event`` declaration."""

    tests: tuple[RecordTest, ...]
    wanted: EventWanted
    #: Every event type the declaration reads (``declared_events``).
    declared: frozenset[EventType]
    side: Callable[[EffectRecord, Found], FamilyMatch | None]

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> OutcomeMatcher:
        return cls(record_tests(params), EventWanted.of(params), declared_events(params),
                   _side_of(params))

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        if not all(test(record, lines) for test in self.tests):
            return None
        listed = record.payload.listed_events()
        wanted = [e for e in listed if self.wanted.matches(e)]
        absence = all(absence_readable(record, e) for e in self.declared)
        return self.side(record, Found(listed, wanted, absence))
