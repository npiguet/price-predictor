"""Detector family ``resolution_outcome``: how a spell or ability left the stack.

A stack object leaves the stack ``resolved``, ``fizzled`` (every target
became illegal, 608.2b), ``partially_fizzled``, ``declined`` or ``countered``
(701.6a). A rule about leaving the stack is on its ``fires`` side when the
outcome is the one it names, and on its near miss when the outcome is one of
``near``. The near miss is counted only where ``keyword``, ``source_keyword``,
``ability_kind`` or ``targeted`` says which stack objects the rule speaks for
(``SCOPES``): without one, every spell and ability of the corpus that resolved would be a
near miss of a counter rule.

The outcome is written on the effect half (root spec § 8.14, FR-042b), and a
countered object writes no effect half at all: its fate is the countering
clause's effect half, which holds the countered spell's ``spell_countered``
event or the countered ability's ``ability_countered``. So a ``countered``
observation is a half holding either (``event_schema.countered_objects``:
a spell by its card, an ability by its host), read as such only for a rule
naming ``countered``, and every other outcome is the effect half's own.

Parameters:

- ``outcome`` (required): the outcome on the ``fires`` side;
- ``near``: the outcomes of the near miss, ``["resolved"]`` by default;
- ``keyword``: the record acts through this keyword's line (ward's
  trigger, the half that counters or resolves);
- ``source_keyword``: the stack object's card prints this keyword (a gift
  spell): the countered object's card on a countering half;
- ``ability_kind``: ``cast`` or ``activated``: the acting key is the card's
  cast spell or an activated ability (``acting_line.ability_kind``), so a
  trigger's resolution is neither; of a countered object it is read from the
  counter's event, not the countering half's own key: a ``spell_countered``
  names a cast spell, an ``ability_countered`` is never a cast;
- ``targeted``: true to count only stack objects that had a target, so a
  fizzle rule's near miss is a targeted spell that resolved. The targets are
  read as cast (``Refs.as_cast``: ``targets_at_cast``, else ``targets``).

The condition entities are the stack object (``refs.source``, or the
countered object) and its targets, whose legality a fizzle reads; the outcome
entity is the stack object, which is what leaves the stack resolved, fizzled
or countered. The outcome is read at the record, through the verdict head's
``resolution_outcome``, on the effect half that carries it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EventType, countered_objects
from effects.domain.provenance import SidecarLookup
from effects.domain.records import (
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import EventlessFamily, FamilyMatch, Matcher, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import (
    CANDIDATE_KINDS,
    ability_kind,
    acting_line,
    card_has_keyword,
    is_keyword_line,
)


@dataclass(frozen=True, slots=True)
class _Observed:
    outcome: ResolutionOutcome
    condition: tuple[str, ...]
    outcome_entities: tuple[str, ...]
    targeted: bool
    #: Whether the record's events say what the stack object is (a countered
    #: object: ``kind``, None for an ability) rather than the acting key.
    from_event: bool = False
    kind: str | None = None


def _observed(record: EffectRecord, countered_asked: bool) -> _Observed | None:
    """The stack object's outcome a record shows, or None where it shows none."""
    if record.kind is not RecordKind.RESOLUTION:
        return None
    if record.moment is Moment.ACTIVATION:
        return _activation_observed(record)
    payload = record.payload
    if not isinstance(payload, ResolutionPayload):
        return None
    countered = _countered(payload) if countered_asked else ()
    if countered:
        return _Observed(ResolutionOutcome.COUNTERED, countered, countered, True,
                         True, _countered_kind(payload))
    if payload.outcome is None:
        return None
    # The targets as cast (V5 #25).
    return _stack_object(record, payload.outcome)


def _stack_object(record: EffectRecord, outcome: ResolutionOutcome) -> _Observed:
    """``outcome`` of the record's own stack object and its targets as cast."""
    refs = record.state.refs
    source = (refs.source,) if refs.source else ()
    return _Observed(outcome, tuple(dict.fromkeys((*source, *refs.as_cast))), source,
                     bool(refs.as_cast))


def _activation_observed(record: EffectRecord) -> _Observed | None:
    outcome = getattr(record.payload, "outcome", None)
    return None if outcome is None else _stack_object(record, outcome)


def _countered(payload: ResolutionPayload) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        ref for event in payload.events for ref in countered_objects(event)))


def _countered_kind(payload: ResolutionPayload) -> str | None:
    """``cast`` where a ``spell_countered`` event is held, none for an ability:
    what leaves the stack is the countered object, not the counter's own line."""
    spell = any(event.type is EventType.SPELL_COUNTERED for event in payload.events)
    return "cast" if spell else None


def _prints(record: EffectRecord, lines: SidecarLookup | None, observed: _Observed,
            keyword: str) -> bool:
    """Whether the stack object's card prints ``keyword``: the countered
    object's on a countering half, the acting line's card otherwise."""
    if observed.outcome is ResolutionOutcome.COUNTERED and observed.outcome_entities != (
            (record.state.refs.source,) if record.state.refs.source else ()):
        return any(entity is not None and card_has_keyword(entity.line_keys, lines, keyword)
                   for entity in map(record.state.entity, observed.outcome_entities))
    return card_has_keyword(record.ability or (), lines, keyword)


#: The parameters that say which stack objects a rule speaks for, so which
#: records its near miss is (``targeted`` when true).
SCOPES = frozenset({"keyword", "source_keyword", "ability_kind", "targeted"})


@dataclass(frozen=True, slots=True)
class ResolutionOutcomeFamily(EventlessFamily):
    id: str = "resolution_outcome"
    kinds: frozenset[str] = frozenset({"resolution"})
    params: frozenset[str] = frozenset({"outcome", "near", "targeted", "source_keyword",
                                        "keyword", "ability_kind"})
    general: frozenset[str] = frozenset({"outcome", "near"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def validate(self, params: Mapping[str, Any]) -> None:
        ResolutionOutcome(params["outcome"])
        for near in params.get("near", ["resolved"]):
            ResolutionOutcome(near)
        if params.get("ability_kind", "cast") not in CANDIDATE_KINDS:
            raise ValueError(
                f"resolution_outcome: ability_kind is one of {sorted(CANDIDATE_KINDS)}")
        dv.check(self.id, params, {"targeted": dv.boolean, "source_keyword": dv.keyword,
                                   "keyword": dv.keyword})

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return OutcomeMatcher.of(params)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


@dataclass(frozen=True, slots=True, eq=False)
class OutcomeMatcher:
    """A compiled ``resolution_outcome`` declaration."""

    fires: ResolutionOutcome
    near: frozenset[ResolutionOutcome]
    #: The declaration carries a scoping parameter: its near miss is counted.
    scoped: bool
    targeted: bool
    #: ``ability_kind``, None where absent.
    kind: str | None
    source_keyword: str | None
    #: ``keyword``, None where absent.
    keyword: str | None

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> OutcomeMatcher:
        return cls(
            ResolutionOutcome(params["outcome"]),
            frozenset(ResolutionOutcome(o) for o in params.get("near", ["resolved"])),
            any(params.get(name) for name in SCOPES),
            bool(params.get("targeted")),
            params.get("ability_kind"), params.get("source_keyword"), params.get("keyword"))

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        observed = _observed(record, ResolutionOutcome.COUNTERED in self.near | {self.fires})
        if observed is None or not self._fits(record, lines, observed):
            return None
        if observed.outcome is self.fires:
            return FamilyMatch(Polarity.FIRES, True, observed.condition,
                               observed.outcome_entities)
        if observed.outcome in self.near and self.scoped:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, observed.condition,
                               observed.outcome_entities)
        return None

    def _fits(self, record: EffectRecord, lines: SidecarLookup | None,
              observed: _Observed) -> bool:
        """Whether the stack object is one ``targeted``, ``ability_kind``,
        ``source_keyword`` and ``keyword`` speak for, each where given."""
        return (self._targeted_fits(observed) and self._kind_fits(record, lines, observed)
                and self._keywords_fit(record, lines, observed))

    def _targeted_fits(self, observed: _Observed) -> bool:
        return not self.targeted or observed.targeted

    def _kind_fits(self, record: EffectRecord, lines: SidecarLookup | None,
                   observed: _Observed) -> bool:
        if self.kind is None:
            return True
        found = observed.kind if observed.from_event else ability_kind(record.ability or (), lines)
        return found == self.kind

    def _keywords_fit(self, record: EffectRecord, lines: SidecarLookup | None,
                      observed: _Observed) -> bool:
        if self.source_keyword is not None and not _prints(
                record, lines, observed, self.source_keyword):
            return False
        return self.keyword is None or is_keyword_line(acting_line(record, lines), self.keyword)


FAMILY = register(ResolutionOutcomeFamily())
