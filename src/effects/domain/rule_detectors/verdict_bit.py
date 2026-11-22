"""Detector family ``verdict_bit``: a playability candidate's verdict bit.

A ``decision`` record lists the candidates a player could play and, for each,
the verdict bits the rules decided. A restricting rule (a sorcery's timing,
"can't be cast", "no legal target") is on its ``fires`` side when a
candidate's bit takes the value the restriction gives it, by default false.

Parameters:

- ``bit`` (required): ``can_play``, ``affordable``, ``has_legal_target``,
  or on gen-3 records ``timing`` or ``failed_check``;
- ``check``: with ``bit = failed_check``, the ``failed_checks`` identifier
  (:class:`PlayabilityCheck`, ``land_play_limit`` among them);
- ``fires_when``: the bit's value on the ``fires`` side, ``false`` by default;
- ``responsible_static``: true to count only candidates a static restricts.

The near miss of an ``affordable`` rule is a candidate failing no check but
``cost`` (its ``failed_checks``): one a timing, a zone or a limit also keeps
from being played says nothing of what its cost allows.

The candidates a rule speaks for, each filter narrowing both sides:

- ``land_play``: true for land-play candidates only (FR-033a), false for
  every other;
- ``keyword``: the candidate is the line this keyword produced (a Crew,
  Cycling, Flashback, Dash or Disguise ability, which acts through the
  keyword's own line), compared in ``keyword_names``' spelling;
- ``source_keyword``: the candidate's source prints this keyword on any of
  its lines (a spell with convoke, improvise, escalate or undaunted, cast
  through its own line), read off its sidecar, so a granted keyword is not
  its own;
- ``cost_symbol``: the candidate's cost holds this symbol: ``T``, ``Q``, a
  mana symbol (``S``, ``C``, ``X``, ``W`` …), or a class of them,
  ``hybrid`` (``W/U``), ``monohybrid`` (``2/W``) or ``phyrexian`` (``W/P``);
  read off the line's ``Cost$`` and the candidate's ``cost_after_adjustment``
  mana, so a spell's mana cost counts whatever its line;
- ``cost_part``: the line's ``Cost$`` holds a part of this name, the name
  before its ``<…>`` (``Sac``, ``PayLife``, ``Discard``, ``CollectEvidence``,
  ``Forage``, ``Waterbend``, ``Blight``), compared without case; unreadable
  on a candidate whose line the sidecar dropped (a permanent spell);
- ``script_param``: the line's root segment holds this parameter, a name or a
  name -> value map as ``outcome_event`` reads it;
- ``candidate``: ``cast`` for the candidate's card's own cast (a spell line,
  or a permanent's implicit cast spell the converter drops), ``activated``
  for an activated ability (a land play is neither);
- ``source_type``: the candidate's card, as the snapshot shows it
  (``refs.source``), has this card type (``creature``, ``sorcery``,
  ``kindred`` …), compared without case;
- ``source_tapped``, ``source_sick``: the candidate's card is (true) or is
  not (false) tapped, or summoning sick, in the snapshot;
- ``source_designation``: the candidate's card holds this gen-3 designation
  (``plotted``, ``foretold`` …);
- ``sorcery_speed``: true for a candidate its rules restrict to sorcery
  timing (307.5): a cast of a card neither an instant nor with flash, or an
  activated ability whose line says ``SorcerySpeed$`` or ``Planeswalker$``
  (a loyalty ability, 606.3); false for the rest; unreadable on a keyword's
  line, whose speed the keyword fixes (name it with ``keyword``);
- ``phase_list_first``, ``phase_list_last``: the first or the last step
  the candidate's ``ActivationPhases$`` names (``combat_begin`` for "only
  before attackers are declared", Forge's ``Upkeep->BeginCombat``;
  ``combat_declare_attackers`` for "before blockers"; ``combat_declare_blockers``
  first for "after blockers"), in ``rule_outcomes.PHASE_ORDER``'s spelling;
- ``min_targets``: an integer of 2 or more: the candidate's line has one
  targeting clause (one segment with ``ValidTgts$``), whose ``TargetMin$``
  is at least this; read off the sidecar, so a candidate whose line it
  dropped is unreadable. Its ``legal_targets`` are then that clause's, and
  with ``some_legal_target`` the ``fires`` side of ``has_legal_target``
  false is a clause that has a legal object but fewer than it must name,
  which it cannot name twice (601.2c: Undo with one creature out). A
  candidate with no legal choice whose clause has as many legal objects as
  its ``TargetMin$`` or more is on neither side: a restriction over the
  chosen set (different controllers, a shared type) decides it, not the
  count;
- ``some_legal_target``: the candidate's ``legal_targets`` list is (true) or
  is not (false) empty;
- ``stack_empty``: the decision is taken with the stack empty (true) or
  holding an object (false), read off the snapshot's ``stack_size``: split
  second's near miss is a candidate answering a spell without it;
- ``candidate_face``: ``front`` for a candidate acting through its card's
  first face, ``alternative`` for one acting through another (a double-faced
  card's back face, an Adventure, an Omen, a split card's second half), read
  off the candidate's provenance key, whose ``face`` counts the card's faces
  from 0; which kind of second face it is, the key does not say.

A candidate a filter cannot read is on neither side.

The verdict bits are the verdict head's, read at ``[ACT]`` (the candidate is
the acting line), which is no entity. The condition entity is the
candidate's card (``refs.source``); the outcome entities are where the
verdict's per-entity part, target legality, is written
(``effect_targets.target_legality_subjects``): every card and player of a
decision whose candidate has a targeting clause, none otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_targets import target_legality_subjects
from effects.domain.provenance import SidecarLine, SidecarLookup
from effects.domain.records import (
    Candidate,
    EffectRecord,
    PlayabilityCheck,
    PlayabilitySubkind,
    RecordKind,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import EventlessFamily, FamilyMatch, Matcher, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import (
    CANDIDATE_KINDS,
    candidate_kind,
    card_has_keyword,
    is_keyword_line,
    line_of,
    require_candidate,
    root_params_of,
    script_params_match,
)
from effects.domain.rule_detectors.entity_facts import has_designation
from effects.domain.rule_detectors.keyword_reading import KeywordReading
from effects.domain.rule_outcomes import PHASE_ORDER, forge_phases
from effects.domain.script_text import params as segment_params
from effects.domain.script_text import segments
from effects.domain.state_snapshot import EntityState

BITS = frozenset({"can_play", "affordable", "has_legal_target", "timing", "failed_check"})
FILTERS = frozenset({"keyword", "source_keyword", "cost_symbol", "cost_part", "script_param",
                     "land_play", "candidate_face", "candidate", "source_type",
                     "source_designation", "sorcery_speed", "phase_list_first",
                     "phase_list_last", "source_tapped", "source_sick", "min_targets",
                     "some_legal_target", "stack_empty"})
CANDIDATE_FACES = frozenset({"front", "alternative"})
CHECKS = frozenset(check.value for check in PlayabilityCheck)

#: The checks the declared values pass (``declared_values``).
_VALUE_CHECKS = {
    "fires_when": dv.boolean, "responsible_static": dv.boolean,
    "land_play": dv.boolean, "sorcery_speed": dv.boolean, "keyword": dv.keyword,
    "source_keyword": dv.keyword, "cost_symbol": dv.cost_symbol,
    "script_param": dv.script_param, "source_type": dv.card_type,
    "source_designation": dv.designation, "source_tapped": dv.boolean,
    "source_sick": dv.boolean, "some_legal_target": dv.boolean,
    "min_targets": dv.integer, "phase_list_first": dv.one_of(*PHASE_ORDER),
    "phase_list_last": dv.one_of(*PHASE_ORDER), "stack_empty": dv.boolean,
}
_COST_TOKEN = re.compile(r"[^\s<]+<[^>]*>|\S+")
_TAP = frozenset({"T", "Q"})
_COLOURS = frozenset("WUBRG")
_SYMBOL_CLASSES = ("hybrid", "monohybrid", "phyrexian")


def _symbol_class(symbol: str) -> str | None:
    halves = symbol.split("/")
    if len(halves) < 2:
        return None
    if "P" in halves:
        return "phyrexian"
    if halves[0].isdigit():
        return "monohybrid"
    return "hybrid" if set(halves) <= _COLOURS else None


def _is_symbol(token: str) -> bool:
    """Whether a ``Cost$`` token is a symbol (``T``, ``Q``, a mana symbol),
    not a part."""
    return token.upper() in _TAP or bool(dv.MANA_SYMBOL.match(token.upper()))


def _cost_tokens(line: SidecarLine | None) -> list[str]:
    """The tokens of the line's ``Cost$``; a part's ``<…>`` may hold spaces
    (``Sac<1/CARDNAME/this token>``)."""
    if line is None:
        return []
    return _COST_TOKEN.findall((root_params_of(line) or {}).get("Cost", ""))


def _written_mana(candidate: Candidate) -> str | None:
    """The candidate's ``cost_after_adjustment`` mana, where it is written."""
    mana = (candidate.cost_after_adjustment or {}).get("mana")
    return mana if isinstance(mana, str) else None


def _cost_symbols(written: list[str], mana: str | None) -> set[str]:
    symbols = {token.upper() for token in written if _is_symbol(token)}
    if mana is not None:
        symbols.update(s.upper() for s in re.findall(r"\{([^}]*)\}", mana))
    return symbols


def _known(found: set[str], readable: bool) -> frozenset[str] | None:
    return frozenset(found) if readable else None


def cost_of(candidate: Candidate, lines: SidecarLookup | None
            ) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    """The candidate's cost symbols and its ``Cost$`` part names (upper-cased),
    each None where nothing says what it is."""
    line = line_of(candidate.ability, lines)
    written = _cost_tokens(line)
    mana = _written_mana(candidate)
    parts = {token.split("<", 1)[0].upper() for token in written if not _is_symbol(token)}
    readable = line is not None
    return (_known(_cost_symbols(written, mana), readable or mana is not None),
            _known(parts, readable))


def _holds_symbol(symbols: frozenset[str], wanted: str) -> bool:
    if wanted.lower() in _SYMBOL_CLASSES:
        return any(_symbol_class(s) == wanted.lower() for s in symbols)
    return wanted.upper() in symbols


def _source(record: EffectRecord) -> EntityState | None:
    source = record.state.refs.source
    return record.state.entity(source) if source else None


def _cast_at_sorcery_speed(candidate: Candidate, record: EffectRecord,
                           lines: SidecarLookup | None,
                           keywords: KeywordResolver) -> bool | None:
    """A cast: its card is neither an instant nor with flash."""
    source = _source(record)
    if source is None:
        return None
    flash = "flash" in keywords.keywords_of(source) or card_has_keyword(
        candidate.ability, lines, "Flash")
    return "instant" not in {t.lower() for t in source.types} and not flash


def _activated_at_sorcery_speed(candidate: Candidate,
                                lines: SidecarLookup | None) -> bool | None:
    """An activated ability: its line says ``SorcerySpeed$`` or
    ``Planeswalker$``; unreadable on a keyword's line."""
    line = line_of(candidate.ability, lines)
    if line is None or line.keyword_display_name:
        return None
    found = root_params_of(line) or {}
    return any(str(found.get(name, "")).lower() == "true"
               for name in ("SorcerySpeed", "Planeswalker"))


def sorcery_speed(candidate: Candidate, record: EffectRecord, lines: SidecarLookup | None,
                  keywords: KeywordResolver) -> bool | None:
    """Whether the candidate's rules restrict it to sorcery timing; None
    where nothing read says (a keyword's line, an unreadable card)."""
    kind = candidate_kind(candidate, lines)
    if kind == "cast":
        return _cast_at_sorcery_speed(candidate, record, lines, keywords)
    if kind == "activated":
        return _activated_at_sorcery_speed(candidate, lines)
    return None


def _phase_list(candidate: Candidate, lines: SidecarLookup | None) -> list[str] | None:
    """The steps the candidate's ``ActivationPhases$`` names, in turn order;
    None where its line names none."""
    found = root_params_of(line_of(candidate.ability, lines)) or {}
    phases = forge_phases(found.get("ActivationPhases"))
    return None if phases is None else [p for p in PHASE_ORDER if p in phases]


def sole_clause_minimum(candidate: Candidate, lines: SidecarLookup | None) -> int | None:
    """The ``TargetMin$`` of the candidate's one targeting clause (1 where it
    names none); None where its line is unreadable or has no targeting clause
    or several, or the minimum is no number (an X)."""
    line = line_of(candidate.ability, lines)
    if line is None or not line.script_text:
        return None
    clauses = [segment_params(segment) for segment in segments(line.script_text)]
    targeting = [clause for clause in clauses if "ValidTgts" in clause]
    if len(targeting) != 1:
        return None
    written = targeting[0].get("TargetMin", "1").strip()
    return int(written) if written.isdigit() else None


# ── the candidate tests, one per filter ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Reading:
    """One decision record as the candidate tests read it."""

    record: EffectRecord
    lines: SidecarLookup | None
    keywords: KeywordResolver


#: A filter's test with its declared value bound: whether a candidate is one
#: the filter speaks for.
CandidateTest = Callable[[Candidate, _Reading], bool]


def _land_play(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    return candidate.land_play is bool(wanted)


def _candidate_kind(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    return candidate_kind(candidate, reading.lines) == wanted


def _stack_empty(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    return (reading.record.state.global_.stack_size == 0) is bool(wanted)


def _sorcery_speed(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    return sorcery_speed(candidate, reading.record, reading.lines,
                         reading.keywords) is bool(wanted)


def _source_type(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    source = _source(reading.record)
    return source is not None and wanted.lower() in {t.lower() for t in source.types}


def _source_flag(flag: str) -> Callable[[Any, Candidate, _Reading], bool]:
    """``source_tapped`` or ``source_sick``: the candidate's card's flag in the snapshot."""
    def test(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
        source = _source(reading.record)
        return source is not None and getattr(source, flag) is bool(wanted)
    return test


def _source_designation(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    source = _source(reading.record)
    return source is not None and bool(has_designation(source, wanted))


def _phase_ends(ends: tuple[tuple[int, str], ...], candidate: Candidate,
                reading: _Reading) -> bool:
    listed = _phase_list(candidate, reading.lines)
    return listed is not None and all(listed[index] == phase for index, phase in ends)


def _min_targets(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    minimum = sole_clause_minimum(candidate, reading.lines)
    if minimum is None or minimum < wanted:
        return False
    # Enough legal objects: a restriction on the set decides it, not the count.
    return candidate.has_legal_target or len(set(candidate.legal_targets)) < minimum


def _some_legal_target(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    return bool(candidate.legal_targets) is bool(wanted)


def _candidate_face(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    alternative = wanted == "alternative"
    return bool(candidate.ability) and (candidate.ability[0].face != 0) == alternative


def _keyword(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    return is_keyword_line(line_of(candidate.ability, reading.lines), wanted)


def _source_keyword(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    return card_has_keyword(candidate.ability, reading.lines, wanted)


def _script_param(wanted: Any, candidate: Candidate, reading: _Reading) -> bool:
    return script_params_match(
        root_params_of(line_of(candidate.ability, reading.lines)), wanted)


@dataclass(frozen=True, slots=True)
class _CostWanted:
    """The cost filters a declaration gives, each None where it gives none."""

    symbol: str | None
    part: str | None


def _symbol_fits(symbols: frozenset[str] | None, wanted: str | None) -> bool:
    if wanted is None:
        return True
    return symbols is not None and _holds_symbol(symbols, wanted)


def _part_fits(parts: frozenset[str] | None, wanted: str | None) -> bool:
    if wanted is None:
        return True
    return parts is not None and wanted.upper() in parts


def _cost(wanted: _CostWanted, candidate: Candidate, reading: _Reading) -> bool:
    symbols, parts = cost_of(candidate, reading.lines)
    return _symbol_fits(symbols, wanted.symbol) and _part_fits(parts, wanted.part)


def _one(name: str, test: Callable[[Any, Candidate, _Reading], bool]
         ) -> tuple[tuple[str, ...], Callable[[Mapping[str, Any]], CandidateTest]]:
    """A table row reading one parameter: ``test`` with its declared value."""
    return (name,), lambda params: partial(test, params[name])


def _phase_ends_of(params: Mapping[str, Any]) -> CandidateTest:
    ends = tuple((index, params[name]) for name, index in (
        ("phase_list_first", 0), ("phase_list_last", -1)) if name in params)
    return partial(_phase_ends, ends)


def _cost_of(params: Mapping[str, Any]) -> CandidateTest:
    return partial(_cost, _CostWanted(params.get("cost_symbol"), params.get("cost_part")))


#: The filters in the order a candidate is read: the parameters that switch a
#: row on, and how the row binds their values into its test.
_TESTS = (
    _one("land_play", _land_play),
    _one("candidate", _candidate_kind),
    _one("stack_empty", _stack_empty),
    _one("sorcery_speed", _sorcery_speed),
    _one("source_type", _source_type),
    _one("source_tapped", _source_flag("tapped")),
    _one("source_sick", _source_flag("sick")),
    _one("source_designation", _source_designation),
    (("phase_list_first", "phase_list_last"), _phase_ends_of),
    _one("min_targets", _min_targets),
    _one("some_legal_target", _some_legal_target),
    _one("candidate_face", _candidate_face),
    _one("keyword", _keyword),
    _one("source_keyword", _source_keyword),
    _one("script_param", _script_param),
    (("cost_symbol", "cost_part"), _cost_of),
)


def candidate_tests(params: Mapping[str, Any]) -> tuple[CandidateTest, ...]:
    """The tests of the filters ``params`` gives, in reading order."""
    return tuple(bind(params) for names, bind in _TESTS
                 if any(name in params for name in names))


# ── the verdict ─────────────────────────────────────────────────────────


def _bit(candidate: Candidate, bit: str, check: str | None) -> bool | None:
    if bit == "failed_check":
        if candidate.failed_checks is None:
            return None
        return not any(failed.check == check for failed in candidate.failed_checks)
    return getattr(candidate, bit)


def _near(candidate: Candidate, bit: str) -> bool:
    """Whether the candidate can be a near miss of a ``bit`` rule: of an
    ``affordable`` rule, only one failing no check but ``cost``, whose verdict
    the cost alone decides; any candidate otherwise."""
    if bit != "affordable":
        return True
    return candidate.failed_checks is not None and all(
        failed.check == PlayabilityCheck.COST for failed in candidate.failed_checks)


def _is_decision(record: EffectRecord) -> bool:
    return (record.kind is RecordKind.PLAYABILITY
            and record.subkind is PlayabilitySubkind.DECISION)


@dataclass(frozen=True, slots=True, eq=False)
class VerdictMatcher:
    """A compiled ``verdict_bit`` declaration."""

    bit: str
    check: str | None
    #: The bit's value on the ``fires`` side. A failed_check "bit" reads true
    #: where the check passed, so the default false is a candidate the named
    #: check failed.
    fires_when: bool
    responsible_static: bool
    tests: tuple[CandidateTest, ...]
    keywords: KeywordReading = field(default_factory=KeywordReading)

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> VerdictMatcher:
        return cls(params.get("bit"), params.get("check"),
                   bool(params.get("fires_when", False)),
                   bool(params.get("responsible_static")), candidate_tests(params))

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        if not _is_decision(record):
            return None
        reading = _Reading(record, lines, self.keywords.of(lines))
        values = [(c, _bit(c, self.bit, self.check)) for c in record.payload.candidates
                  if self._speaks_for(c, reading)]
        fires = any(v is self.fires_when for _, v in values)
        if not fires and not self._has_near_miss(values):
            return None
        polarity = Polarity.FIRES if fires else Polarity.DOES_NOT_FIRE
        source = record.state.refs.source
        return FamilyMatch(polarity, fires, (source,) if source else (),
                           target_legality_subjects(record))

    def _speaks_for(self, candidate: Candidate, reading: _Reading) -> bool:
        """Whether the candidate is one the declaration's filters speak for."""
        if self.responsible_static and not candidate.responsible_static:
            return False
        return all(test(candidate, reading) for test in self.tests)

    def _has_near_miss(self, values: list[tuple[Candidate, bool | None]]) -> bool:
        return any(v is not None and _near(c, self.bit) for c, v in values)


# ── the family ──────────────────────────────────────────────────────────


def _check_bit(params: Mapping[str, Any]) -> None:
    bit = params.get("bit")
    if bit not in BITS:
        raise ValueError(f"verdict_bit: unknown bit {bit!r}")
    if bit == "failed_check" and params.get("check") not in CHECKS:
        raise ValueError(f"verdict_bit: a failed_check declaration names its check, "
                         f"one of {sorted(CHECKS)}; got {params.get('check')!r}")


def _check_choices(params: Mapping[str, Any]) -> None:
    if params.get("candidate_face", "front") not in CANDIDATE_FACES:
        raise ValueError(f"verdict_bit: candidate_face is one of {sorted(CANDIDATE_FACES)}")
    if params.get("candidate", "cast") not in CANDIDATE_KINDS:
        raise ValueError(f"verdict_bit: candidate is one of {sorted(CANDIDATE_KINDS)}")


def _check_pairings(params: Mapping[str, Any]) -> None:
    if "min_targets" in params and params["min_targets"] < 2:
        raise ValueError("verdict_bit: min_targets is 2 or more; a clause of one "
                         "target is any targeting clause")
    if "check" in params and params.get("bit") != "failed_check":
        raise ValueError("verdict_bit: check names a failed_check declaration's check")


#: The gen-3 field a parameter, or a bit, makes the family read.
_PARAM_NEEDS = {"land_play": "land_play", "source_designation": "designations"}
_BIT_NEEDS = {"timing": "timing", "failed_check": "failed_checks"}


@dataclass(frozen=True, slots=True)
class VerdictBit(EventlessFamily):
    id: str = "verdict_bit"
    kinds: frozenset[str] = frozenset({"playability"})
    params: frozenset[str] = frozenset(
        {"bit", "check", "fires_when", "responsible_static"}) | FILTERS
    general: frozenset[str] = frozenset(
        {"bit", "check", "fires_when", "keyword", "source_keyword", "cost_symbol", "cost_part",
         "land_play", "candidate_face", "stack_empty"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        needs = {field for name, field in _PARAM_NEEDS.items() if name in params}
        bit_field = _BIT_NEEDS.get(params.get("bit"))
        return frozenset(needs | {bit_field}) if bit_field else frozenset(needs)

    def validate(self, params: Mapping[str, Any]) -> None:
        _check_bit(params)
        _check_choices(params)
        require_candidate(self.id, params)
        dv.check(self.id, params, _VALUE_CHECKS)
        _check_pairings(params)

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return VerdictMatcher.of(params)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


FAMILY = register(VerdictBit())
