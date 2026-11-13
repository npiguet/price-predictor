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

The candidates a rule speaks for, each filter narrowing both sides:

- ``land_play``: true for land-play candidates only (FR-033a), false for
  every other; read off the gen-3 candidate, so a gen-2 candidate, or one
  collected before land plays were candidates, is unreadable;
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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_targets import target_legality_subjects
from effects.domain.provenance import SidecarLookup
from effects.domain.records import (
    Candidate,
    EffectRecord,
    PlayabilityCheck,
    PlayabilitySubkind,
    RecordKind,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import (
    card_has_keyword,
    is_keyword_line,
    line_of,
    root_params_of,
    script_params_match,
)
from effects.domain.rule_detectors.entity_facts import has_designation
from effects.domain.rule_outcomes import PHASE_ORDER, forge_phases

GEN2_BITS = frozenset({"can_play", "affordable", "has_legal_target"})
GEN3_BITS = frozenset({"timing", "failed_check"})
FILTERS = frozenset({"keyword", "source_keyword", "cost_symbol", "cost_part", "script_param",
                     "land_play", "candidate_face", "candidate", "source_type",
                     "source_designation", "sorcery_speed", "phase_list_first",
                     "phase_list_last"})
CANDIDATE_FACES = frozenset({"front", "alternative"})
#: What a candidate is: its card's cast, or an activated ability (a land play
#: is ``land_play``'s).
CANDIDATE_KINDS = frozenset({"cast", "activated"})
CHECKS = frozenset(check.value for check in PlayabilityCheck)

#: A mana symbol as ``Cost$`` spells it (``W``, ``12``, ``W/U``, ``2/W``,
#: ``W/P``); ``T`` and ``Q`` are symbols too, and any other token is a part.
_MANA = re.compile(r"^(\d+|[WUBRGCSXYZ]|[WUBRGC\d]+/[WUBRGCP](/P)?)$")
_COST_TOKEN = re.compile(r"[^\s<]+<[^>]*>|\S+")
_TAP = frozenset({"T", "Q"})
_COLOURS = frozenset("WUBRG")


def _symbol_class(symbol: str) -> str | None:
    halves = symbol.split("/")
    if len(halves) < 2:
        return None
    if "P" in halves:
        return "phyrexian"
    if halves[0].isdigit():
        return "monohybrid"
    return "hybrid" if set(halves) <= _COLOURS else None


def cost_of(candidate: Candidate, lines: SidecarLookup | None
            ) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    """The candidate's cost symbols and its ``Cost$`` part names (upper-cased),
    each None where nothing says what it is."""
    line = line_of(candidate.ability, lines)
    found = root_params_of(line)
    # A part's ``<…>`` may hold spaces (``Sac<1/CARDNAME/this token>``).
    written = _COST_TOKEN.findall((found or {}).get("Cost", "")) if line is not None else []
    symbols = {t.upper() for t in written if t.upper() in _TAP or _MANA.match(t.upper())}
    parts = {t.split("<", 1)[0].upper() for t in written if t.upper() not in _TAP
             and not _MANA.match(t.upper())}
    mana = (candidate.cost_after_adjustment or {}).get("mana")
    if isinstance(mana, str):
        symbols.update(s.upper() for s in re.findall(r"\{([^}]*)\}", mana))
    known = line is not None or isinstance(mana, str)
    return (frozenset(symbols) if known else None,
            frozenset(parts) if line is not None else None)


def _holds_symbol(symbols: frozenset[str], wanted: str) -> bool:
    if wanted.lower() in ("hybrid", "monohybrid", "phyrexian"):
        return any(_symbol_class(s) == wanted.lower() for s in symbols)
    return wanted.upper() in symbols


def candidate_kind(candidate: Candidate, lines: SidecarLookup | None) -> str | None:
    """``cast`` or ``activated``; None for a land play or an unreadable key."""
    if candidate.land_play or not candidate.ability:
        return None
    key = candidate.ability[0]
    if key.trait_kind == "keyword":
        return "activated"
    if key.trait_kind != "spell":
        return None
    line = line_of(candidate.ability, lines)
    if line is None:
        # A permanent's implicit cast spell, which the converter drops.
        return "cast" if key.index_within_kind == 0 and key.option is None else None
    found = root_params_of(line) or {}
    return "cast" if "SP" in found else "activated" if "AB" in found else None


def sorcery_speed(candidate: Candidate, record: EffectRecord,
                  lines: SidecarLookup | None) -> bool | None:
    """Whether the candidate's rules restrict it to sorcery timing; None
    where nothing read says (a keyword's line, an unreadable card)."""
    kind = candidate_kind(candidate, lines)
    if kind == "cast":
        source = _source(record)
        if source is None:
            return None
        flash = "flash" in KeywordResolver(lines).keywords_of(source) or card_has_keyword(
            candidate.ability, lines, "Flash")
        return "instant" not in {t.lower() for t in source.types} and not flash
    if kind == "activated":
        line = line_of(candidate.ability, lines)
        if line is None or line.keyword_display_name:
            return None
        found = root_params_of(line) or {}
        return any(str(found.get(name, "")).lower() == "true"
                   for name in ("SorcerySpeed", "Planeswalker"))
    return None


def _phase_list(candidate: Candidate, lines: SidecarLookup | None) -> list[str] | None:
    """The steps the candidate's ``ActivationPhases$`` names, in turn order;
    None where its line names none."""
    found = root_params_of(line_of(candidate.ability, lines)) or {}
    phases = forge_phases(found.get("ActivationPhases"))
    return None if phases is None else [p for p in PHASE_ORDER if p in phases]


def _source(record: EffectRecord):
    source = record.state.refs.source
    return record.state.entity(source) if source else None


def _passes(candidate: Candidate, lines: SidecarLookup | None,
            params: Mapping[str, Any], record: EffectRecord) -> bool:
    """Whether the candidate is one the declaration's filters speak for."""
    if "land_play" in params and candidate.land_play is not bool(params["land_play"]):
        return False
    if "candidate" in params and candidate_kind(candidate, lines) != params["candidate"]:
        return False
    if "source_type" in params:
        source = _source(record)
        if source is None or params["source_type"].lower() not in {
                t.lower() for t in source.types}:
            return False
    if "source_designation" in params:
        source = _source(record)
        if source is None or not has_designation(source, params["source_designation"]):
            return False
    if "sorcery_speed" in params and sorcery_speed(candidate, record, lines) is not bool(
            params["sorcery_speed"]):
        return False
    if "phase_list_first" in params or "phase_list_last" in params:
        listed = _phase_list(candidate, lines)
        if listed is None:
            return False
        if "phase_list_first" in params and listed[0] != params["phase_list_first"]:
            return False
        if "phase_list_last" in params and listed[-1] != params["phase_list_last"]:
            return False
    if "candidate_face" in params:
        if not candidate.ability:
            return False
        alternative = candidate.ability[0].face != 0
        if alternative != (params["candidate_face"] == "alternative"):
            return False
    if "keyword" in params and not is_keyword_line(
            line_of(candidate.ability, lines), params["keyword"]):
        return False
    if "source_keyword" in params and not card_has_keyword(
            candidate.ability, lines, params["source_keyword"]):
        return False
    if "script_param" in params and not script_params_match(
            root_params_of(line_of(candidate.ability, lines)), params["script_param"]):
        return False
    if "cost_symbol" in params or "cost_part" in params:
        symbols, parts = cost_of(candidate, lines)
        if "cost_symbol" in params and (
                symbols is None or not _holds_symbol(symbols, params["cost_symbol"])):
            return False
        if "cost_part" in params and (
                parts is None or params["cost_part"].upper() not in parts):
            return False
    return True


def _bit(candidate, bit: str, check: str | None) -> bool | None:
    if bit == "failed_check":
        if candidate.failed_checks is None:
            return None
        return not any(failed.check == check for failed in candidate.failed_checks)
    return getattr(candidate, bit)


@dataclass(frozen=True, slots=True)
class VerdictBit:
    id: str = "verdict_bit"
    params: frozenset[str] = frozenset(
        {"bit", "check", "fires_when", "responsible_static"}) | FILTERS
    general: frozenset[str] = frozenset(
        {"bit", "check", "fires_when", "keyword", "source_keyword", "cost_symbol", "cost_part",
         "land_play", "candidate_face"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        bit = params.get("bit")
        needs = {"land_play"} if "land_play" in params else set()
        if "source_designation" in params:
            needs.add("designations")
        if bit == "timing":
            needs.add("timing")
        if bit == "failed_check":
            needs.add("failed_checks")
        return frozenset(needs)

    def validate(self, params: Mapping[str, Any]) -> None:
        bit = params.get("bit")
        if bit not in GEN2_BITS | GEN3_BITS:
            raise ValueError(f"verdict_bit: unknown bit {bit!r}")
        if bit == "failed_check" and params.get("check") not in CHECKS:
            raise ValueError(f"verdict_bit: a failed_check declaration names its check, "
                             f"one of {sorted(CHECKS)}; got {params.get('check')!r}")
        if params.get("candidate_face", "front") not in CANDIDATE_FACES:
            raise ValueError(f"verdict_bit: candidate_face is one of {sorted(CANDIDATE_FACES)}")
        if params.get("candidate", "cast") not in CANDIDATE_KINDS:
            raise ValueError(f"verdict_bit: candidate is one of {sorted(CANDIDATE_KINDS)}")
        dv.check(self.id, params, {
            "fires_when": dv.boolean, "responsible_static": dv.boolean,
            "land_play": dv.boolean, "sorcery_speed": dv.boolean, "keyword": dv.keyword,
            "source_keyword": dv.keyword, "cost_symbol": dv.cost_symbol,
            "script_param": dv.script_param, "source_type": dv.card_type,
            "source_designation": dv.designation,
            "phase_list_first": dv.one_of(*PHASE_ORDER),
            "phase_list_last": dv.one_of(*PHASE_ORDER)})
        if "check" in params and bit != "failed_check":
            raise ValueError("verdict_bit: check names a failed_check declaration's check")

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        bit = params.get("bit")
        if record.kind is not RecordKind.PLAYABILITY or (
                record.subkind is not PlayabilitySubkind.DECISION):
            return None
        # A failed_check "bit" reads true where the check passed, so the
        # default false is a candidate the named check failed.
        fires_when = bool(params.get("fires_when", False))
        candidates = [
            c for c in record.payload.candidates
            if (not params.get("responsible_static") or c.responsible_static)
            and _passes(c, lines, params, record)]
        values = [_bit(c, bit, params.get("check")) for c in candidates]
        values = [v for v in values if v is not None]
        if not values:
            return None
        fires = any(v is fires_when for v in values)
        polarity = Polarity.FIRES if fires else Polarity.DOES_NOT_FIRE
        source = record.state.refs.source
        return FamilyMatch(polarity, fires, (source,) if source else (),
                           target_legality_subjects(record))


FAMILY = register(VerdictBit())
