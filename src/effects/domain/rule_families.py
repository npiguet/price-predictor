"""Which rules category a record belongs to (FR-047, FR-047a, FR-048).

Curation balances its budget across these families (FR-049). A family is a
pure function of the record and its sidecars, so the survey and write passes
agree without coordinating:

==============  ===========================================================
kind            family
==============  ===========================================================
resolution      the acting line's ``script_api_type`` (a mode's own on an
                ``option`` line, FR-005a)
trigger         ``Mode$`` of the trigger line's root segment, ``(no-mode)``
                when the script names none
continuous      the acting static line's ``script_api_type`` (its ``Mode$``)
rewrite         ``Event$`` of the replacement line's root segment
combat          the sorted set of damage-step keywords on the participants,
                ``none`` when they carry none
playability     FR-047a: the responsible static's ``Mode$``, else the first
                failing verdict bit (``decision``), or the rarest single
                static mode across ``forbidden`` (``attackers``/``blockers``)
==============  ===========================================================

Whatever the kind, a record acting through a keyword line belongs to that
keyword's family (FR-048): a ward trigger and a ward resolution both teach ward.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

from effects.domain.ability_tokenizer import display_name_of
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME, KeywordResolver
from effects.domain.provenance import ProvenanceKey, SidecarLookup
from effects.domain.records import (
    Candidate,
    EffectRecord,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
)
from effects.domain.script_text import root_param


class FamilySentinel(StrEnum):
    """The families a record takes where its script names no rule of its own.

    A string enum, so a sentinel keys the manifest's family tables exactly as
    the bare string does.
    """

    NONE = "none"
    NO_MODE = "(no-mode)"
    #: An acting key no sidecar line answers for; the build keeps the record.
    NO_TEXT = "(no-text)"
    #: The ``decision`` families of FR-047a, in verdict-bit order.
    CANNOT_PLAY = "cannot-play"
    UNAFFORDABLE = "unaffordable"
    NO_LEGAL_TARGET = "no-legal-target"


KEYWORD_API_TYPE = "Keyword"


def _is_keyword_line(line) -> bool:
    return line.script_api_type == KEYWORD_API_TYPE and bool(line.script_text)


def _keyword_family(line) -> str | None:
    if line is not None and _is_keyword_line(line):
        return display_name_of(line.script_text)
    return None


def _declared_mode(line) -> str | None:
    """The mode a static line declares: its keyword family, else ``Mode$``.

    The root segment is read before ``script_api_type`` because the two part
    company on a merged line.
    """
    keyword = _keyword_family(line)
    if keyword is not None:
        return keyword
    return root_param(line.script_text, "Mode") or line.script_api_type


def static_mode(keys: Iterable[ProvenanceKey], sidecars: SidecarLookup) -> str | None:
    line = sidecars.first_line(keys)
    return None if line is None else _declared_mode(line)


def candidate_family(candidate: Candidate, sidecars: SidecarLookup) -> str:
    """One ``decision`` candidate's family (FR-047a): the responsible static's
    mode, else the first false verdict bit, else ``none``."""
    if candidate.responsible_static:
        mode = static_mode(candidate.responsible_static, sidecars)
        if mode is not None:
            return mode
    if not candidate.can_play:
        return FamilySentinel.CANNOT_PLAY
    if not candidate.affordable:
        return FamilySentinel.UNAFFORDABLE
    if not candidate.has_legal_target:
        return FamilySentinel.NO_LEGAL_TARGET
    return FamilySentinel.NONE


def legality_modes(record: EffectRecord, sidecars: SidecarLookup) -> frozenset[str]:
    """Every single static mode restricting a creature on a legality record.

    A ``Mode$`` list (Pacifism's ``CantAttack,CantBlock``) is split; a keyword
    family is taken whole, since its name is not a mode list.
    """
    modes: set[str] = set()
    for entry in record.payload.forbidden:
        if not entry.responsible_static:
            continue
        line = sidecars.first_line(entry.responsible_static)
        if line is None:
            continue
        declared = _declared_mode(line)
        if _is_keyword_line(line):
            modes.add(declared)
        elif declared:
            modes.update(part.strip() for part in declared.split(",") if part.strip())
    return frozenset(modes)


def legality_family(
    modes: frozenset[str], mode_counts: Mapping[str, int] | None = None,
) -> str:
    """The rarest of ``modes`` by the corpus-wide ``mode_counts``, so stacked
    restrictions never found a family of their own; ties and a missing table
    fall to the first mode by name."""
    if not modes:
        return FamilySentinel.NONE
    counts = mode_counts or {}
    return min(modes, key=lambda mode: (counts.get(mode, 0), mode))


def combat_family(record: EffectRecord, resolver: KeywordResolver) -> str:
    """The sorted set of damage-step keywords the participants carry."""
    found: set[str] = set()
    for entity in record.state.entities:
        if entity.combat is None:
            continue
        found |= resolver.keywords_of(entity) & KEYWORDS_BY_NAME.keys()
    return ",".join(sorted(found)) if found else FamilySentinel.NONE


def rule_family(
    record: EffectRecord, sidecars: SidecarLookup, *,
    resolver: KeywordResolver | None = None,
    legality_mode_counts: Mapping[str, int] | None = None,
) -> str:
    """The rules category ``record`` belongs to (module docstring).

    ``resolver`` is passed in so a caller walking many records shares its
    memo. Every caller placing one dataset's records passes the same
    ``legality_mode_counts``, from its manifest. A ``decision`` record's
    family is its first candidate's: Java writes one candidate per record.
    """
    match record.kind:
        case RecordKind.COMBAT:
            return combat_family(record, resolver or KeywordResolver(sidecars))
        case RecordKind.PLAYABILITY:
            if record.subkind is PlayabilitySubkind.DECISION:
                payload: PlayabilityDecisionPayload = record.payload
                if not payload.candidates:
                    return FamilySentinel.NONE
                return candidate_family(payload.candidates[0], sidecars)
            return legality_family(
                legality_modes(record, sidecars), legality_mode_counts,
            )

    line = sidecars.first_line(record.ability or ())
    if line is None:
        return FamilySentinel.NO_TEXT
    keyword = _keyword_family(line)
    if keyword is not None:
        return keyword
    match record.kind:
        case RecordKind.TRIGGER:
            return root_param(line.script_text, "Mode") or FamilySentinel.NO_MODE
        case RecordKind.REWRITE:
            return root_param(line.script_text, "Event") or FamilySentinel.NO_TEXT
    return line.script_api_type or FamilySentinel.NO_TEXT
