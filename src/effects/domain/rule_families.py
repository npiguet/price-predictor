"""Which rules category a record belongs to (FR-047, FR-047a, FR-048).

Curation balances its budget across these families (FR-049), so a rule that
appears on a handful of cards trains as often as one printed on thousands: a
class budget split by record count would spend nearly all of itself on
``DealDamage`` and ``ChangesZone`` and leave ``Mill`` or ``Phases`` to chance.

A family is a pure function of the record and the sidecars its keys resolve
through, so the survey pass and the write pass reach the same answer for the
same record without coordinating:

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

Prior art: none for the classification itself; ``sampling_class`` in
``train_effect_model`` is the sibling one level up, and the combat case reuses
gate 2's :class:`KeywordResolver`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from effects.domain.ability_tokenizer import display_name_of
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME, KeywordResolver
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    Candidate,
    EffectRecord,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
)

#: The family of a record that has none of its own: a combat with no
#: damage-step keyword on the board, a legality record no static restricted, a
#: decision whose every verdict bit is true.
NONE = "none"
#: A trigger whose script names no ``Mode$`` — 244 of 3,081 triggered lines.
NO_MODE = "(no-mode)"
#: An acting key no sidecar line answers for: an unconverted script, a dropped
#: or runtime-only trait. Kept as a family rather than dropped, because the
#: build keeps the record.
NO_TEXT = "(no-text)"
#: The ``decision`` families of FR-047a, in the order the verdict bits are read.
CANNOT_PLAY = "cannot-play"
UNAFFORDABLE = "unaffordable"
NO_LEGAL_TARGET = "no-legal-target"

#: ``script_api_type`` of a sidecar line that is a printed keyword.
KEYWORD_API_TYPE = "Keyword"
_SEGMENT_SEPARATOR = "[SEG]"


def root_segment(script_text: str) -> str:
    """The text before the first ``[SEG]``: the trait's own script line."""
    return script_text.split(_SEGMENT_SEPARATOR, 1)[0]


def root_param(script_text: str | None, name: str) -> str | None:
    """The value of ``name$`` on the root segment, or None when absent."""
    if not script_text:
        return None
    for part in root_segment(script_text).split("|"):
        key, dollar, value = part.partition("$")
        if dollar and key.strip() == name:
            return value.strip() or None
    return None


def _line(sidecars, keys: Iterable[ProvenanceKey]):
    """The first sidecar line any of ``keys`` resolves to, or None."""
    for key in keys:
        try:
            line = sidecars.line_for(key)
        except KeyError:
            line = None
        if line is not None:
            return line
    return None


def _keyword_family(line) -> str | None:
    if line is not None and line.script_api_type == KEYWORD_API_TYPE and line.script_text:
        return display_name_of(line.script_text)
    return None


def static_mode(keys: Iterable[ProvenanceKey], sidecars) -> str | None:
    """The ``Mode$`` of the static ``keys`` name, through the sidecar.

    A static's ``script_api_type`` is its ``Mode$`` (1,218 of 1,234 static
    lines), but the root segment is read first because the two part company on
    a merged line. A static folded into a keyword line (protection's
    ``CantBlockBy``, say) has no script of its own there, and answers with the
    keyword's display name, which is the family FR-048 gives it anyway.
    """
    line = _line(sidecars, keys)
    if line is None:
        return None
    keyword = _keyword_family(line)
    if keyword is not None:
        return keyword
    return root_param(line.script_text, "Mode") or line.script_api_type


def candidate_family(candidate: Candidate, sidecars) -> str:
    """One ``decision`` candidate's family (FR-047a).

    The responsible static's mode where one is named and resolves; otherwise
    the first false verdict bit, read in the order ``can_play``,
    ``affordable``, ``has_legal_target``; otherwise ``none``.
    """
    if candidate.responsible_static:
        mode = static_mode(candidate.responsible_static, sidecars)
        if mode is not None:
            return mode
    if not candidate.can_play:
        return CANNOT_PLAY
    if not candidate.affordable:
        return UNAFFORDABLE
    if not candidate.has_legal_target:
        return NO_LEGAL_TARGET
    return NONE


def legality_modes(record: EffectRecord, sidecars) -> frozenset[str]:
    """Every single static mode restricting a creature on a legality record.

    A static's ``Mode$`` may list several modes — Pacifism's is
    ``CantAttack,CantBlock`` — and each is a rule of its own, so the list is
    split. A keyword family (``Protection from black``) is taken whole, since
    its name is not a mode list.
    """
    modes: set[str] = set()
    for entry in record.payload.forbidden:
        if not entry.responsible_static:
            continue
        line = _line(sidecars, entry.responsible_static)
        if line is None:
            continue
        keyword = _keyword_family(line)
        if keyword is not None:
            modes.add(keyword)
            continue
        declared = root_param(line.script_text, "Mode") or line.script_api_type
        if declared:
            modes.update(part.strip() for part in declared.split(",") if part.strip())
    return frozenset(modes)


def legality_family(
    modes: frozenset[str], mode_counts: Mapping[str, int] | None = None,
) -> str:
    """The family of a legality record restricted by ``modes``.

    The rarest of them by the corpus-wide ``mode_counts``, so a board stacking
    several restrictions counts toward the rule the corpus holds least of,
    rather than founding a family of its own for the combination. Ties and a
    missing table (a dataset built before the counts existed) fall to the
    first mode by name, which keeps the family a single rule either way.
    """
    if not modes:
        return NONE
    counts = mode_counts or {}
    return min(modes, key=lambda mode: (counts.get(mode, 0), mode))


def combat_family(record: EffectRecord, resolver: KeywordResolver) -> str:
    """The sorted set of damage-step keywords the participants carry."""
    found: set[str] = set()
    for entity in record.state.entities:
        if entity.combat is None:
            continue
        found |= resolver.keywords_of(entity) & KEYWORDS_BY_NAME.keys()
    return ",".join(sorted(found)) if found else NONE


def rule_family(
    record: EffectRecord, sidecars, *, resolver: KeywordResolver | None = None,
    legality_mode_counts: Mapping[str, int] | None = None,
) -> str:
    """The rules category ``record`` belongs to (module docstring).

    ``sidecars`` answers ``line_for(key)``; ``resolver`` is the combat
    keyword resolver, passed in so a caller walking many records shares its
    memo, and built over ``sidecars`` when omitted. ``legality_mode_counts``
    is the dataset's corpus-wide count of records per legality mode, which
    picks a legality record's rarest mode; every caller placing records of
    one dataset passes the same table, from its manifest.

    A ``decision`` record's family is its first candidate's: Java writes one
    candidate per record, which is the unit feature 023 already trains on.
    """
    match record.kind:
        case RecordKind.COMBAT:
            return combat_family(record, resolver or KeywordResolver(sidecars))
        case RecordKind.PLAYABILITY:
            if record.subkind is PlayabilitySubkind.DECISION:
                payload: PlayabilityDecisionPayload = record.payload
                if not payload.candidates:
                    return NONE
                return candidate_family(payload.candidates[0], sidecars)
            return legality_family(
                legality_modes(record, sidecars), legality_mode_counts,
            )

    line = _line(sidecars, record.ability or ())
    if line is None:
        return NO_TEXT
    keyword = _keyword_family(line)
    if keyword is not None:
        return keyword
    match record.kind:
        case RecordKind.TRIGGER:
            return root_param(line.script_text, "Mode") or NO_MODE
        case RecordKind.REWRITE:
            return root_param(line.script_text, "Event") or NO_TEXT
    return line.script_api_type or NO_TEXT
