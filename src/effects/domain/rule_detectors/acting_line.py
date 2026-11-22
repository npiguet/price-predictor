"""What the detector families read off a record's lines and its source.

Not a family: the reads several families share, so a keyword or a script
parameter means one thing whichever family a declaration names.

- A keyword's own line. A trigger or an activated ability a keyword generates
  (exalted, equip, cycling, an etbCounter replacement) acts through the
  keyword's line, whose script is the keyword text (``Equip:2``) with no
  ``Mode$`` or API of its own; :func:`is_keyword_line` compares its display
  name in ``keyword_names``' spelling, so ``Cumulative upkeep`` and
  ``cumulative_upkeep`` agree.
- A card's printed keywords (:func:`card_has_keyword`): any line of the card's
  sidecar a keyword produced, the line acting or not, so a spell with convoke
  is found through the card while the spell acts through its own line.
- A script parameter (:func:`script_params_match`): a root-segment ``Key$``
  present, or holding a value.
- The acting ability's source entity (:func:`source_entities`): the record's
  ``refs.source``, else, on a record that names none (a ``continuous``
  record), the battlefield entities printed from the acting script file or,
  for a token, made from it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import LineKind, ProvenanceKey, SidecarLine, SidecarLookup
from effects.domain.records import Candidate, EffectRecord
from effects.domain.script_text import params, root
from effects.domain.state_snapshot import EntityState, StateSnapshot

#: A :func:`script_params_match` value meaning "present, whatever its value".
ANY = "*"


def line_of(keys: Iterable[ProvenanceKey], lines: SidecarLookup | None) -> SidecarLine | None:
    """The first line ``keys`` name, or None where no line is readable."""
    keys = tuple(keys)
    if lines is None or not keys:
        return None
    return lines.first_line(keys)


def acting_line(record: EffectRecord, lines: SidecarLookup | None) -> SidecarLine | None:
    """The record's acting line, or None where no line is readable.

    The line the join gives the first acting key that has one: a chapter's
    ``built`` key reads its own chapter line, whose script is that chapter's
    trigger; a keyword's bare key naming several lines (``AMBIGUOUS``) reads
    none, never the first of them.
    """
    return line_of(record.ability or (), lines)


def root_params_of(line: SidecarLine | None) -> dict[str, str] | None:
    """A line's root-segment parameters, or None where it has no script."""
    if line is None or not line.script_text:
        return None
    return params(root(line.script_text))


def acting_root_params(record: EffectRecord, lines: SidecarLookup | None) -> dict[str, str] | None:
    """The acting line's root-segment parameters, or None where no line is readable."""
    return root_params_of(acting_line(record, lines))


#: What an acting ability is: its card's cast, or an activated ability.
CANDIDATE_KINDS = frozenset({"cast", "activated"})


#: The keywords whose alternative-cost line is no cast: a special action
#: (suspend 702.62, foretell 702.143, plot 702.170) or an activated ability
#: from a hand or a graveyard (unearth, ninjutsu, encore, embalm, eternalize).
#: ``built.kind`` cannot tell them apart (every one is ``spell``), the line's
#: kind is the converter's ``alternate cost`` for all of them.
_NOT_A_CAST = frozenset(normalize_keyword(name) for name in (
    "Suspend", "Foretell", "Plot", "Unearth", "Ninjutsu", "Encore", "Embalm", "Eternalize"))


def _keyword_ability_kind(keys: tuple[ProvenanceKey, ...], lines: SidecarLookup | None) -> str:
    """A keyword's own line: a cast where the keyword's alternative cost casts
    the spell (awaken, bestow, flashback, harmonize, morph ...), else an
    activated ability (equip, cycling, crew ...), the answer too where no line
    is readable."""
    line = line_of(keys, lines)
    if line is None or line.line_kind != LineKind.ALTERNATE_COST:
        return "activated"
    name = normalize_keyword(line.keyword_display_name or "")
    return "activated" if name in _NOT_A_CAST else "cast"


def ability_kind(keys: Iterable[ProvenanceKey], lines: SidecarLookup | None) -> str | None:
    """``cast`` or ``activated``; None for no key, a static or trigger, or an
    unreadable key. A keyword's own line is a cast when it is an alternative
    cost that casts (:func:`_keyword_ability_kind`), else an activated ability."""
    keys = tuple(keys)
    if not keys:
        return None
    key = keys[0]
    if key.trait_kind == "keyword":
        return _keyword_ability_kind(keys, lines)
    if key.trait_kind != "spell":
        return None
    line = line_of(keys, lines)
    if line is None:
        # A permanent's implicit cast spell, which the converter drops.
        return "cast" if key.index_within_kind == 0 and key.option is None else None
    found = root_params_of(line) or {}
    return "cast" if "SP" in found else "activated" if "AB" in found else None


def candidate_kind(candidate: Candidate, lines: SidecarLookup | None) -> str | None:
    """:func:`ability_kind` of a playability candidate; None for a land play."""
    return None if candidate.land_play else ability_kind(candidate.ability, lines)


def apis_of(api: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """A declared ``api``'s APIs: the one named, or each of a list."""
    return (api,) if isinstance(api, str) else tuple(api)


def require_candidate(family: str, params: Mapping[str, Any]) -> None:
    """Refuse a ``candidate`` that is no kind, and a ``source_keyword`` without
    the ``candidate`` it speaks for: the keyword is the card's, so an Aura's
    activated or granted ability would be taken for its cast."""
    wanted = "source_keyword" in params or "candidate" in params
    if wanted and params.get("candidate") not in CANDIDATE_KINDS:
        raise ValueError(f"{family}: a source_keyword is the candidate card's, so the declaration "
                         f"names its candidate, one of {sorted(CANDIDATE_KINDS)}")


def same_keyword(name: str | None, keyword: str) -> bool:
    """Whether ``name`` spells ``keyword``, in the package's spelling."""
    return name is not None and normalize_keyword(name) == normalize_keyword(keyword)


def is_keyword_line(line: SidecarLine | None, keyword: str) -> bool:
    """Whether ``line`` is the line the keyword ``keyword`` produced."""
    return line is not None and same_keyword(line.keyword_display_name, keyword)


def card_has_keyword(keys: Iterable[ProvenanceKey], lines: SidecarLookup | None,
                     keyword: str) -> bool:
    """Whether the card the first of ``keys`` belongs to prints ``keyword``.

    Read off the card's sidecar, so a keyword only another card grants is not
    the card's; a card with no sidecar has none.
    """
    keys = tuple(keys)
    if lines is None or not keys:
        return False
    try:
        sidecar = lines.get(keys[0].script_file)
    except KeyError:
        return False
    return any(same_keyword(line.keyword_display_name, keyword) for line in sidecar.lines)


def script_params_match(found: Mapping[str, str] | None, wanted: Any) -> bool:
    """Whether a segment's parameters ``found`` hold ``wanted``.

    ``wanted`` is a parameter name, present whatever its value (``Monstrosity``
    for ``Monstrosity$ True``), or a mapping of names to values, each compared
    without case, :data:`ANY` for present at all.
    """
    if found is None:
        return False
    if isinstance(wanted, str):
        wanted = {wanted: ANY}
    if not isinstance(wanted, Mapping) or not wanted:
        raise ValueError(f"script_param {wanted!r}: a parameter name or a name -> value map")
    return all(_param_holds(found, name, value) for name, value in wanted.items())


def _param_holds(found: Mapping[str, str], name: str, value: Any) -> bool:
    """Whether ``found`` has ``name``, with ``value`` where it is not :data:`ANY`."""
    if name not in found:
        return False
    return value == ANY or str(found[name]).lower() == str(value).lower()


def source_entities(state: StateSnapshot,
                    keys: Iterable[ProvenanceKey] = ()) -> tuple[EntityState, ...]:
    """The acting ability's source: ``refs.source``'s entity, else the
    battlefield entities printed from the acting script file.

    A ``continuous`` record names no source and its snapshot drops the acting
    static from its host's printed lines, so the host is found by the script
    file its other lines share.
    """
    if state.refs.source is not None:
        entity = state.entity(state.refs.source)
        return (entity,) if entity is not None else ()
    files = {key.script_file for key in keys}
    if not files:
        return ()
    stems = {_stem(f) for f in files}
    return tuple(e for e in state.entities if e.on_battlefield and _printed_from(e, files, stems))


def _printed_from(entity: EntityState, files: set[str], stems: set[str]) -> bool:
    """Whether the entity prints a line of one of the script ``files``, or is
    a token made from one (``stems``)."""
    if any(key.script_file in files for key in entity.printed):
        return True
    return entity.token_script_id is not None and entity.token_script_id in stems


def source_ids(record: EffectRecord) -> tuple[str, ...]:
    """The ids of the acting ability's source (:func:`source_entities`): the
    condition entity a rule about the acting object names."""
    return tuple(e.id for e in source_entities(record.state, record.ability or ()))


def _stem(script_file: str) -> str:
    """``tokenscripts/role_monster.txt`` -> ``role_monster``: a token's
    ``token_script_id``, which a host printing no other line is found by."""
    name = script_file.rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".txt") else name


def source_is_token_of_kind(state: StateSnapshot, kind: str,
                            keys: Iterable[ProvenanceKey] = ()) -> bool:
    """Whether the acting ability's source is a token of the kind ``kind``
    (a Food's, a Clue's, a Monster Role's own ability).

    A kind is a subtype (``Food``) or, for a Role, a word of the token's name:
    every Role is an ``Aura Role``, told apart only by its name (rule 111.10j).
    """
    wanted = kind.lower()
    return any(entity.token_script_id is not None
               and wanted in {s.lower() for s in entity.subtypes} | set(
                   entity.name.lower().split())
               for entity in source_entities(state, keys))
