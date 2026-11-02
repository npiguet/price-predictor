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
- The game-end rescue record (:func:`is_rescued_orphans`), whose board is
  read after its events rather than before them.
- The acting ability's source entity (:func:`source_entities`): the record's
  ``refs.source``, else, on a record that names none (a ``continuous``
  record), the battlefield entities printed from the acting script file or,
  for a token, made from it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey, SidecarLine, SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.script_text import params, root
from effects.domain.state_snapshot import EntityState, StateSnapshot

#: A :func:`script_params_match` value meaning "present, whatever its value".
ANY = "*"


def is_rescued_orphans(record: EffectRecord) -> bool:
    """Whether ``record`` is the collector's game-end rescue of events no
    bracket held: an effect half naming no acting line and no cost half.

    Its events span whatever stretch of the game no resolution was open
    over, and its snapshot is the board at the game's end, after them, with
    the loser already gone, so no family reading the board as it stood
    before the events can read one.
    """
    return (record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION
            and not record.ability and record.link_id is None)


def line_of(keys: Iterable[ProvenanceKey], lines: SidecarLookup | None) -> SidecarLine | None:
    """The first line ``keys`` name, or None where no line is readable."""
    keys = tuple(keys)
    if lines is None or not keys:
        return None
    return lines.first_line(keys)


def acting_line(record: EffectRecord, lines: SidecarLookup | None) -> SidecarLine | None:
    """The record's acting line, or None where no line is readable."""
    return line_of(record.ability or (), lines)


def root_params_of(line: SidecarLine | None) -> dict[str, str] | None:
    """A line's root-segment parameters, or None where it has no script."""
    if line is None or not line.script_text:
        return None
    return params(root(line.script_text))


def acting_root_params(record: EffectRecord, lines: SidecarLookup | None) -> dict[str, str] | None:
    """The acting line's root-segment parameters, or None where no line is readable."""
    return root_params_of(acting_line(record, lines))


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
    for name, value in wanted.items():
        if name not in found:
            return False
        if value != ANY and str(found[name]).lower() != str(value).lower():
            return False
    return True


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
    return tuple(e for e in state.entities if e.on_battlefield and (
        any(key.script_file in files for key in e.printed)
        or (e.token_script_id is not None and e.token_script_id in stems)))


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
