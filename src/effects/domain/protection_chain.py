"""What the acting chain and the board say about preventing damage, read from
the sidecars: the records a damage-prevented copy refuses (702.16e).

Torch-free: ``build-corpus``'s survey and write workers call it.
"""

from __future__ import annotations

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord
from effects.domain.script_text import api_of, params, root, segments
from effects.domain.state_snapshot import LineKind, Zone


def refuses_damage(record: EffectRecord, lines: SidecarLookup) -> bool:
    """Whether the chain or the board makes a prevention show more than the
    copy does."""
    return _unpreventable_or_dependent(record, lines) or no_prevention_on_the_board(record, lines)


#: The damage APIs whose amount the prevention changes.
_DAMAGE_APIS = frozenset({"DealDamage", "DamageAll", "EachDamage"})
#: Chain parameters after which a clause reads what the damage did.
_DEPENDENT_PREFIXES = ("Remember", "Imprint")
_DEPENDENT_APIS = frozenset({"StoreSVar"})


def _unpreventable_or_dependent(record: EffectRecord, lines: SidecarLookup) -> bool:
    """Whether the acting chain says its damage can't be prevented
    (``NoPrevention$``), or holds a clause the prevention would change
    without the copy showing it: one remembering what was dealt, storing an
    amount, or reading an amount from a variable (a drain's ``LifeAmount$ Y``,
    ``Y`` the damage dealt this way). Conservative: a variable amount in any
    later clause refuses the record. An acting line the sidecars cannot show
    refuses it too."""
    keys = record.ability or ()
    line = lines.first_line(keys)
    if line is None or not line.script_text:
        return bool(keys)
    return any(_clause_refuses(params(segment)) for segment in segments(line.script_text))


def _clause_refuses(found: dict[str, str]) -> bool:
    """Whether one clause of the acting chain refuses the record."""
    if "NoPrevention" in found:
        return True
    api = api_of(found)
    if api in _DAMAGE_APIS:
        return False
    return _reads_the_damage(api, found) or _has_variable_amount(found)


def _reads_the_damage(api: str | None, found: dict[str, str]) -> bool:
    return api in _DEPENDENT_APIS or any(k.startswith(_DEPENDENT_PREFIXES) for k in found)


def _has_variable_amount(found: dict[str, str]) -> bool:
    return any(_is_amount(key) and not _literal(value) for key, value in found.items())


def _is_amount(key: str) -> bool:
    return key.endswith(("Amount", "Num")) or key.startswith("Num") or key == "CounterNum"


def _literal(value: str) -> bool:
    return value.strip().lstrip("+-").isdigit()


def no_prevention_on_the_board(record: EffectRecord, lines: SidecarLookup) -> bool:
    """Whether a static on the board says damage can't be prevented: an
    active ``Mode$ CantPreventDamage`` line of a permanent or command-zone
    object."""
    keys = (key for entity in record.state.entities
            if entity.on_battlefield or entity.zone == Zone.COMMAND
            for card_line in entity.lines
            if card_line.kind == LineKind.STATIC and card_line.active is not False
            for key in card_line.keys)
    return any(_says_cant_prevent(lines.line_or_none(key)) for key in keys)


def _says_cant_prevent(found) -> bool:
    return (found is not None and bool(found.script_text)
            and params(root(found.script_text)).get("Mode") == "CantPreventDamage")
