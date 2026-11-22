"""``validate``: whether each variant's ``script_edit`` applies to its card's Forge script."""

from .vocabulary import SCRIPT_EDIT_OPERATIONS

_SCRIPT_EDIT_MEMBERS = {"card", "remove", "add", "note"}


def _are_script_lines(lines):
    return isinstance(lines, list) and all(isinstance(x, str) for x in lines)


def _unknown_member_errors(label, edit):
    unknown = set(edit) - _SCRIPT_EDIT_MEMBERS
    if unknown:
        return [f"{label}: script_edit has unknown member(s) {sorted(unknown)}"]
    return []


def _shape_errors(label, edit):
    """``(errors, stop)``: what is wrong with the edit's own members; ``stop`` when its lines
    or its card cannot be read at all."""
    errs = _unknown_member_errors(label, edit)
    lists = [edit.get(key, []) for key in ("remove", "add")]
    if not all(_are_script_lines(lines) for lines in lists):
        return errs + [f"{label}: script_edit remove and add are lists of script lines"], True
    if not any(lines for lines in lists):
        errs.append(f"{label}: script_edit changes nothing")
    card = edit.get("card")
    if not isinstance(card, str) or not card.strip():
        return errs + [f"{label}: script_edit names no card"], True
    return errs, False


def _script_edit_errors(label, edit, scripts):
    """What keeps one ``script_edit`` from applying to its card's Forge script (FR-058f)."""
    if not isinstance(edit, dict):
        return [f"{label}: script_edit is not an object"]
    errs, stop = _shape_errors(label, edit)
    if stop:
        return errs
    card = edit["card"]
    script = scripts(card)
    if script is None:
        return errs + [f"{label}: script_edit card {card!r} has no Forge script"]
    present = {line.strip() for line in script}
    return errs + [f"{label}: script_edit removes {line!r}, not in its Forge script"
                   for line in edit.get("remove", []) if line.strip() not in present]


def _is_card_level(variant):
    op = variant.get("operation")
    return isinstance(op, str) and op in SCRIPT_EDIT_OPERATIONS


def validate_script_edits(s, scripts):
    """``(errors, warnings)`` of one scenario's variants' ``script_edit`` (FR-058f).

    ``scripts`` reads a card's Forge script lines by name (:func:`card_scripts`).
    A card-level variant (``SCRIPT_EDIT_OPERATIONS``) without one is a warning:
    a fact no card script can state, such as a player's counters, has none.
    """
    errs, warns = [], []
    for v in s.get("variants", [])[1:]:
        label = f"{s.get('id')}/{v.get('label')}"
        if "script_edit" in v:
            errs += _script_edit_errors(label, v["script_edit"], scripts)
        elif _is_card_level(v):
            warns.append(f"{label}: {v['operation']} variant has no script_edit")
    return errs, warns
