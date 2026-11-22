"""``validate``: every sidecar against the rules index and Forge's cardsfolder."""

import json
import sys
from collections import defaultdict

from .gen3_fields import GEN3_FIELDS, GEN3_KEYS, gen3_ids
from .ordering import rule_sort_key, unit_sort_key
from .reading import card_names, card_scripts, load_rules, load_unit
from .triage import validate_triage, wanted_rules
from .vocabulary import (
    DEPARTURE_RESOLUTION,
    DEPARTURE_STATUS,
    FORK_OPERATIONS,
    HEAD_OUTPUT_PREFIXES,
    HEAD_OUTPUTS,
    INPUT_STATUS,
    INPUT_TAGS,
    OUTPUT_STATUS,
    OUTPUT_TAGS,
    POLARITIES,
    RECORD_KINDS,
    SCRIPT_EDIT_OPERATIONS,
)


def validate_heads(rows, triage):
    errs = []
    seen = [r.get("id") for r in rows]
    for rid in sorted(wanted_rules(triage) - set(seen), key=rule_sort_key):
        errs.append(f"{rid}: no heads line")
    for rid in sorted(set(seen) - wanted_rules(triage)):
        errs.append(f"{rid}: heads line for a rule that needs none")
    for r in rows:
        rid = r.get("id")
        status, outputs = r.get("output"), r.get("outputs") or []
        if status not in OUTPUT_STATUS:
            errs.append(f"{rid}: output {status!r}")
        if not outputs:
            errs.append(f"{rid}: no outputs named")
        for o in outputs:
            if not (o in HEAD_OUTPUTS or o.startswith(HEAD_OUTPUT_PREFIXES + OUTPUT_TAGS)):
                errs.append(f"{rid}: unknown output {o!r}")
        names_existing = any(
            o in HEAD_OUTPUTS or o.startswith(HEAD_OUTPUT_PREFIXES) for o in outputs)
        if status == "trained" and not names_existing:
            errs.append(f"{rid}: trained but names no existing output")
        if r.get("input") not in INPUT_STATUS:
            errs.append(f"{rid}: input {r.get('input')!r}")
        missing = r.get("inputs_missing") or []
        if (r.get("input") == "incomplete") != bool(missing):
            errs.append(f"{rid}: input {r.get('input')!r} disagrees with inputs_missing")
        for i in missing:
            if not i.startswith(INPUT_TAGS):
                errs.append(f"{rid}: untagged missing input {i!r}")
        if GEN3_FIELDS.exists():
            if any(k not in r for k in GEN3_KEYS):
                errs.append(f"{rid}: gen3 fields missing; run gen3-fields")
            else:
                ids = set(gen3_ids())
                for k in ("gen3_inputs", "gen3_outputs", "gen3_fixes"):
                    for i in r[k]:
                        if i not in ids:
                            errs.append(f"{rid}: unknown {k} id {i!r}")
    return errs


def _names_a_card(value):
    """Whether a detector parameter's value names one card (a ``Card.namedX`` selector)."""
    return isinstance(value, str) and "named" in value.lower()


def validate_gen3_fields(s, families):
    """The detector, operation and departure fields of one scenario (FR-002, FR-003, FR-005).

    ``families`` is the detector registry, ``family id -> declared params``.
    """
    errs = []
    sid = s.get("id")
    detector = s.get("detector")
    if not isinstance(detector, dict) or not detector:
        errs.append(f"{sid}: no detector (declare one, or {{\"none\": reason}})")
    elif "none" in detector:
        if len(detector) != 1 or not str(detector["none"]).strip():
            errs.append(f"{sid}: a none detector carries its reason and nothing else")
    elif "same_as" in detector:
        if set(detector) != {"same_as", "reason"} or not str(detector["reason"]).strip():
            errs.append(f"{sid}: a same_as detector names its scenario and the fact the two "
                        "share (reason), and nothing else")
        elif sid in (detector["same_as"] if isinstance(detector["same_as"], list)
                     else [detector["same_as"]]):
            errs.append(f"{sid}: a scenario is no duplicate of itself")
    else:
        family = detector.get("family")
        if family not in families:
            errs.append(f"{sid}: unknown detector family {family!r}")
        else:
            undeclared = set(detector.get("params") or {}) - families[family]
            if undeclared:
                errs.append(f"{sid}: {family} declares no parameter {sorted(undeclared)}")
        if detector.get("record_kind") != s.get("record_kind"):
            errs.append(f"{sid}: detector record_kind {detector.get('record_kind')!r} is not "
                        f"the scenario's {s.get('record_kind')!r}")
        for name, value in (detector.get("params") or {}).items():
            if _names_a_card(value):
                errs.append(f"{sid}: detector parameter {name} names a card ({value!r}); "
                            "describe the rule's condition, not the scenario's board")
        for key in ("outcome", "near_miss"):
            if not str(detector.get(key) or "").strip():
                errs.append(f"{sid}: detector has no {key}")
    variants = s.get("variants", [])
    if variants:
        base = variants[0]
        if base.get("delta"):
            errs.append(f"{sid}: the first variant is the base and its delta is not empty")
        if "operation" in base:
            errs.append(f"{sid}: the base variant carries an operation")
    for v in variants[1:]:
        op = v.get("operation")
        label = f"{sid}/{v.get('label')}"
        if op is None:
            errs.append(f"{label}: no operation (one of {sorted(map(str, FORK_OPERATIONS))}, "
                        "or {\"unclassifiable\": reason})")
        elif isinstance(op, dict):
            if set(op) != {"unclassifiable"} or not str(op["unclassifiable"]).strip():
                errs.append(f"{label}: operation {op!r}")
        elif op not in FORK_OPERATIONS:
            errs.append(f"{label}: unknown operation {op!r}")
    departure = s.get("departure")
    if departure is not None:
        status, resolution = departure.get("status"), departure.get("resolution")
        if status not in DEPARTURE_STATUS:
            errs.append(f"{sid}: departure status {status!r}")
        if resolution is not None and resolution not in DEPARTURE_RESOLUTION:
            errs.append(f"{sid}: departure resolution {resolution!r}")
        if resolution is not None and status != "confirmed":
            errs.append(f"{sid}: a departure has a resolution only once confirmed")
    return errs


_SCRIPT_EDIT_MEMBERS = {"card", "remove", "add", "note"}


def _script_edit_errors(label, edit, scripts):
    """What keeps one ``script_edit`` from applying to its card's Forge script (FR-058f)."""
    if not isinstance(edit, dict):
        return [f"{label}: script_edit is not an object"]
    errs = []
    unknown = set(edit) - _SCRIPT_EDIT_MEMBERS
    if unknown:
        errs.append(f"{label}: script_edit has unknown member(s) {sorted(unknown)}")
    lists = [edit.get(key, []) for key in ("remove", "add")]
    if not all(isinstance(lines, list) and all(isinstance(x, str) for x in lines)
               for lines in lists):
        return errs + [f"{label}: script_edit remove and add are lists of script lines"]
    if not any(lines for lines in lists):
        errs.append(f"{label}: script_edit changes nothing")
    card = edit.get("card")
    if not isinstance(card, str) or not card.strip():
        return errs + [f"{label}: script_edit names no card"]
    script = scripts(card)
    if script is None:
        return errs + [f"{label}: script_edit card {card!r} has no Forge script"]
    present = {line.strip() for line in script}
    for line in edit.get("remove", []):
        if line.strip() not in present:
            errs.append(f"{label}: script_edit removes {line!r}, not in its Forge script")
    return errs


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
        elif isinstance(op := v.get("operation"), str) and op in SCRIPT_EDIT_OPERATIONS:
            warns.append(f"{label}: {v['operation']} variant has no script_edit")
    return errs, warns


def detector_families():
    """``family id -> declared params`` from the package's registry."""
    from effects.domain.rule_detectors import families

    return {family_id: family.params for family_id, family in families().items()}


def validate_scenarios(doc, triage, all_rule_ids, names, families=None, scripts=None):
    errs, warns = [], []
    scen = doc.get("scenarios", [])
    covered = {rid for s in scen for rid in s.get("rules", [])}
    skipped = {s["id"] for s in doc.get("skipped", [])}
    wanted = wanted_rules(triage)
    for rid in sorted(wanted - covered - skipped, key=rule_sort_key):
        errs.append(f"{rid}: neither covered by a scenario nor skipped")
    for s in scen:
        sid = s.get("id")
        for rid in s.get("rules", []):
            if rid not in all_rule_ids:
                errs.append(f"{sid}: unknown rule {rid}")
        if s.get("record_kind") not in RECORD_KINDS:
            errs.append(f"{sid}: record_kind {s.get('record_kind')!r}")
        variants = s.get("variants", [])
        if len(variants) < 2:
            errs.append(f"{sid}: fewer than two variants")
        pols = {v.get("polarity") for v in variants}
        if pols - POLARITIES:
            errs.append(f"{sid}: polarity {sorted(pols - POLARITIES, key=str)}")
        if "fires" not in pols or "does_not_fire" not in pols:
            warns.append(f"{sid}: no fires/does_not_fire pair")
        for c in s.get("cards", []):
            if c not in names:
                errs.append(f"{sid}: card not in Forge cardsfolder: {c!r}")
        if families is not None:
            errs += validate_gen3_fields(s, families)
        if scripts is not None:
            e2, w2 = validate_script_edits(s, scripts)
            errs += e2
            warns += w2
    return errs, warns


def cmd_validate(args):
    rules = load_rules()
    all_ids = {r["id"] for r in rules}
    ids_by_unit = defaultdict(list)
    for r in rules:
        ids_by_unit[r["unit"]].append(r["id"])
    names = card_names() if not args.no_cards else None
    scripts = card_scripts() if not args.no_cards else None
    families = detector_families()
    units = args.units or sorted(ids_by_unit, key=unit_sort_key)
    n_err = 0
    for unit in units:
        errs, warns = [], []
        try:
            tri = load_unit("triage", unit)
        except ValueError as e:
            tri, errs = None, [str(e)]
        if tri is not None:
            errs += validate_triage(tri, ids_by_unit[unit])
        try:
            doc = load_unit("scenarios", unit)
        except json.JSONDecodeError as e:
            doc, errs = None, errs + [f"scenarios json: {e}"]
        if doc is not None and tri is not None:
            known = names if names is not None else _AnyName()
            e2, w2 = validate_scenarios(doc, tri, all_ids, known, families, scripts)
            errs += e2
            warns += w2
        if args.heads and tri is not None and wanted_rules(tri):
            try:
                heads = load_unit("heads", unit)
            except ValueError as e:
                heads, errs = None, errs + [str(e)]
            errs += ["no heads file"] if heads is None else validate_heads(heads, tri)
        for e in errs:
            print(f"ERROR {unit}: {e}")
        if args.warnings:
            for w in warns:
                print(f"warn  {unit}: {w}")
        n_err += len(errs)
    print(f"{n_err} errors")
    sys.exit(1 if n_err else 0)


class _AnyName:
    def __contains__(self, _):
        return True
