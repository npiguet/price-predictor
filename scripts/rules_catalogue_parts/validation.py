"""``validate``: every sidecar against the rules index and Forge's cardsfolder."""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass

from .heads_check import validate_heads
from .ordering import rule_sort_key, unit_sort_key
from .reading import card_names, card_scripts, load_rules, load_unit
from .scenario_fields import validate_gen3_fields
from .script_edits import validate_script_edits
from .triage import validate_triage, wanted_rules
from .vocabulary import POLARITIES, RECORD_KINDS


def detector_families():
    """``family id -> declared params`` from the package's registry."""
    from effects.domain.rule_detectors import families

    return {family_id: family.params for family_id, family in families().items()}


@dataclass(frozen=True)
class ScenarioChecks:
    """What ``validate_scenarios`` checks a scenario file against, beside its triage.

    ``families`` (the detector registry) and ``scripts`` (a card's Forge script lines by name)
    are None where the gen-3 and script-edit checks do not run.
    """

    all_rule_ids: object
    names: object
    families: object = None
    scripts: object = None


def _variant_set_issues(sid, variants):
    """``(errors, warnings)`` of a scenario's variant count and polarities."""
    errs, warns = [], []
    if len(variants) < 2:
        errs.append(f"{sid}: fewer than two variants")
    pols = {v.get("polarity") for v in variants}
    if pols - POLARITIES:
        errs.append(f"{sid}: polarity {sorted(pols - POLARITIES, key=str)}")
    if "fires" not in pols or "does_not_fire" not in pols:
        warns.append(f"{sid}: no fires/does_not_fire pair")
    return errs, warns


def _gen3_issues(s, checks):
    """``(errors, warnings)`` of the gen-3 and script-edit checks, where they run."""
    errs, warns = [], []
    if checks.families is not None:
        errs += validate_gen3_fields(s, checks.families)
    if checks.scripts is not None:
        e2, w2 = validate_script_edits(s, checks.scripts)
        errs += e2
        warns += w2
    return errs, warns


def _scenario_errors(s, checks):
    """``(errors, warnings)`` of one scenario."""
    sid = s.get("id")
    errs = [f"{sid}: unknown rule {rid}" for rid in s.get("rules", [])
            if rid not in checks.all_rule_ids]
    if s.get("record_kind") not in RECORD_KINDS:
        errs.append(f"{sid}: record_kind {s.get('record_kind')!r}")
    e2, warns = _variant_set_issues(sid, s.get("variants", []))
    errs += e2
    errs += [f"{sid}: card not in Forge cardsfolder: {c!r}"
             for c in s.get("cards", []) if c not in checks.names]
    e3, w3 = _gen3_issues(s, checks)
    return errs + e3, warns + w3


def validate_scenarios(doc, triage, checks):
    errs, warns = [], []
    scen = doc.get("scenarios", [])
    covered = {rid for s in scen for rid in s.get("rules", [])}
    skipped = {s["id"] for s in doc.get("skipped", [])}
    for rid in sorted(wanted_rules(triage) - covered - skipped, key=rule_sort_key):
        errs.append(f"{rid}: neither covered by a scenario nor skipped")
    for s in scen:
        e2, w2 = _scenario_errors(s, checks)
        errs += e2
        warns += w2
    return errs, warns


class _AnyName:
    def __contains__(self, _):
        return True


def _triage_errors(unit, unit_rule_ids):
    """``(triage, errors)``: the unit's triage lines (None while unwritten) and what is wrong."""
    try:
        tri = load_unit("triage", unit)
    except ValueError as e:
        return None, [str(e)]
    return tri, ([] if tri is None else validate_triage(tri, unit_rule_ids))


def _scenarios_errors(unit, tri, checks):
    """``(errors, warnings)`` of the unit's scenario file, which needs its triage."""
    try:
        doc = load_unit("scenarios", unit)
    except json.JSONDecodeError as e:
        return [f"scenarios json: {e}"], []
    if doc is None or tri is None:
        return [], []
    return validate_scenarios(doc, tri, checks)


def _heads_errors(unit, tri):
    if tri is None or not wanted_rules(tri):
        return []
    try:
        heads = load_unit("heads", unit)
    except ValueError as e:
        return [str(e)]
    return ["no heads file"] if heads is None else validate_heads(heads, tri)


def _unit_report(unit, unit_rule_ids, checks, with_heads):
    """``(errors, warnings)`` of one unit, in the order they print."""
    tri, errs = _triage_errors(unit, unit_rule_ids)
    e2, warns = _scenarios_errors(unit, tri, checks)
    errs += e2
    if with_heads:
        errs += _heads_errors(unit, tri)
    return errs, warns


def _scenario_checks(all_ids, no_cards):
    names = None if no_cards else card_names()
    scripts = None if no_cards else card_scripts()
    return ScenarioChecks(all_ids, _AnyName() if names is None else names,
                          detector_families(), scripts)


def _print_report(unit, errs, warns):
    for e in errs:
        print(f"ERROR {unit}: {e}")
    for w in warns:
        print(f"warn  {unit}: {w}")


def cmd_validate(args):
    rules = load_rules()
    ids_by_unit = defaultdict(list)
    for r in rules:
        ids_by_unit[r["unit"]].append(r["id"])
    checks = _scenario_checks({r["id"] for r in rules}, args.no_cards)
    units = args.units or sorted(ids_by_unit, key=unit_sort_key)
    n_err = 0
    for unit in units:
        errs, warns = _unit_report(unit, ids_by_unit[unit], checks, args.heads)
        _print_report(unit, errs, warns if args.warnings else [])
        n_err += len(errs)
    print(f"{n_err} errors")
    sys.exit(1 if n_err else 0)
