"""``validate --heads``: one unit's heads lines against its triage and the gen-3 ids."""

from .gen3_registry import GEN3_FIELDS, GEN3_KEYS, gen3_ids
from .ordering import rule_sort_key
from .triage import wanted_rules
from .vocabulary import (
    HEAD_OUTPUT_PREFIXES,
    HEAD_OUTPUTS,
    INPUT_STATUS,
    INPUT_TAGS,
    OUTPUT_STATUS,
    OUTPUT_TAGS,
)


def _names_existing_output(output):
    return output in HEAD_OUTPUTS or output.startswith(HEAD_OUTPUT_PREFIXES)


def _is_known_output(output):
    return output in HEAD_OUTPUTS or output.startswith(HEAD_OUTPUT_PREFIXES + OUTPUT_TAGS)


def _coverage_errors(rows, triage):
    """The rules that need a heads line and have none, and the lines of rules that need none."""
    seen = [r.get("id") for r in rows]
    errs = [f"{rid}: no heads line"
            for rid in sorted(wanted_rules(triage) - set(seen), key=rule_sort_key)]
    errs += [f"{rid}: heads line for a rule that needs none"
             for rid in sorted(set(seen) - wanted_rules(triage))]
    return errs


def _output_errors(rid, status, outputs):
    errs = []
    if status not in OUTPUT_STATUS:
        errs.append(f"{rid}: output {status!r}")
    if not outputs:
        errs.append(f"{rid}: no outputs named")
    errs += [f"{rid}: unknown output {o!r}" for o in outputs if not _is_known_output(o)]
    if status == "trained" and not any(_names_existing_output(o) for o in outputs):
        errs.append(f"{rid}: trained but names no existing output")
    return errs


def _input_errors(rid, row):
    errs = []
    if row.get("input") not in INPUT_STATUS:
        errs.append(f"{rid}: input {row.get('input')!r}")
    missing = row.get("inputs_missing") or []
    if (row.get("input") == "incomplete") != bool(missing):
        errs.append(f"{rid}: input {row.get('input')!r} disagrees with inputs_missing")
    errs += [f"{rid}: untagged missing input {i!r}"
             for i in missing if not i.startswith(INPUT_TAGS)]
    return errs


def _gen3_id_errors(rid, row):
    """What ``gen3-fields`` adds to a line: its keys, and that every id it names exists."""
    if not GEN3_FIELDS.exists():
        return []
    if any(k not in row for k in GEN3_KEYS):
        return [f"{rid}: gen3 fields missing; run gen3-fields"]
    ids = set(gen3_ids())
    return [f"{rid}: unknown {k} id {i!r}"
            for k in ("gen3_inputs", "gen3_outputs", "gen3_fixes")
            for i in row[k] if i not in ids]


def _row_errors(row):
    rid = row.get("id")
    return [*_output_errors(rid, row.get("output"), row.get("outputs") or []),
            *_input_errors(rid, row),
            *_gen3_id_errors(rid, row)]


def validate_heads(rows, triage):
    errs = _coverage_errors(rows, triage)
    for r in rows:
        errs += _row_errors(r)
    return errs
