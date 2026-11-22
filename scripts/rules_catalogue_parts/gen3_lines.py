"""``gen3-fields``: the gen-3 ids one heads line needs, and the verdicts it may hold."""

from .gen3_registry import requirement_closure
from .vocabulary import OUTPUT_TAGS

#: What a name maps to when no registry entry, override, cover or drop names it.
_UNMAPPED = object()


def _target_of(key, lookup, overrides):
    """What ``(rule id, name)`` maps to: an id, ``"covered"``, None (dropped) or ``_UNMAPPED``."""
    if key in overrides:
        return overrides[key]
    return lookup.get(key[1], _UNMAPPED)


def _mapped_names(row, lookup, overrides):
    """``(inputs, outputs, dropped, unmapped)``: where each missing input and tagged output goes."""
    ins, outs, dropped, unmapped = set(), set(), set(), []
    tagged_outputs = [o for o in row["outputs"] if o.startswith(OUTPUT_TAGS)]
    for context, names in (("input", row["inputs_missing"]), ("output", tagged_outputs)):
        for name in names:
            target = _target_of((row["id"], name), lookup[context], overrides)
            if target is _UNMAPPED:
                unmapped.append(name)
            elif target is None:
                dropped.add(name)
            elif target != "covered":
                (ins if context == "input" else outs).add(target)
    return ins, outs, dropped, unmapped


def _add_requirements(ins, outs, entries):
    """An entry's requires come with it: floating effects are read through every line."""
    for req in requirement_closure(entries, ins | outs) - (ins | outs):
        is_output = req.startswith(("out.", "planned.timing", "planned.failed"))
        (outs if is_output else ins).add(req)


def gen3_line(row, entries, lookup, overrides):
    """The gen3_* fields of one heads line, and the names nothing maps."""
    ins, outs, dropped, unmapped = _mapped_names(row, lookup, overrides)
    _add_requirements(ins, outs, entries)
    fixes = {i for i in ins | outs if i.startswith("fix.")}
    ins -= fixes
    outs -= fixes
    learnable = not dropped and not unmapped
    return {
        "gen3_inputs": sorted(ins), "gen3_outputs": sorted(outs), "gen3_fixes": sorted(fixes),
        "gen3_dropped": sorted(dropped), "learnable_after_gen3": learnable,
    }, unmapped


def learnable_today(row):
    """Whether the model learns a rule as the layout stands: an output trained
    on its outcome, and every fact deciding it read."""
    return row["output"] == "trained" and row["input"] == "complete"


def turn_action_rules(by_rule):
    """The rules whose declared detectors all read ``turn_action`` records: no
    class of the mixture writes those, so only the verdict ``recorded`` is
    right for them (FR-036). A ``none`` detector declares nothing."""
    from effects.domain.records import RecordKind

    out = set()
    for rule, scenarios in by_rule.items():
        kinds = [s["detector"].get("record_kind") for s in scenarios
                 if isinstance(s.get("detector"), dict) and "family" in s["detector"]]
        if kinds and all(kind == RecordKind.TURN_ACTION for kind in kinds):
            out.add(rule)
    return out


def turn_action_verdict_errors(heads_rows, by_rule):
    """One message per turn_action-only rule whose heads verdict is not ``recorded``."""
    only = turn_action_rules(by_rule)
    return [f"{row['id']}: shown by turn_action records only, verdict {row['output']!r} "
            f"is not 'recorded'" for row in heads_rows
            if row["id"] in only and row["output"] != "recorded"]
