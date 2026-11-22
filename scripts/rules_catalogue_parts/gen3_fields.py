"""``gen3-fields``: each heads line's gen-3 ids, the learnable list and GEN3-FIELDS.md."""

import json
import re
import sys
from collections import Counter, defaultdict

from .locations import RULES
from .ordering import rule_sort_key, unit_sort_key
from .reading import load_scenarios_by_rule, read_jsonl
from .vocabulary import OUTPUT_TAGS


def requirement_closure(entries, ids):
    """``ids`` and every entry they require, directly or through another."""
    out, todo = set(ids), list(ids)
    while todo:
        for req in entries[todo.pop()].get("requires", ()):
            if req not in out:
                out.add(req)
                todo.append(req)
    return out


_GEN3_IDS = None


def gen3_ids():
    global _GEN3_IDS
    if _GEN3_IDS is None:
        _GEN3_IDS = gen3_registry()[1].keys()
    return _GEN3_IDS


GEN3_FIELDS_NAME = "gen3-fields.json"
GEN3_FIELDS = RULES / GEN3_FIELDS_NAME
GEN3_KEYS = ("gen3_inputs", "gen3_outputs", "gen3_fixes", "gen3_dropped", "learnable_after_gen3")
#: The registry's member listing the rules learnable at the last run, written by gen3-fields.
LEARNABLE = "learnable"


def gen3_registry(path=None):
    """``(entries by id, {context: {name: id or None}})`` from gen3-fields.json.

    A name is looked up in its own context: a missing input among the inputs, an
    untrained output among the outputs, because a few names (``new:text_change``,
    ``new:room_unlocked``) were proposed in both roles. Within a context a name maps
    to one entry; ``None`` is covered or dropped.
    """
    reg = json.loads((path or GEN3_FIELDS).read_text(encoding="utf-8"))
    entries = {}
    by_context = {"input": defaultdict(set), "output": defaultdict(set)}

    def add(entry, contexts):
        entries[entry["id"]] = entry
        for name in entry["names"]:
            for c in contexts:
                by_context[c][name].add(entry["id"])

    for e in reg["inputs"]:
        add(e, ["input"])
    for e in reg["outputs"] + reg["fixes"]:
        add(e, ["output"])
    for e in reg["extensions"]:
        add(e, [c for c in ("input", "output") if c in e["applies_to"]])
    for e in reg["planned"]:
        add(e, ["output"] if e["id"] in ("planned.timing", "planned.failed_checks") else ["input"])
    lookup = {}
    for c, names in by_context.items():
        lookup[c] = {}
        for name, ids in names.items():
            if len(ids) > 1:
                raise ValueError(f"{name} maps to {sorted(ids)} as an {c}")
            lookup[c][name] = next(iter(ids))
        for name in reg["covered"]["names"]:
            lookup[c].setdefault(name, "covered")
        for group in reg["dropped"]:
            for name in group["names"]:
                lookup[c].setdefault(name, None)
    overrides = {(o["rule"], o["name"]): o["to"] for o in reg["overrides"]}
    return reg, entries, lookup, overrides


def gen3_line(row, entries, lookup, overrides):
    """The gen3_* fields of one heads line, and the names nothing maps."""
    ins, outs, fixes, dropped, unmapped = set(), set(), set(), set(), []
    tagged_outputs = [o for o in row["outputs"] if o.startswith(OUTPUT_TAGS)]
    for context, names in (("input", row["inputs_missing"]), ("output", tagged_outputs)):
        for name in names:
            key = (row["id"], name)
            if key in overrides:
                target = overrides[key]
            elif name in lookup[context]:
                target = lookup[context][name]
            else:
                unmapped.append(name)
                continue
            if target is None:
                dropped.add(name)
            elif target == "covered":
                continue
            elif context == "input":
                ins.add(target)
            else:
                outs.add(target)
    # An entry's requires come with it: floating effects are read through every line.
    for req in requirement_closure(entries, ins | outs) - (ins | outs):
        is_output = req.startswith(("out.", "planned.timing", "planned.failed"))
        (outs if is_output else ins).add(req)
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


def write_learnable(path, learnable):
    """Set the registry's ``learnable`` member to ``learnable``, leaving every
    other byte of the hand-kept file as it is."""
    text = path.read_text(encoding="utf-8")
    value = json.dumps(learnable)
    member = re.compile(rf'^(  "{LEARNABLE}": )\[[^\]]*\]', re.M)
    if member.search(text):
        text = member.sub(lambda m: m.group(1) + value, text)
    else:
        body = text.rstrip()
        if not body.endswith("}"):
            raise ValueError(f"{path} does not end with its closing brace")
        text = f'{body[:-1].rstrip()},\n  "{LEARNABLE}": {value}\n}}\n'
    if json.loads(text)[LEARNABLE] != learnable:
        raise ValueError(f"{path}: could not write its {LEARNABLE} member")
    path.write_text(text, encoding="utf-8", newline="\n")


def print_learnable_change(previous, current):
    """The rules the change made learnable, and any it made unlearnable, against
    the list the previous run recorded."""
    if previous is None:
        print(f"no previous {LEARNABLE} list: recording {len(current)} rules as the baseline")
        return
    gained = sorted(set(current) - set(previous), key=rule_sort_key)
    lost = sorted(set(previous) - set(current), key=rule_sort_key)
    for rule in gained:
        print(f"NEWLY LEARNABLE {rule}")
    for rule in lost:
        print(f"NO LONGER LEARNABLE {rule}")
    print(f"{len(gained)} rules newly learnable, {len(lost)} no longer, "
          f"against the previous {GEN3_FIELDS_NAME}")


def cmd_gen3_fields(args):
    rules_dir = args.rules_dir
    registry_path = rules_dir / GEN3_FIELDS_NAME
    reg, entries, lookup, overrides = gen3_registry(registry_path)
    heads = [(path, read_jsonl(path)) for path in
             sorted((rules_dir / "heads").glob("*.jsonl"), key=lambda p: unit_sort_key(p.stem))]
    errors = turn_action_verdict_errors([row for _, rows in heads for row in rows],
                                        load_scenarios_by_rule(rules_dir))
    for error in errors:
        print(f"VERDICT {error}")
    if errors:
        print(f"{len(errors)} turn_action-only rules not marked recorded; nothing written")
        sys.exit(1)
    all_rows, n_unmapped = [], 0
    for path, rows in heads:
        for row in rows:
            fields, unmapped = gen3_line(row, entries, lookup, overrides)
            for name in unmapped:
                print(f"UNMAPPED {row['id']}: {name}")
            n_unmapped += len(unmapped)
            for k in GEN3_KEYS:
                row.pop(k, None)
            row.update(fields)
            all_rows.append(row)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
    if n_unmapped:
        print(f"{n_unmapped} unmapped names")
        sys.exit(1)
    write_gen3_summary(reg, entries, all_rows, rules_dir)
    learnable = sorted((r["id"] for r in all_rows if learnable_today(r)), key=rule_sort_key)
    print_learnable_change(reg.get(LEARNABLE), learnable)
    write_learnable(registry_path, learnable)


def write_gen3_summary(reg, entries, rows, rules_dir=RULES):
    """rules/GEN3-FIELDS.md: each canonical field, with the rules it serves."""

    def closure(i):
        return requirement_closure(entries, {i})

    serves, alone = Counter(), Counter()
    for r in rows:
        ids = set(r["gen3_inputs"] + r["gen3_outputs"] + r["gen3_fixes"])
        serves.update(ids)
        if r["gen3_dropped"]:
            continue
        # Alone: this entry and what it requires are the rule's whole change.
        for i in ids:
            subsumed = any(ids <= closure(j) for j in ids if j != i and i in closure(j))
            if ids <= closure(i) and not subsumed:
                alone[i] += 1
    now = sum(1 for r in rows if learnable_today(r))
    after = sum(1 for r in rows if r["learnable_after_gen3"])
    out = ["# Gen-3 inputs and outputs", "",
           "Generated by `python scripts/rules_catalogue.py gen3-fields` from "
           "`rules/gen3-fields.json` and `rules/heads/`; do not edit by hand. Each rule's line "
           "in `rules/heads/<unit>.jsonl` carries `gen3_inputs`, `gen3_outputs`, `gen3_fixes`, "
           "`gen3_dropped` and `learnable_after_gen3`.", "",
           f"Rules: {len(rows)}. Learnable today: {now}. Learnable once every field, extension, "
           f"planned item and fix below is in: {after}. "
           f"Left out by a dropped name: {len(rows) - after}.", "",
           "Columns: rules = rules that need it, counting the rules that need it through another "
           "entry's requires; alone = rules for which it, with what it requires, is the whole "
           "change needed.", "",
           "## Conventions", ""]
    out += [f"- **{k.replace('_', ' ')}.** {v}" for k, v in reg["conventions"].items()] + [""]

    def table(title, items, cols):
        out.extend([f"## {title} ({len(items)})", "",
                    "| id | " + " | ".join(cols) + " | rules | alone |",
                    "|---|" + "---|" * len(cols) + "---:|---:|"])
        for e in sorted(items, key=lambda e: -serves[e["id"]]):
            cells = " | ".join(str(e.get(c, "")) for c in cols)
            out.append(f"| `{e['id']}` | {cells} | {serves[e['id']]} | {alone[e['id']]} |")
        out.append("")

    table("New inputs", reg["inputs"], ["slot", "shape", "source", "what"])
    table("New outputs", reg["outputs"], ["slot", "shape", "source", "what"])
    table("Extensions of existing inputs and outputs", reg["extensions"], ["applies_to", "what"])
    table("Already planned in the gen-3 design record", reg["planned"], ["what"])
    table("Collection and curation fixes", reg["fixes"], ["what"])
    out.extend(["## Dropped names", "", "| reason | names | rules |", "|---|---|---:|"])
    for group in reg["dropped"]:
        hit = sum(1 for r in rows if set(r["gen3_dropped"]) & set(group["names"]))
        names = ", ".join(f"`{n}`" for n in group["names"])
        out.append(f"| {group['reason']} | {names} | {hit} |")
    for o in reg["overrides"]:
        if o["to"] is None:
            out.append(f"| {o['reason']} ({o['rule']}) | `{o['name']}` | 1 |")
    out.append("")
    (rules_dir / "GEN3-FIELDS.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"learnable today {now}, after gen-3 {after}, of {len(rows)}")
