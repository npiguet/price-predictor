"""Comprehensive Rules catalogue: split the rules text, batch it for agents, validate sidecars,
report status.

Subcommands:
  split     parse rules/raw/MagicCompRules-*.txt into rules/rules.jsonl, rules/cr/<unit>.md
            and rules/cr/index.md
  packets   group units into agent work packets (rules/packets.json)
  validate  check every triage and scenario sidecar against the rules index and Forge's cardsfolder
  status    regenerate rules/STATUS.md from the sidecars

A unit is a three-digit section (704), except sections 701 and 702, whose units are their
second-level rules (702.2, one keyword each), because those two sections hold a third of all rules.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "rules"
CARDSFOLDER = ROOT.parent / "forge" / "forge-gui" / "res" / "cardsfolder"
if not CARDSFOLDER.exists():  # a worktree under .claude/worktrees/ sits three levels deeper
    CARDSFOLDER = ROOT.parents[3] / "forge" / "forge-gui" / "res" / "cardsfolder"

SPLIT_SECTIONS = {"701", "702"}
CATEGORIES = {"mechanical", "legality", "decision", "definitional", "out_of_scope"}
OBSERVABILITY = {"direct", "indirect", "needs_new_fields", "none"}
RECORD_KINDS = {"resolution", "rewrite", "continuous", "combat", "trigger", "playability"}
SCENARIO_CATEGORIES = {"mechanical", "legality"}
POLARITIES = {"fires", "does_not_fire", "other"}
# Heads pass (rules/MODEL-FIELDS.md): what the model reads and is trained on per rule.
OUTPUT_STATUS = {"trained", "recorded", "missing", "planned"}
INPUT_STATUS = {"complete", "incomplete"}
HEAD_OUTPUTS = {
    "gate", "zone_outcome", "tap_state", "damage_taken", "power_delta", "toughness_delta",
    "pt_duration", "types_gained", "types_lost", "colors_gained", "colors_lost",
    "type_color_duration", "keywords_gained", "keywords_lost", "control_change",
    "attached_to_change", "life_delta", "cards_drawn", "cards_discarded", "cards_milled",
    "library_events", "poison_delta", "target_legal", "attacker_legal", "blocker_legal",
    "min_blockers", "can_play", "affordable", "has_legal_target", "mana_paid", "trigger_fired",
    "created_objects",
}
HEAD_OUTPUT_PREFIXES = ("counters_delta_", "mana_delta_")
OUTPUT_TAGS = ("record:", "new:", "planned:")
INPUT_TAGS = ("snapshot:", "new:", "planned:")

RULE_RE = re.compile(r"^(\d{3})\.(\d+)([a-z]*)\.? (.*)$")
SECTION_RE = re.compile(r"^(\d{3})\. (.*)$")
CHAPTER_RE = re.compile(r"^(\d)\. (.*)$")
REF_RE = re.compile(r"\b(\d{3}(?:\.\d+[a-z]*)?)\b")


def unit_of(section, number):
    return f"{section}.{number}" if section in SPLIT_SECTIONS else section


def unit_sort_key(unit):
    head, _, tail = unit.partition(".")
    return (int(head), int(tail) if tail else -1)


def rule_sort_key(rule_id):
    m = re.match(r"(\d{3})\.(\d+)([a-z]*)", rule_id)
    letters = m.group(3)
    return (int(m.group(1)), int(m.group(2)), len(letters), letters)


def clean(line):
    return line.replace(" ", " ").strip()


def raw_file():
    files = sorted((RULES / "raw").glob("MagicCompRules-*.txt"))
    if not files:
        sys.exit("no rules/raw/MagicCompRules-*.txt")
    return files[-1]


def cmd_split(_args):
    lines = [clean(raw) for raw in raw_file().read_text(encoding="utf-8-sig").splitlines()]
    starts = [i for i, line in enumerate(lines) if line == "1. Game Concepts"]
    glossaries = [i for i, line in enumerate(lines) if line == "Glossary"]
    credits = [i for i, line in enumerate(lines) if line == "Credits"]
    body, glossary = lines[starts[1]:glossaries[1]], lines[glossaries[1] + 1:credits[1]]

    rules, chapters, sections = [], {}, {}
    for line in body:
        if not line:
            continue
        m = RULE_RE.match(line)
        if m:
            sec, num, letters, text = m.groups()
            rid = f"{sec}.{num}{letters}"
            parent = f"{sec}.{num}" if letters else sec
            refs = sorted({r for r in REF_RE.findall(text) if r != rid})
            rules.append({"id": rid, "unit": unit_of(sec, num), "section": sec, "chapter": sec[0],
                          "parent": parent, "text": text, "examples": [], "refs": refs})
            continue
        m = SECTION_RE.match(line)
        if m:
            sections[m.group(1)] = m.group(2)
            continue
        m = CHAPTER_RE.match(line)
        if m and len(line) < 60:
            chapters[m.group(1)] = m.group(2)
            continue
        if not rules:
            raise ValueError(f"text before the first rule: {line!r}")
        if line.startswith("Example:"):
            rules[-1]["examples"].append(line[len("Example:"):].strip())
        else:
            rules[-1]["text"] += "\n" + line

    # a split section's unit title is its first rule's text,
    # e.g. "Deathtouch" from "702.2. Deathtouch"
    unit_titles = {}
    for r in rules:
        if r["unit"] not in unit_titles:
            in_split = r["section"] in SPLIT_SECTIONS
            unit_titles[r["unit"]] = (
                r["text"].split("\n")[0][:80] if in_split else sections[r["section"]]
            )

    with open(RULES / "rules.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rules:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_unit = defaultdict(list)
    for r in rules:
        by_unit[r["unit"]].append(r)
    cr = RULES / "cr"
    cr.mkdir(parents=True, exist_ok=True)
    for old in cr.glob("*.md"):
        old.unlink()
    index_rows = []
    for unit in sorted(by_unit, key=unit_sort_key):
        rs = by_unit[unit]
        sec = rs[0]["section"]
        title = unit_titles[unit]
        chapter = rs[0]["chapter"]
        out = [f"# {unit}. {title}", "",
               f"Chapter {chapter} ({chapters[chapter]}), section {sec} ({sections[sec]}).", ""]
        for r in rs:
            out.append(f"**{r['id']}** {r['text']}")
            for ex in r["examples"]:
                out.append(f"> Example: {ex}")
            out.append("")
        (cr / f"{unit}.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
        chars = sum(len(r["text"]) + sum(len(e) for e in r["examples"]) for r in rs)
        index_rows.append((unit, title, len(rs), chars))

    gl_out, entry = ["# Glossary", ""], []
    for line in glossary + [""]:
        if line:
            entry.append(line)
        elif entry:
            gl_out.append(f"**{entry[0]}** " + " ".join(entry[1:]))
            gl_out.append("")
            entry = []
    (cr / "glossary.md").write_text("\n".join(gl_out), encoding="utf-8", newline="\n")

    idx = ["# Comprehensive Rules index", "",
           f"Source: `rules/raw/{raw_file().name}`. "
           f"{len(rules)} rules in {len(index_rows)} units.", "",
           "| unit | title | rules | chars | file |", "|---|---|---|---|---|"]
    for unit, title, n, chars in index_rows:
        idx.append(f"| {unit} | {title} | {n} | {chars} | [cr/{unit}.md]({unit}.md) |")
    idx += ["", "Glossary: [cr/glossary.md](glossary.md)."]
    (cr / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rules)} rules, {len(index_rows)} units, {len(gl_out) // 2} glossary entries")


def load_rules():
    with open(RULES / "rules.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def units_with_sizes(rules):
    sizes, counts = Counter(), Counter()
    for r in rules:
        sizes[r["unit"]] += len(r["text"]) + sum(len(e) for e in r["examples"])
        counts[r["unit"]] += 1
    return sizes, counts


def cmd_packets(args):
    """Greedy in-order packing of units up to --max-chars, never splitting a unit or crossing
    a chapter."""
    rules = load_rules()
    sizes, counts = units_with_sizes(rules)
    chapter_of = {r["unit"]: r["chapter"] for r in rules}
    packets, current, chars = [], [], 0
    for unit in sorted(sizes, key=unit_sort_key):
        full = chars + sizes[unit] > args.max_chars
        if current and (full or chapter_of[current[-1]] != chapter_of[unit]):
            packets.append(current)
            current, chars = [], 0
        current.append(unit)
        chars += sizes[unit]
    packets.append(current)
    out = [{"id": f"p{i:03d}", "chapter": chapter_of[p[0]], "units": p,
            "rules": sum(counts[u] for u in p), "chars": sum(sizes[u] for u in p)}
           for i, p in enumerate(packets)]
    (RULES / "packets.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(out)} packets; chars per packet max {max(p['chars'] for p in out)}")


PREMARKED_CHAPTERS = {"8": "multiplayer rules; collection plays two-player games only",
                      "9": "casual variant; collection plays two-player sealed games only"}


def cmd_premark(_args):
    """Triage chapters 8 and 9 as out_of_scope without an agent."""
    by_unit = defaultdict(list)
    for r in load_rules():
        if r["chapter"] in PREMARKED_CHAPTERS:
            by_unit[r["unit"]].append(r)
    (RULES / "triage").mkdir(exist_ok=True)
    for unit, rs in by_unit.items():
        with open(RULES / "triage" / f"{unit}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps({"id": r["id"], "category": "out_of_scope",
                                    "observability": "none", "record_kinds": [], "outcome": "",
                                    "note": PREMARKED_CHAPTERS[r["chapter"]]}) + "\n")
    print(f"premarked {len(by_unit)} units")


NAMES_CACHE = RULES / ".forge-card-names.txt"  # gitignored; delete it after updating Forge


def card_names():
    if NAMES_CACHE.exists():
        return set(NAMES_CACHE.read_text(encoding="utf-8").splitlines())
    names = set()
    for p in CARDSFOLDER.rglob("*.txt"):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Name:"):
                    names.add(line[5:].strip())
    NAMES_CACHE.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8", newline="\n")
    return names


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path.name}:{n}: {e}") from e
    return out


#: Each per-unit file a unit may have, by its directory under rules/.
UNIT_FILES = {"triage": ".jsonl", "scenarios": ".json", "heads": ".jsonl"}


def load_unit(kind, unit):
    """A unit's triage, scenarios or heads file, parsed; None while it is not written."""
    p = RULES / kind / f"{unit}{UNIT_FILES[kind]}"
    if not p.exists():
        return None
    return read_jsonl(p) if p.suffix == ".jsonl" else json.loads(p.read_text(encoding="utf-8"))


def requirement_closure(entries, ids):
    """``ids`` and every entry they require, directly or through another."""
    out, todo = set(ids), list(ids)
    while todo:
        for req in entries[todo.pop()].get("requires", ()):
            if req not in out:
                out.add(req)
                todo.append(req)
    return out


def needs_scenario(triage_row):
    return (triage_row["category"] in SCENARIO_CATEGORIES
            and triage_row["observability"] != "none")


def wanted_rules(triage):
    """The rules that need a scenario, and so a heads line too."""
    return {r["id"] for r in triage if needs_scenario(r)}


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


_GEN3_IDS = None


def gen3_ids():
    global _GEN3_IDS
    if _GEN3_IDS is None:
        _GEN3_IDS = gen3_registry()[1].keys()
    return _GEN3_IDS


def validate_triage(rows, rule_ids):
    errs = []
    seen = [r.get("id") for r in rows]
    missing = sorted(set(rule_ids) - set(seen), key=rule_sort_key)
    extra = sorted(set(seen) - set(rule_ids))
    if missing:
        errs.append(f"missing rules {missing}")
    if extra:
        errs.append(f"unknown rules {extra}")
    for r in rows:
        if r.get("category") not in CATEGORIES:
            errs.append(f"{r.get('id')}: category {r.get('category')!r}")
        if r.get("observability") not in OBSERVABILITY:
            errs.append(f"{r.get('id')}: observability {r.get('observability')!r}")
        bad = set(r.get("record_kinds") or []) - RECORD_KINDS
        if bad:
            errs.append(f"{r.get('id')}: record_kinds {sorted(bad)}")
    return errs


def validate_scenarios(doc, triage, all_rule_ids, names):
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
    return errs, warns


def cmd_validate(args):
    rules = load_rules()
    all_ids = {r["id"] for r in rules}
    ids_by_unit = defaultdict(list)
    for r in rules:
        ids_by_unit[r["unit"]].append(r["id"])
    names = card_names() if not args.no_cards else None
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
            e2, w2 = validate_scenarios(doc, tri, all_ids, known)
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


GEN3_FIELDS = RULES / "gen3-fields.json"
GEN3_KEYS = ("gen3_inputs", "gen3_outputs", "gen3_fixes", "gen3_dropped", "learnable_after_gen3")


def gen3_registry():
    """``(entries by id, {context: {name: id or None}})`` from gen3-fields.json.

    A name is looked up in its own context: a missing input among the inputs, an
    untrained output among the outputs, because a few names (``new:text_change``,
    ``new:room_unlocked``) were proposed in both roles. Within a context a name maps
    to one entry; ``None`` is covered or dropped.
    """
    reg = json.loads(GEN3_FIELDS.read_text(encoding="utf-8"))
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


def cmd_gen3_fields(_args):
    reg, entries, lookup, overrides = gen3_registry()
    all_rows, n_unmapped = [], 0
    for path in sorted((RULES / "heads").glob("*.jsonl"), key=lambda p: unit_sort_key(p.stem)):
        rows = read_jsonl(path)
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
    write_gen3_summary(reg, entries, all_rows)


def write_gen3_summary(reg, entries, rows):
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
    now = sum(1 for r in rows if r["output"] == "trained" and r["input"] == "complete")
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
    (RULES / "GEN3-FIELDS.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"learnable today {now}, after gen-3 {after}, of {len(rows)}")


def cmd_status(_args):
    rules = load_rules()
    sizes, counts = units_with_sizes(rules)
    title = {}
    for line in (RULES / "cr" / "index.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and re.match(r"\d{3}", cells[1]):
            title[cells[1]] = cells[2]
    cat_total, kind_total = Counter(), Counter()
    rows = []
    n_tri = n_scen = n_pairs = 0
    for unit in sorted(counts, key=unit_sort_key):
        tri = load_unit("triage", unit)
        doc = load_unit("scenarios", unit)
        cats = Counter(r["category"] for r in tri) if tri else Counter()
        cat_total.update(cats)
        want = sum(1 for r in tri if needs_scenario(r)) if tri else 0
        if tri is not None:
            n_tri += 1
        ns = len(doc["scenarios"]) if doc else 0
        if doc:
            n_scen += 1
            n_pairs += ns
            kind_total.update(s["record_kind"] for s in doc["scenarios"])
        verified = bool(doc and doc.get("verified"))
        tri_mark = "x" if tri is not None else " "
        scen_mark = "x" if doc is not None else ("–" if tri is not None and want == 0 else " ")
        ver_mark = "x" if verified else ("–" if scen_mark == "–" else " ")
        cat_order = ("mechanical", "legality", "decision", "definitional", "out_of_scope")
        cat_str = " ".join(f"{c[:4]}{cats[c]}" for c in cat_order if cats[c])
        rows.append(f"| {unit} | {title.get(unit, '')} | {counts[unit]} | [{tri_mark}] "
                    f"| [{scen_mark}] | [{ver_mark}] | {cat_str} | {ns} |")
    out = ["# Rules catalogue status", "",
           "Generated by `python scripts/rules_catalogue.py status`; do not edit by hand.", "",
           f"Units triaged: {n_tri}/{len(counts)}. Units with scenarios: {n_scen}. "
           f"Scenarios: {n_pairs}.", "",
           "Rules by category: "
           + ", ".join(f"{k} {v}" for k, v in cat_total.most_common()) + ".", "",
           "Scenarios by record kind: "
           + (", ".join(f"{k} {v}" for k, v in kind_total.most_common()) or "none") + ".", "",
           "Columns: triage, scenarios (– = no rule in the unit needs one), verified.", "",
           "| unit | title | rules | triage | scen | verified | categories | scenarios |",
           "|---|---|---|---|---|---|---|---|"] + rows
    (RULES / "STATUS.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"triaged {n_tri}/{len(counts)}, scenario units {n_scen}, scenarios {n_pairs}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    p = sub.add_parser("packets")
    p.add_argument("--max-chars", type=int, default=16000)
    p = sub.add_parser("validate")
    p.add_argument("units", nargs="*")
    p.add_argument("--no-cards", action="store_true")
    p.add_argument("--warnings", action="store_true")
    p.add_argument("--heads", action="store_true", help="also check rules/heads/")
    sub.add_parser("status")
    sub.add_parser("premark")
    sub.add_parser("gen3-fields")
    args = ap.parse_args()
    {"split": cmd_split, "packets": cmd_packets, "validate": cmd_validate, "status": cmd_status,
     "premark": cmd_premark, "gen3-fields": cmd_gen3_fields}[args.cmd](args)


if __name__ == "__main__":
    main()
