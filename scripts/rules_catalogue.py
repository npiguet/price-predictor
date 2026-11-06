"""Comprehensive Rules catalogue: split the rules text, batch it for agents, validate sidecars,
report status.

Subcommands:
  split     parse rules/raw/MagicCompRules-*.txt into rules/rules.jsonl, rules/cr/<unit>.md
            and rules/cr/index.md
  packets   group units into agent work packets (rules/packets.json)
  validate  check every triage and scenario sidecar against the rules index and Forge's cardsfolder
  status    regenerate rules/STATUS.md from the sidecars
  gen3-fields
            write each heads line's gen-3 ids and learnable_after_gen3 and rules/GEN3-FIELDS.md,
            print the rules newly learnable against the list rules/gen3-fields.json recorded at
            the last run, and record the new list there; refuses a turn_action-only rule whose
            verdict is not `recorded`
  directed-rules --counts FILE
            write rules/directed-fork-rules.txt (the rules enabled for directed forks) and
            rules/coverage-wording-cards.txt (the cards of their scenarios) from gen-2's
            per-rule counts

A unit is a three-digit section (704), except sections 701 and 702, whose units are their
second-level rules (702.2, one keyword each), because those two sections hold a third of all rules.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from _lib.paths import FORGE_RES, OUTPUT, REPO

from effects.domain import rule_catalogue

RULES = REPO / "rules"
CARDSFOLDER = FORGE_RES / "cardsfolder"

SPLIT_SECTIONS = {"701", "702"}
# The catalogue's vocabulary is the package's, so the corpus code that reads these
# sidecars and this validator spell every enum alike.
CATEGORIES = rule_catalogue.CATEGORIES
OBSERVABILITY = rule_catalogue.OBSERVABILITY
RECORD_KINDS = rule_catalogue.RECORD_KINDS
SCENARIO_CATEGORIES = rule_catalogue.SCENARIO_CATEGORIES
POLARITIES = rule_catalogue.POLARITIES
# Heads pass (rules/MODEL-FIELDS.md): what the model reads and is trained on per rule.
OUTPUT_STATUS = rule_catalogue.OUTPUT_STATUS
INPUT_STATUS = rule_catalogue.INPUT_STATUS
# Derived from the model's own layout rather than kept by hand.
HEAD_OUTPUTS = rule_catalogue.head_outputs()
HEAD_OUTPUT_PREFIXES = rule_catalogue.HEAD_OUTPUT_PREFIXES
OUTPUT_TAGS = rule_catalogue.OUTPUT_TAGS
INPUT_TAGS = rule_catalogue.INPUT_TAGS
FORK_OPERATIONS = rule_catalogue.FORK_OPERATIONS

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


DEPARTURE_STATUS = frozenset(rule_catalogue.DepartureStatus)
DEPARTURE_RESOLUTION = frozenset(rule_catalogue.DepartureResolution)


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


def detector_families():
    """``family id -> declared params`` from the package's registry."""
    from effects.domain.rule_detectors import families

    return {family_id: family.params for family_id, family in families().items()}


def validate_scenarios(doc, triage, all_rule_ids, names, families=None):
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
    return errs, warns


def cmd_validate(args):
    rules = load_rules()
    all_ids = {r["id"] for r in rules}
    ids_by_unit = defaultdict(list)
    for r in rules:
        ids_by_unit[r["unit"]].append(r["id"])
    names = card_names() if not args.no_cards else None
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
            e2, w2 = validate_scenarios(doc, tri, all_ids, known, families)
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
                 if isinstance(s.get("detector"), dict) and "none" not in s["detector"]]
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


#: Fewer `fires` records than this over all sources makes a rule rare (FR-014b).
RARE_FIRES = 100
#: Enabled for directed forks whatever their counts, each for its reason (FR-014b).
ALWAYS_DIRECTED = {
    "704.5f": "the user's T038 decision of 2026-11-02, whatever its counts",
    "702.98a": "its 8 does_not_fire records on gen-2 are one Dead Reveler in one game, far "
               "below the 68 per polarity a rule test needs (the user's decision of 2026-11-06)",
}
#: Left out whatever their counts, each for its reason: no detector can tell
#: their fork from its live game, so a directed fork of theirs would add no
#: record of the side it is for (the user's decision of 2026-11-06, T096).
NOT_DIRECTED = {
    "104.2a": "the AI never concedes, the only forkable variant (add Platinum Angel) stops "
              "any other loss, and the lacking polarity is no event",
    "500.11": "500.11-1 (Eon Hub) has no detector, a static skip filing no event, and "
              "500.11-2's contrast is a choice",
    "701.12d": "701.12d-1 has no detector and its only forkable variant is polarity other; "
               "701.12d-2's contrast is ownership, not a board edit",
    "702.62d": "the suspend cast happens inside a resolution, never at a priority stop, so "
               "no playability record exists for it",
}
#: Left out of this collection only: they qualified on the recount after the T030
#: detectors (2026-11-06), when the list of 70 was fixed for production; by the user's
#: rule, more than five newcomers wait for a later collection rather than delay this one
#: by a registration each.
DEFERRED_DIRECTED = dict.fromkeys((
    "104.4a", "104.4c", "115.7a", "701.10c", "701.11c", "701.40c", "701.40d",
    "702.19c", "702.19e", "702.19f", "702.131a", "702.170c", "712.4c", "712.16",
), "qualified after the production list was fixed; registered for a later collection")
FIRES, DOES_NOT_FIRE = "fires", "does_not_fire"


def rule_unit(rule_id):
    section, _, rest = rule_id.partition(".")
    return unit_of(section, re.match(r"\d+", rest).group(0))


def load_scenarios_by_rule(rules_dir):
    """``rule id -> [scenario, …]`` over every scenarios file under ``rules_dir``."""
    by_rule = defaultdict(list)
    paths = sorted((rules_dir / "scenarios").glob("*.json"), key=lambda p: unit_sort_key(p.stem))
    for path in paths:
        for scenario in json.loads(path.read_text(encoding="utf-8")).get("scenarios", []):
            for rule in scenario.get("rules", []):
                by_rule[rule].append(scenario)
    return by_rule


def counted_on_gen2(scenario):
    """Whether the scenario's detector tags gen-2 records: a family declared
    over a record kind gen-2 writes, and no gen-3 field read, by the
    declaration or by the family's params."""
    from effects.domain.records import RecordKind
    from effects.domain.rule_detectors import families

    declaration = scenario.get("detector") or {}
    family = families().get(declaration.get("family"))
    return (family is not None and declaration.get("record_kind") != RecordKind.TURN_ACTION
            and not declaration.get("needs")
            and not family.needs(declaration.get("params") or {}))


def rule_counts(counts, rule, scenarios):
    """``(fires, does_not_fire)`` over all sources, or None where the counts
    cannot speak for the rule. The counts list only rules some record tagged,
    so a rule they omit has none of either when a detector of its tags gen-2
    records; one not countable there, declared ``none``, or with no gen-2
    detector is not shown by them at all."""
    found = counts["rules"].get(rule)
    if found is not None:
        return (sum(found.get(FIRES, {}).values()), sum(found.get(DOES_NOT_FIRE, {}).values()))
    if rule in counts["not_countable"] or rule in counts["none"]:
        return None
    return (0, 0) if any(counted_on_gen2(s) for s in scenarios) else None


def lacking_polarities(fires, does_not_fire):
    """The sides the counts lack: `fires` on a rare rule, `does_not_fire` on a one-sided one."""
    return (({FIRES} if fires < RARE_FIRES else set())
            | ({DOES_NOT_FIRE} if does_not_fire == 0 else set()))


def forkable_variants(scenarios, lacking):
    """``(scenario id, label, polarity, operation)`` of every non-base variant
    classified into an operation, those of a polarity the counts lack first."""
    found = [(s["id"], v["label"], v["polarity"], v["operation"])
             for s in scenarios for v in s.get("variants", [])[1:]
             if isinstance(v.get("operation"), str)]
    return sorted(found, key=lambda f: f[2] not in lacking)


def directed_rules(counts, by_rule):
    """The rules FR-014b enables, in unit order: ``rule -> {counts, lacking,
    variants, by_counts}``; ``by_counts`` is False for a rule listed only as
    one of ``ALWAYS_DIRECTED``. A rule of ``NOT_DIRECTED`` or ``DEFERRED_DIRECTED`` is
    never listed."""
    out = {}
    for rule, scenarios in by_rule.items():
        if rule in NOT_DIRECTED or rule in DEFERRED_DIRECTED:
            continue
        found = rule_counts(counts, rule, scenarios)
        lacking = lacking_polarities(*found) if found is not None else set()
        by_counts = bool(lacking)
        if not by_counts and rule not in ALWAYS_DIRECTED:
            continue
        variants = forkable_variants(scenarios, lacking)
        if variants:
            out[rule] = {"counts": found, "lacking": lacking, "variants": variants,
                         "by_counts": by_counts}
    return dict(sorted(out.items(), key=lambda item: (unit_sort_key(rule_unit(item[0])),
                                                      rule_sort_key(item[0]))))


def converted_card_of_face(cards_folder):
    """``face name -> converted card name`` over a converted tree: a scenario
    may name a back face (Ribbons, a meld result), which the coverage round
    knows only by the card's own ``name:``, its first."""
    from price_predictor.domain.tokenizer import extract_card_name

    out = {}
    for path in cards_folder.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        card = extract_card_name(text)
        for line in text.splitlines():
            if line.startswith("name:"):
                out.setdefault(line[len("name:"):].strip(), card)
        out[card] = card
    return out


def wording_cards(chosen, by_rule, card_of_face):
    """The cards of the scenarios of the rules enabled for their counts, as
    the converted lower-case names (FR-038a), and the folded names no
    converted card carries."""
    from effects.domain.card_names import fold_card_name

    named = {fold_card_name(card) for rule, row in chosen.items() if row["by_counts"]
             for scenario in by_rule[rule] for card in scenario.get("cards", [])}
    unknown = sorted(name for name in named if name not in card_of_face)
    return sorted({card_of_face.get(name, name) for name in named}), unknown


def cmd_directed_rules(args):
    rules_dir = args.rules_dir
    counts = json.loads(args.counts.read_text(encoding="utf-8"))
    by_rule = load_scenarios_by_rule(rules_dir)
    chosen = directed_rules(counts, by_rule)
    cards, unknown = wording_cards(chosen, by_rule, converted_card_of_face(args.cards_folder))
    (rules_dir / "directed-fork-rules.txt").write_text(
        "".join(f"{rule}\n" for rule in chosen), encoding="utf-8", newline="\n")
    (rules_dir / "coverage-wording-cards.txt").write_text(
        "".join(f"{card}\n" for card in cards), encoding="utf-8", newline="\n")
    for rule, row in chosen.items():
        fires, does_not_fire = row["counts"] if row["counts"] is not None else ("?", "?")
        why = ", ".join(sorted(({"rare"} if FIRES in row["lacking"] else set())
                               | ({"one-sided"} if DOES_NOT_FIRE in row["lacking"] else set())))
        print(f"{rule:12s} fires {fires:>9} does_not_fire {does_not_fire:>9}  {why or 'always'}")
        for scenario_id, label, polarity, operation in row["variants"]:
            mark = "*" if polarity in row["lacking"] else " "
            print(f"    {mark} {scenario_id} {label} ({polarity}): {operation}")
    rare = sum(FIRES in row["lacking"] for row in chosen.values())
    one_sided = sum(DOES_NOT_FIRE in row["lacking"] for row in chosen.values())
    for rule, reason in ALWAYS_DIRECTED.items():
        print(f"always listed {rule}: {reason}")
    for rule, reason in {**NOT_DIRECTED, **DEFERRED_DIRECTED}.items():
        print(f"left out {rule}: {reason}")
    print(f"{len(chosen)} rules ({rare} rare, {one_sided} one-sided, "
          f"{sum(not row['by_counts'] for row in chosen.values())} always), {len(cards)} wording "
          f"cards; * marks a variant of a polarity the counts lack")
    if unknown:
        print(f"no converted card under {args.cards_folder}: {', '.join(unknown)}")


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
    p = sub.add_parser("gen3-fields", help="write each heads line's gen-3 ids and "
                       "learnable_after_gen3, and print the rules newly learnable (FR-016)")
    p.add_argument("--rules-dir", type=Path, default=RULES, help=argparse.SUPPRESS)
    p = sub.add_parser("directed-rules", help="write the rules enabled for directed forks and "
                       "the cards of their scenarios (FR-014b, FR-038a)")
    p.add_argument("--counts", type=Path, required=True,
                   help="gen-2's per-rule counts, as `effects rule-counts` writes them")
    p.add_argument("--cards-folder", type=Path, default=OUTPUT / "cardsfolder",
                   help="the converted card tree whose names the wording list uses "
                   "(default: %(default)s)")
    p.add_argument("--rules-dir", type=Path, default=RULES, help=argparse.SUPPRESS)
    args = ap.parse_args()
    {"split": cmd_split, "packets": cmd_packets, "validate": cmd_validate, "status": cmd_status,
     "premark": cmd_premark, "gen3-fields": cmd_gen3_fields,
     "directed-rules": cmd_directed_rules}[args.cmd](args)


if __name__ == "__main__":
    main()
