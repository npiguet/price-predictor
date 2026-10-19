"""Comprehensive Rules catalogue: split the rules text, batch it for agents, validate sidecars, report status.

Subcommands:
  split     parse rules/raw/MagicCompRules-*.txt into rules/rules.jsonl, rules/cr/<unit>.md and rules/cr/index.md
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
    lines = [clean(l) for l in raw_file().read_text(encoding="utf-8-sig").splitlines()]
    starts = [i for i, l in enumerate(lines) if l == "1. Game Concepts"]
    glossaries = [i for i, l in enumerate(lines) if l == "Glossary"]
    credits = [i for i, l in enumerate(lines) if l == "Credits"]
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

    # a split section's unit title is its first rule's text, e.g. "Deathtouch" from "702.2. Deathtouch"
    unit_titles = {}
    for r in rules:
        if r["unit"] not in unit_titles:
            in_split = r["section"] in SPLIT_SECTIONS
            unit_titles[r["unit"]] = r["text"].split("\n")[0][:80] if in_split else sections[r["section"]]

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
        out = [f"# {unit}. {title}", "",
               f"Chapter {rs[0]['chapter']} ({chapters[rs[0]['chapter']]}), section {sec} ({sections[sec]}).", ""]
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
           f"Source: `rules/raw/{raw_file().name}`. {len(rules)} rules in {len(index_rows)} units.", "",
           "| unit | title | rules | chars | file |", "|---|---|---|---|---|"]
    for unit, title, n, chars in index_rows:
        idx.append(f"| {unit} | {title} | {n} | {chars} | [cr/{unit}.md]({unit}.md) |")
    idx += ["", "Glossary: [cr/glossary.md](glossary.md)."]
    (cr / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rules)} rules, {len(index_rows)} units, {len(gl_out) // 2} glossary entries")


def load_rules():
    with open(RULES / "rules.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def units_with_sizes(rules):
    sizes, counts = Counter(), Counter()
    for r in rules:
        sizes[r["unit"]] += len(r["text"]) + sum(len(e) for e in r["examples"])
        counts[r["unit"]] += 1
    return sizes, counts


def cmd_packets(args):
    """Greedy in-order packing of units up to --max-chars, never splitting a unit or crossing a chapter."""
    rules = load_rules()
    sizes, counts = units_with_sizes(rules)
    chapter_of = {r["unit"]: r["chapter"] for r in rules}
    packets, current, chars = [], [], 0
    for unit in sorted(sizes, key=unit_sort_key):
        if current and (chars + sizes[unit] > args.max_chars or chapter_of[current[-1]] != chapter_of[unit]):
            packets.append(current)
            current, chars = [], 0
        current.append(unit)
        chars += sizes[unit]
    packets.append(current)
    out = [{"id": f"p{i:03d}", "chapter": chapter_of[p[0]], "units": p,
            "rules": sum(counts[u] for u in p), "chars": sum(sizes[u] for u in p)} for i, p in enumerate(packets)]
    (RULES / "packets.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
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
                f.write(json.dumps({"id": r["id"], "category": "out_of_scope", "observability": "none",
                                    "record_kinds": [], "outcome": "",
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


def load_triage(unit):
    p = RULES / "triage" / f"{unit}.jsonl"
    return read_jsonl(p) if p.exists() else None


def load_scenarios(unit):
    p = RULES / "scenarios" / f"{unit}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def validate_triage(unit, rows, rule_ids):
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


def validate_scenarios(unit, doc, triage, all_rule_ids, names):
    errs, warns = [], []
    scen = doc.get("scenarios", [])
    covered = {rid for s in scen for rid in s.get("rules", [])}
    skipped = {s["id"] for s in doc.get("skipped", [])}
    wanted = {r["id"] for r in triage if r["category"] in SCENARIO_CATEGORIES and r["observability"] != "none"}
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
            tri = load_triage(unit)
        except ValueError as e:
            tri, errs = None, [str(e)]
        if tri is not None:
            errs += validate_triage(unit, tri, ids_by_unit[unit])
        try:
            doc = load_scenarios(unit)
        except json.JSONDecodeError as e:
            doc, errs = None, errs + [f"scenarios json: {e}"]
        if doc is not None and tri is not None:
            e2, w2 = validate_scenarios(unit, doc, tri, all_ids, names if names is not None else _AnyName())
            errs += e2
            warns += w2
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
        tri = load_triage(unit)
        doc = load_scenarios(unit)
        cats = Counter(r["category"] for r in tri) if tri else Counter()
        cat_total.update(cats)
        want = sum(1 for r in tri if r["category"] in SCENARIO_CATEGORIES and r["observability"] != "none") if tri else 0
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
        cat_str = " ".join(f"{c[:4]}{cats[c]}" for c in ("mechanical", "legality", "decision", "definitional", "out_of_scope") if cats[c])
        rows.append(f"| {unit} | {title.get(unit, '')} | {counts[unit]} | [{tri_mark}] | [{scen_mark}] | [{ver_mark}] | {cat_str} | {ns} |")
    out = ["# Rules catalogue status", "",
           "Generated by `python scripts/rules_catalogue.py status`; do not edit by hand.", "",
           f"Units triaged: {n_tri}/{len(counts)}. Units with scenarios: {n_scen}. Scenarios: {n_pairs}.", "",
           "Rules by category: " + ", ".join(f"{k} {v}" for k, v in cat_total.most_common()) + ".", "",
           "Scenarios by record kind: " + (", ".join(f"{k} {v}" for k, v in kind_total.most_common()) or "none") + ".", "",
           "Columns: triage, scenarios (– = no rule in the unit needs one), verified.", "",
           "| unit | title | rules | triage | scen | verified | categories | scenarios |",
           "|---|---|---|---|---|---|---|---|"] + rows
    (RULES / "STATUS.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"triaged {n_tri}/{len(counts)}, scenario units {n_scen}, scenarios {n_pairs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    p = sub.add_parser("packets")
    p.add_argument("--max-chars", type=int, default=16000)
    p = sub.add_parser("validate")
    p.add_argument("units", nargs="*")
    p.add_argument("--no-cards", action="store_true")
    p.add_argument("--warnings", action="store_true")
    sub.add_parser("status")
    sub.add_parser("premark")
    args = ap.parse_args()
    {"split": cmd_split, "packets": cmd_packets, "validate": cmd_validate, "status": cmd_status,
     "premark": cmd_premark}[args.cmd](args)


if __name__ == "__main__":
    main()
