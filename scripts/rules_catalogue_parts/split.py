"""``split``: the Comprehensive Rules text into rules.jsonl, cr/<unit>.md and cr/index.md."""

import json
import re
import sys
from collections import defaultdict

from .locations import RULES
from .ordering import SPLIT_SECTIONS, unit_of, unit_sort_key

RULE_RE = re.compile(r"^(\d{3})\.(\d+)([a-z]*)\.? (.*)$")
SECTION_RE = re.compile(r"^(\d{3})\. (.*)$")
CHAPTER_RE = re.compile(r"^(\d)\. (.*)$")
REF_RE = re.compile(r"\b(\d{3}(?:\.\d+[a-z]*)?)\b")


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
