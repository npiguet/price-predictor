"""``split``: the files it writes, rules.jsonl and cr/ (one markdown file a unit, the glossary
and the index)."""

import json
from collections import defaultdict

from .locations import RULES
from .ordering import unit_sort_key

CR = RULES / "cr"


def write_rules(rules):
    with open(RULES / "rules.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rules:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _unit_markdown(rs, title, chapters, sections):
    """One unit's cr file: its heading, then each rule with its examples."""
    unit, sec, chapter = rs[0]["unit"], rs[0]["section"], rs[0]["chapter"]
    out = [f"# {unit}. {title}", "",
           f"Chapter {chapter} ({chapters[chapter]}), section {sec} ({sections[sec]}).", ""]
    for r in rs:
        out.append(f"**{r['id']}** {r['text']}")
        out += [f"> Example: {ex}" for ex in r["examples"]]
        out.append("")
    return "\n".join(out)


def write_units(rules, unit_titles, chapters, sections):
    """Replace cr/<unit>.md for every unit; the index rows ``(unit, title, rules, chars)``."""
    by_unit = defaultdict(list)
    for r in rules:
        by_unit[r["unit"]].append(r)
    CR.mkdir(parents=True, exist_ok=True)
    for old in CR.glob("*.md"):
        old.unlink()
    index_rows = []
    for unit in sorted(by_unit, key=unit_sort_key):
        rs = by_unit[unit]
        title = unit_titles[unit]
        text = _unit_markdown(rs, title, chapters, sections)
        (CR / f"{unit}.md").write_text(text, encoding="utf-8", newline="\n")
        chars = sum(len(r["text"]) + sum(len(e) for e in r["examples"]) for r in rs)
        index_rows.append((unit, title, len(rs), chars))
    return index_rows


def write_glossary(glossary):
    """cr/glossary.md, one entry a paragraph; the number of entries."""
    gl_out, entry = ["# Glossary", ""], []
    for line in glossary + [""]:
        if line:
            entry.append(line)
        elif entry:
            gl_out.append(f"**{entry[0]}** " + " ".join(entry[1:]))
            gl_out.append("")
            entry = []
    (CR / "glossary.md").write_text("\n".join(gl_out), encoding="utf-8", newline="\n")
    return len(gl_out) // 2


def write_index(rules, index_rows, raw_name):
    idx = ["# Comprehensive Rules index", "",
           f"Source: `rules/raw/{raw_name}`. "
           f"{len(rules)} rules in {len(index_rows)} units.", "",
           "| unit | title | rules | chars | file |", "|---|---|---|---|---|"]
    for unit, title, n, chars in index_rows:
        idx.append(f"| {unit} | {title} | {n} | {chars} | [cr/{unit}.md]({unit}.md) |")
    idx += ["", "Glossary: [cr/glossary.md](glossary.md)."]
    (CR / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8", newline="\n")
