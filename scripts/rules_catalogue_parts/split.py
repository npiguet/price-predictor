"""``split``: the Comprehensive Rules text into rules.jsonl, cr/<unit>.md and cr/index.md."""

import re
import sys

from .cr_files import write_glossary, write_index, write_rules, write_units
from .locations import RULES
from .ordering import SPLIT_SECTIONS, unit_of

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


class _Parser:
    """The rules, chapters and sections of the body of the Comprehensive Rules text."""

    def __init__(self):
        self.rules, self.chapters, self.sections = [], {}, {}

    def feed(self, line):
        if not line:
            return
        for read in (self._rule, self._section, self._chapter):
            if read(line):
                return
        self._continuation(line)

    def _rule(self, line):
        m = RULE_RE.match(line)
        if not m:
            return False
        sec, num, letters, text = m.groups()
        rid = f"{sec}.{num}{letters}"
        parent = f"{sec}.{num}" if letters else sec
        refs = sorted({r for r in REF_RE.findall(text) if r != rid})
        self.rules.append({"id": rid, "unit": unit_of(sec, num), "section": sec,
                           "chapter": sec[0], "parent": parent, "text": text, "examples": [],
                           "refs": refs})
        return True

    def _section(self, line):
        m = SECTION_RE.match(line)
        if m:
            self.sections[m.group(1)] = m.group(2)
        return bool(m)

    def _chapter(self, line):
        m = CHAPTER_RE.match(line)
        if m and len(line) < 60:
            self.chapters[m.group(1)] = m.group(2)
            return True
        return False

    def _continuation(self, line):
        if not self.rules:
            raise ValueError(f"text before the first rule: {line!r}")
        if line.startswith("Example:"):
            self.rules[-1]["examples"].append(line[len("Example:"):].strip())
        else:
            self.rules[-1]["text"] += "\n" + line


def _unit_titles(rules, sections):
    """A split section's unit title is its first rule's text,
    e.g. "Deathtouch" from "702.2. Deathtouch"."""
    unit_titles = {}
    for r in rules:
        if r["unit"] not in unit_titles:
            in_split = r["section"] in SPLIT_SECTIONS
            unit_titles[r["unit"]] = (
                r["text"].split("\n")[0][:80] if in_split else sections[r["section"]]
            )
    return unit_titles


def _body_and_glossary():
    lines = [clean(raw) for raw in raw_file().read_text(encoding="utf-8-sig").splitlines()]
    starts = [i for i, line in enumerate(lines) if line == "1. Game Concepts"]
    glossaries = [i for i, line in enumerate(lines) if line == "Glossary"]
    credits = [i for i, line in enumerate(lines) if line == "Credits"]
    return lines[starts[1]:glossaries[1]], lines[glossaries[1] + 1:credits[1]]


def cmd_split(_args):
    body, glossary = _body_and_glossary()
    parsed = _Parser()
    for line in body:
        parsed.feed(line)
    rules = parsed.rules
    unit_titles = _unit_titles(rules, parsed.sections)
    write_rules(rules)
    index_rows = write_units(rules, unit_titles, parsed.chapters, parsed.sections)
    n_glossary = write_glossary(glossary)
    write_index(rules, index_rows, raw_file().name)
    print(f"{len(rules)} rules, {len(index_rows)} units, {n_glossary} glossary entries")
