"""Whole-tree invariants of a converted tree: its sidecars read against their text."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from effects.domain.provenance import SidecarError
from effects.infrastructure.sidecar_io import SIDECAR_SUFFIX, SidecarCache

from .model import Line, TallyCheck
from .sidecars import TREES

PLACEHOLDERS = ("CARDNAME", "ABILITY", "{C:")
LEVEL_PREFIX = re.compile(r"level\[(\d+)\]")
FACE_SEPARATOR = "ALTERNATE"
EXAMPLES = 3

EMPTY_PROSE = "keyed-lines-with-empty-prose"
SPELL_WITHOUT_LINE = "spells-of-instants-and-sorceries-with-no-line"
MELD = "meld-halves-without-face-1"
CLASS_LEVEL = "class-level-lines-with-kind-indexed-key"
RAISING = "sidecars-raising-sidecar-error"


def placeholder_name(placeholder: str) -> str:
    return f"keyed-lines-with-placeholder[{placeholder}]"


class TreeTally(TallyCheck):
    NAMES = (
        EMPTY_PROSE,
        *(placeholder_name(p) for p in PLACEHOLDERS),
        SPELL_WITHOUT_LINE,
        MELD,
        CLASS_LEVEL,
        RAISING,
    )


def face_types(text_lines: list[str]) -> list[list[str]]:
    """The words of each face's ``types:`` line, face by face."""
    faces: list[list[str]] = [[]]
    for line in text_lines:
        if line == FACE_SEPARATOR:
            faces.append([])
        elif line.startswith("types:"):
            faces[-1] = line[len("types:"):].split()
    return faces


def prose_of(data_line: dict, text_lines: list[str]) -> str:
    index = data_line["line_index"]
    text = text_lines[index] if 0 <= index < len(text_lines) else ""
    offset = data_line.get("prose_offset")
    return text[offset:] if offset is not None else text.split(": ", 1)[-1]


def tally_keyed_lines(tally: TreeTally, data: dict, text_lines: list[str]) -> None:
    for line in data["lines"]:
        if not line["provenance"]:
            continue
        prose = prose_of(line, text_lines)
        tally.add(EMPTY_PROSE, not prose.strip())
        for placeholder in PLACEHOLDERS:
            tally.add(placeholder_name(placeholder), placeholder in prose)


def tally_spells(tally: TreeTally, data: dict, types: list[list[str]]) -> None:
    """Spell keys of instant and sorcery faces; flagged where the sidecar drops one."""
    declared = [k for line in data["lines"] for k in line["provenance"]]
    for dropped, keys in ((False, declared), (True, data["dropped_keys"])):
        for key in keys:
            on_spell_face = key["face"] < len(types) and {"instant", "sorcery"} & set(
                types[key["face"]]
            )
            if key["trait_kind"] == "spell" and on_spell_face:
                tally.add(SPELL_WITHOUT_LINE, dropped)


def tally_meld(tally: TreeTally, data: dict, text_lines: list[str]) -> None:
    if "layout: meld" not in text_lines:
        return
    keys = [k for line in data["lines"] for k in line["provenance"]] + data["dropped_keys"]
    tally.add(MELD, all(k["face"] != 1 for k in keys))


def declares_built_trait(key: dict) -> bool:
    return key["trait_kind"] != "keyword" or key.get("built") is not None


def tally_class_levels(tally: TreeTally, data: dict, text_lines: list[str]) -> None:
    """Levels 2 and up declare the bare ``Class:N`` keyword key and no key it built: no
    other kind of key, and no ``built`` key of the keyword."""
    for line in data["lines"]:
        if line["line_kind"] != "level":
            continue
        match = LEVEL_PREFIX.match(text_lines[line["line_index"]])
        indexed = any(declares_built_trait(k) for k in line["provenance"])
        tally.add(CLASS_LEVEL, bool(match) and int(match.group(1)) >= 2 and indexed)


def raises_sidecar_error(root: Path, tree: str, data: dict) -> str | None:
    """The error's text where the sidecar fails to load through a ``SidecarCache``."""
    try:
        SidecarCache({tree: root / tree}).get(data["script_file"])
    except SidecarError as error:
        return str(error)
    return None


def tree_lines(root: Path) -> list[Line]:
    """Every tree check's line over the converted tree at ``root``."""
    tally = TreeTally()
    failures: list[str] = []
    for tree in TREES:
        for path in sorted((root / tree).rglob(f"*{SIDECAR_SUFFIX}")):
            data = json.loads(path.read_text(encoding="utf-8"))
            text_path = path.with_name(path.name[: -len(SIDECAR_SUFFIX)] + ".txt")
            text_lines = text_path.read_text(encoding="utf-8").split("\n")
            tally_keyed_lines(tally, data, text_lines)
            tally_spells(tally, data, face_types(text_lines))
            tally_meld(tally, data, text_lines)
            tally_class_levels(tally, data, text_lines)
            failure = raises_sidecar_error(root, tree, data)
            tally.add(RAISING, failure is not None)
            failures.extend(filter(None, [failure]))
    for failure in failures[:EXAMPLES]:
        print(f"  SidecarError: {failure}", file=sys.stderr, flush=True)
    return tally.lines()
