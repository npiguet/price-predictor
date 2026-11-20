"""The spans of a script line the script surface's grammar skips (FR-009,
FR-009a): values kept as one token, and ``TokenScript$`` values split at
underscores.

Found on the raw text, before any splitting, since which parameter a value
belongs to is lost once it is split.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from effects.domain.script_text import CHAIN_KEYS, SEG

#: Parameters whose value items stay one token on the script surface (FR-009).
_WHOLE_VALUE_RE = re.compile(
    r"(?<![A-Za-z])"
    rf"(?:{'|'.join((*sorted(CHAIN_KEYS), 'CounterType'))})"
    r"\$[ \t]*([^\s|]+)"
)
#: ``Choices$`` names chain labels only on a chooser API; elsewhere it is a
#: selector and splits like any other value.
_CHOICES_RE = re.compile(r"(?<![A-Za-z])Choices\$[ \t]*([^\s|]+)")
_CHOOSER_API_RE = re.compile(
    r"(?<![A-Za-z])(?:SP|DB|AB)\$[ \t]*"
    r"(?:Charm|GenericChoice|AssignGroup|VillainousChoice|Vote)(?![A-Za-z])"
)
#: The label opening a segment: at the start (an option line) or after ``[SEG]``.
_SEGMENT_OPENER_RE = re.compile(rf"(?:^|{re.escape(SEG)})[ \t]*(SV\d+)(?=:)")
#: A renamed chain label wherever it stands. The converter follows more
#: parameters than ``CHAIN_KEYS`` lists (``AddTrigger$``, ``Trigger$``,
#: ``ResultSubAbilities$`` after ``range:``... see ``TraitScript.java`` and
#: ``GrantedTraits.java`` in forge-connector), and a label after any of them
#: is as much one token as after ``Execute$``.
_CHAIN_LABEL_RE = re.compile(r"(?<![A-Za-z0-9_])SV\d+(?![A-Za-z0-9_])")
_TOKEN_SCRIPT_VALUE_RE = re.compile(r"(?<![A-Za-z])TokenScript\$[ \t]*([^\s|]+)")


class Region(NamedTuple):
    """A span the script grammar skips: kept ``whole`` (FR-009), or split at
    underscores (FR-009a). Sorts underscore-first at an equal span."""

    start: int
    end: int
    whole: bool


def script_regions(text: str) -> list[Region]:
    """The ordered, non-overlapping regions of ``text`` the grammar skips."""
    regions = [
        *_whole_value_regions(text),
        *_choices_regions(text),
        *(
            Region(match.start(1), match.end(1), whole=True)
            for match in _SEGMENT_OPENER_RE.finditer(text)
        ),
        *(
            Region(match.start(), match.end(), whole=True)
            for match in _CHAIN_LABEL_RE.finditer(text)
        ),
        *(
            Region(match.start(1), match.end(1), whole=False)
            for match in _TOKEN_SCRIPT_VALUE_RE.finditer(text)
        ),
    ]
    regions.sort()
    return _without_overlaps(regions)


def _whole_value_regions(text: str) -> list[Region]:
    regions: list[Region] = []
    for match in _WHOLE_VALUE_RE.finditer(text):
        regions.extend(_item_regions(match))
    return regions


def _choices_regions(text: str) -> list[Region]:
    """The ``Choices$`` items of the segments on a chooser API."""
    regions: list[Region] = []
    for match in _CHOICES_RE.finditer(text):
        if _CHOOSER_API_RE.search(_segment_at(text, match.start())):
            regions.extend(_item_regions(match))
    return regions


def _without_overlaps(regions: list[Region]) -> list[Region]:
    """``regions``, sorted, minus each one that starts inside the one kept
    before it."""
    ordered: list[Region] = []
    for region in regions:
        if not (ordered and region.start < ordered[-1].end):
            ordered.append(region)
    return ordered


def _item_regions(match: re.Match) -> list[Region]:
    """One whole region per non-empty item of a comma list in group 1."""
    regions: list[Region] = []
    offset = match.start(1)
    for item in match.group(1).split(","):
        if item:
            regions.append(Region(offset, offset + len(item), whole=True))
        offset += len(item) + 1
    return regions


def _segment_at(text: str, position: int) -> str:
    """The ``[SEG]``-delimited segment of ``text`` holding ``position``."""
    start = text.rfind(SEG, 0, position)
    start = 0 if start < 0 else start + len(SEG)
    end = text.find(SEG, position)
    return text[start:] if end < 0 else text[start:end]
