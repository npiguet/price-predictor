"""The grammar of a sidecar line's ``script_text``: segments and parameters.

A line's script text is the trait's whole chain (FR-002): the root segment,
then every chained ability, joined by :data:`SEG` and each later segment
opened by its renamed label (``SV1: DB$ Draw``). A segment is ``Key$ value``
parameters separated by ``|``; a printed keyword converts to its bare display
name, which has neither.

Prior art: Forge's own ``FileSection.parseToMap`` splits the same way on the
source script; this module reads the converted chain, which Forge never sees.
"""

from __future__ import annotations

import re

#: Separator between the segments of a chained script text (FR-002).
SEG = "[SEG]"

#: The label that opens a chained segment, in its FR-002a form.
SEGMENT_OPENER = re.compile(r"^\s*(SV\d+):\s*")

#: The parameters whose value names a chain segment (FR-002a).
CHAIN_KEYS = frozenset({"Execute", "SubAbility", "RepeatSubAbility", "ReplaceWith"})

#: The keys whose value names the segment's API type.
_API_KEYS: tuple[str, ...] = ("SP", "AB", "DB")


def segments(text: str) -> list[str]:
    """A chained text's segments, each stripped and without its ``SVn:`` opener."""
    return [SEGMENT_OPENER.sub("", part).strip() for part in text.split(SEG)]


def root(text: str) -> str:
    """The text before the first :data:`SEG`, as written: the trait's own line."""
    return text.split(SEG, 1)[0]


def params(segment: str) -> dict[str, str]:
    """``key -> raw value`` of one segment's ``Key$ value`` parameters.

    Split at the first ``$`` only, so a value carrying its own ``$``-prefix
    (``Count$Valid Creature``) stays whole and reads as a non-literal. The
    first occurrence of a key wins, and a part with no ``$`` is not a
    parameter.
    """
    found: dict[str, str] = {}
    for part in segment.split("|"):
        key, separator, value = part.partition("$")
        key = key.strip()
        if separator and key and key not in found:
            found[key] = value.strip()
    return found


def root_param(text: str | None, name: str) -> str | None:
    """The value of ``name$`` on the root segment; None when absent or empty."""
    if not text:
        return None
    return params(root(text)).get(name) or None


def api_of(found: dict[str, str]) -> str | None:
    """The segment's effect API, whichever of ``SP$``/``AB$``/``DB$`` names it.

    None where none does, or where the one present is empty: an empty API
    names no effect.
    """
    for key in _API_KEYS:
        if found.get(key):
            return found[key]
    return None


def is_bare_keyword(text: str | None) -> bool:
    """Whether ``text`` is a printed keyword's display name rather than a script.

    A printed keyword converts to a line whose script text is the keyword
    itself (White Knight's is exactly ``First Strike``), with no parameter.
    """
    return bool(text) and "$" not in text and "|" not in text
