"""A card's lines as ``[ABILITY]`` slots.

Each ``[ABILITY]`` slot is one line of its card with the line's flags (root
spec § 9.2): every line the card carries, in force or not, granted lines
included. A keyword line the record names no script for, such as a keyword a
resolved effect granted or a keyword counter, reads its keyword's reminder
text through a :func:`keyword_line_key`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace

from effects.domain.input_layout import (
    _SCALAR_WIDTH,
    _push_scalar,
)
from effects.domain.input_vocabularies import (
    _KEYWORD_COUNTERS,
)
from effects.domain.keyword_names import normalize_keyword, overlay_keyword_set
from effects.domain.provenance import ProvenanceKey, TraitKind
from effects.domain.state_snapshot import (
    CardLine,
    EntityState,
    LineGrant,
    LineKind,
)

# ── the ability slots ─────────────────────────────────────────────────────

#: The tree of a :func:`keyword_line_key`: no folder, as ``counterfactual``
#: is none; the batcher reads the key's text as its keyword's reminder text.
KEYWORD_LINE_TREE = "keyword-line"


def keyword_line_key(fragment: str) -> ProvenanceKey:
    """The key a keyword line with no script of its own is read through: a
    keyword a resolved effect granted, a keyword counter's keyword. Its text
    is the keyword's reminder text (root spec § 9.2)."""
    return ProvenanceKey(
        f"{KEYWORD_LINE_TREE}/{fragment}", 0, TraitKind.KEYWORD.value, 0,
    )


def keyword_line_fragment(key: ProvenanceKey) -> str | None:
    """The keyword fragment a :func:`keyword_line_key` reads; None for any
    other key."""
    tree, _, fragment = key.script_file.partition("/")
    return fragment if tree == KEYWORD_LINE_TREE else None


@dataclass(frozen=True, slots=True)
class SlotLine:
    """One line of a card as an ``[ABILITY]`` slot reads it: its key and its
    line flags (root spec § 9.2)."""

    key: ProvenanceKey
    #: The keywords the slot states, in the overlay's spelling: the keyword a
    #: gen-3 keyword line is, every one of them where several lines share the
    #: key (a static granting two keywords).
    keywords: frozenset[str] = frozenset()
    #: In force: its face is up, its static's condition holds.
    active: bool = True
    #: Not printed on the card.
    granted: bool = False
    #: Why a granted line is granted (:class:`LineGrant`); None on a printed
    #: line and where the record does not say.
    grant: str | None = None
    #: How many times the spell paid this line's cost; None where it did not.
    cast_with: int | None = None
    #: The ``Choices$`` positions a modal line chose.
    chosen: tuple[int, ...] = ()

    @property
    def by_effect(self) -> bool:
        """Granted by a resolved effect (a pump, an animation, an emblem's own
        line), the line flag: why it is granted, not how long it lasts."""
        return self.grant == LineGrant.EFFECT

    def merged(self, other: SlotLine) -> SlotLine:
        """Two lines of one key as one slot: in force where either is, granted
        where both are, for the reason both share."""
        cast_with = max((n for n in (self.cast_with, other.cast_with) if n is not None),
                        default=None)
        return SlotLine(
            key=self.key, keywords=self.keywords | other.keywords,
            active=self.active or other.active, granted=self.granted and other.granted,
            grant=self.grant if self.grant == other.grant else None, cast_with=cast_with,
            chosen=tuple(dict.fromkeys((*self.chosen, *other.chosen))),
        )


def _as_is(key: ProvenanceKey) -> tuple[ProvenanceKey, ...]:
    return (key,)


def slot_lines(
    entity: EntityState,
    board_keys: Callable[[ProvenanceKey], tuple[ProvenanceKey, ...]] = _as_is,
) -> tuple[SlotLine, ...]:
    """A card's lines as ``[ABILITY]`` slots, each key once, in line order.

    A line's key becomes one slot per key ``board_keys`` gives it
    (:meth:`LineLookup.board_keys`): a saga's bare keyword key, one slot per
    chapter line, each with the line's flags. Without it every key is one slot.

    Gen-3: every line, in force or not (a face not up, an unmet static, the
    real face of a face-down card are slots flagged inactive), with the
    line's own ``grant``; a granted line of a record written before
    ``grant`` existed reads none, as nothing else says why it is granted. A
    line with no key is read through :func:`keyword_line_key` where it is a
    keyword line, and has no slot otherwise. A keyword counter whose keyword
    no line carries adds a keyword line granted by the counter.
    """
    own = _gen3_slot_lines(entity.lines)
    merged = _merge_by_key({}, _expanded(own, board_keys))
    return tuple(_merge_by_key(merged, _keyword_counter_lines(entity, merged)).values())


def _expanded(lines: Iterable[SlotLine],
              board_keys: Callable[[ProvenanceKey], tuple[ProvenanceKey, ...]],
              ) -> Iterator[SlotLine]:
    """Each line once per key ``board_keys`` gives its key."""
    for line in lines:
        keys = board_keys(line.key)
        yield from (line if key == line.key else replace(line, key=key) for key in keys)


def _merge_by_key(merged: dict[ProvenanceKey, SlotLine],
                  lines: Iterable[SlotLine]) -> dict[ProvenanceKey, SlotLine]:
    """``merged`` with ``lines`` added, a key already there merged into its slot."""
    for line in tuple(lines):
        earlier = merged.get(line.key)
        merged[line.key] = line if earlier is None else earlier.merged(line)
    return merged


def _gen3_slot_lines(lines: tuple[CardLine, ...]) -> Iterator[SlotLine]:
    for line in lines:
        key = _slot_key(line)
        if key is not None:
            yield SlotLine(
                key, keywords=overlay_keyword_set(line.keyword), active=line.in_force,
                granted=line.granted, grant=line.grant, cast_with=line.cast_with,
                chosen=line.chosen or (),
            )


def _slot_key(line: CardLine) -> ProvenanceKey | None:
    """The key a gen-3 line's slot reads: its own, a keyword line's reminder
    line, or None for a line with neither."""
    if line.keys:
        return line.keys[0]
    if line.kind == LineKind.KEYWORD and line.fragment:
        return keyword_line_key(line.fragment)
    return None


def _keyword_counter_lines(entity: EntityState,
                           merged: dict[ProvenanceKey, SlotLine]) -> Iterator[SlotLine]:
    """A keyword line per keyword counter whose keyword no line carries."""
    carried = set().union(*(line.keywords for line in merged.values()))
    for name, count in entity.counters.items():
        keyword = normalize_keyword(name)
        if count > 0 and keyword in _KEYWORD_COUNTERS - carried:
            carried.add(keyword)
            yield SlotLine(keyword_line_key(name), keywords=overlay_keyword_set(keyword),
                           granted=True, grant=LineGrant.COUNTER.value)


def ability_features(line: SlotLine, *, chosen: bool = False) -> tuple[float, ...]:
    """An ``[ABILITY]`` slot's line flags: active, granted, granted by an
    effect, cast with (a flag and the count), and chosen, which a charm's
    mode slot sets where the spell chose that mode."""
    out = [
        1.0 if line.active else 0.0,
        1.0 if line.granted else 0.0,
        1.0 if line.by_effect else 0.0,
        0.0 if line.cast_with is None else 1.0,
    ]
    _push_scalar(out, line.cast_with or 0)
    out.append(1.0 if chosen else 0.0)
    return tuple(out)


ABILITY_FEATURE_WIDTH = 4 + _SCALAR_WIDTH + 1
