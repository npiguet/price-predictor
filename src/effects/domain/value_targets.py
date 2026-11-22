"""What a line's script states about amounts and costs: the value head's targets.

Read from the script text whatever the encoding surface (prose has no ``$``
parameters), summed over a chained text's segments, one row per line (a text
two lines share is two rows). The masking rules:

- **A line with no script, and a keyword line, state nothing**: every target is
  masked (``Bushido:2`` has no ``$`` parameter to read).
- **A segment the root grants adds nothing** (``Abilities$ SV3``, ``AddTrigger$``
  and the other ``GRANT_KEYS``: Scorn Blade Berserker's backup trigger hands
  an ``AB$ Draw`` to another creature, which is that creature's line's draw,
  not the trigger's): the line states its own amounts, and what it grants is
  left out, not masked, since the head predicts what a line does.
- **A chooser anywhere in the chain masks the amounts** (which option runs is
  not the script's to say), and **a coin flip's branches mask what they
  state** (one of them runs).
- **A hybrid cost masks its mana targets** (``RW`` is one symbol of either
  colour, which a colour count cannot state).
- **A target is masked where the script states no fixed value** (``NumDmg$ X``
  depends on the board, which the head does not read). An amount no segment
  mentions is a stated zero, not a mask.
- **Cost targets come from ``Cost$`` only (FR-058a).** A card's mana cost
  covers the whole card, so a spell line masks its mana targets; its tap and
  sacrifice targets still read from ``Cost$``.

``script_variants.numeric_params`` is not reused: it skips a pump's signed
``NumAtt$ +2``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

from effects.domain.field_types import FieldType
from effects.domain.provenance import KEYWORD_API_TYPE, SidecarLine
from effects.domain.script_text import (
    CHAIN_KEYS,
    CHOOSER_APIS,
    GRANT_KEYS,
    SEG,
    SEGMENT_OPENER,
    api_of,
    params,
    segments,
)
from effects.domain.state_snapshot import COLORS
from price_predictor.domain.value_objects import ManaCost

#: ``(target, type)`` in the head's output order: append, never reorder. Each
#: target is predicted and scored as a per-entity field of its type is.
VALUE_TARGETS: tuple[tuple[str, FieldType], ...] = (
    ("damage", FieldType.COUNT),
    ("power_change", FieldType.SIGNED_DELTA),
    ("toughness_change", FieldType.SIGNED_DELTA),
    ("counters_placed", FieldType.COUNT),
    ("counters_removed", FieldType.COUNT),
    ("cards_drawn", FieldType.COUNT),
    ("mana_w", FieldType.COUNT),
    ("mana_u", FieldType.COUNT),
    ("mana_b", FieldType.COUNT),
    ("mana_r", FieldType.COUNT),
    ("mana_g", FieldType.COUNT),
    ("mana_c", FieldType.COUNT),
    ("mana_generic", FieldType.COUNT),
    ("cost_taps", FieldType.BINARY),
    ("cost_sacrifices", FieldType.BINARY),
)
VALUE_TARGET_NAMES: tuple[str, ...] = tuple(name for name, _ in VALUE_TARGETS)

#: Amount targets any segment states, and the script keys each one sums: a
#: pump's ``NumAtt$``/``NumDef$``, a static's ``AddPower$``/``AddToughness$``.
_AMOUNT_KEYS: dict[str, tuple[str, ...]] = {
    "damage": ("NumDmg",),
    "power_change": ("NumAtt", "AddPower"),
    "toughness_change": ("NumDef", "AddToughness"),
}
#: ``API -> (target, key, Forge's default)`` for keys that mean the target only
#: on that API (``NumCards$`` elsewhere counts discards or mills). With a None
#: default an absent key masks the target.
_API_AMOUNTS: dict[str, tuple[str, str, str | None]] = {
    "Draw": ("cards_drawn", "NumCards", "1"),
    "PutCounter": ("counters_placed", "CounterNum", "1"),
    "PutCounterAll": ("counters_placed", "CounterNum", "1"),
    "RemoveCounter": ("counters_removed", "CounterNum", "1"),
    "RemoveCounterAll": ("counters_removed", "CounterNum", None),
}
_COUNTER_TARGETS: tuple[str, ...] = ("counters_placed", "counters_removed")
#: APIs that move counters in a direction or amount the script does not fix.
_COUNTER_UNSTATED = frozenset({"MoveCounter", "AddOrRemoveCounter", "ReplaceCounter"})
_AMOUNT_TARGETS: tuple[str, ...] = (
    *_AMOUNT_KEYS, *dict.fromkeys(target for target, _, _ in _API_AMOUNTS.values()),
)
_MANA_TARGETS: tuple[str, ...] = (
    *(f"mana_{color.lower()}" for color in COLORS), "mana_generic",
)
_COST_TARGETS: tuple[str, ...] = (*_MANA_TARGETS, "cost_taps", "cost_sacrifices")

_LITERAL_RE = re.compile(r"[+-]?\d+")
_COST_GROUP_RE = re.compile(r"<[^>]*>")
#: A shard payable with either of two colours (``RW``), with two generic mana
#: or a colour (``2W``), or with a colour or life (``WP``): one symbol,
#: whichever is paid, which the cost's mana targets cannot state.
_HYBRID_SHARD = re.compile(r"2[WUBRG]|[WUBRG][WUBRGP]")
#: The parameters naming the segments one outcome of a coin flip runs.
_COIN_KEYS = frozenset({"HeadsSubAbility", "TailsSubAbility", "WinSubAbility", "LoseSubAbility"})
#: The keys that name a trait the line gives away. ``Trigger$`` on a static
#: (an exert's "when you do, +1/+0") and ``PreventionSubAbility$`` (what a
#: shield does when it prevents) name what the root's own effect does.
_GRANTED = GRANT_KEYS - {"Trigger", "PreventionSubAbility"}
#: A grant's list of labels (``SV3,SV4``, ``SV3 & SV4``).
_LISTING = re.compile(r"\s*[,&]\s*")


@dataclass(frozen=True, slots=True)
class ValueTargets:
    """One line's targets, aligned with :data:`VALUE_TARGETS`; ``values[i]``
    is meaningful only where ``mask[i]`` is True."""

    values: tuple[float, ...]
    mask: tuple[bool, ...]


class ValueLine(NamedTuple):
    """The unit the value head trains on: one line. Two lines sharing an
    encoding ``text`` (so one row of ``e``) differ by their ``script``.

    ``script`` is None where the line has none; ``keyword`` marks a keyword
    line, a ``Keyword`` script or a reminder-text fallback.
    """

    text: str
    script: str | None
    keyword: bool

    @classmethod
    def of(cls, text: str, line: SidecarLine | None) -> ValueLine:
        """``text``'s line, None for a text no sidecar line renders."""
        if line is None:
            return cls(text, None, True)
        return cls(text, line.script_text, line.script_api_type == KEYWORD_API_TYPE)

    @property
    def targets(self) -> ValueTargets:
        return value_targets(self.script, keyword=self.keyword)

    def order(self) -> tuple[str, str, bool]:
        """A sort key putting a text's lines together, a missing script first."""
        return (self.text, self.script or "", self.keyword)


class _Amounts:
    """The targets a script states so far."""

    def __init__(self) -> None:
        self.values = dict.fromkeys(VALUE_TARGET_NAMES, 0.0)
        self.masked: set[str] = set()
        #: Every amount target a segment mentioned, stated or not.
        self.touched: set[str] = set()

    def add(self, target: str, raw: str | None) -> None:
        """``raw`` into ``target``: a literal sums, anything else masks."""
        self.touched.add(target)
        if raw is not None and _LITERAL_RE.fullmatch(raw):
            self.values[target] += int(raw)
        else:
            self.masked.add(target)

    def mask(self, targets: Iterable[str]) -> None:
        self.masked.update(targets)

    def targets(self) -> ValueTargets:
        return ValueTargets(
            values=tuple(self.values[name] for name in VALUE_TARGET_NAMES),
            mask=tuple(name not in self.masked for name in VALUE_TARGET_NAMES),
        )


@lru_cache(maxsize=None)
def value_targets(script: str | None, *, keyword: bool = False) -> ValueTargets:
    """The value head's targets for one line's script. Cached per script.

    A line with no script and a keyword line state nothing, so every target is
    masked. A chooser at any depth (a charm, a ``GenericChoice``) masks the
    amounts, which option runs being the player's; a coin flip's branches mask
    the amounts they state, one of them running and the sum of both stating
    neither.
    """
    amounts = _Amounts()
    if script is None or keyword:
        amounts.mask(VALUE_TARGET_NAMES)
        return amounts.targets()
    parts = script.split(SEG)
    parsed = [params(segment) for segment in segments(script)]
    performed = _performed(parsed, _labels(parts))
    if _holds_chooser(parsed, performed):
        amounts.mask(_AMOUNT_TARGETS)
    else:
        _read_chain(parts, parsed, performed, amounts)
    root = parsed[0]
    _read_cost(root.get("Cost"), "SP" in root, amounts)
    return amounts.targets()


def _holds_chooser(parsed: list[dict[str, str]], performed: list[int]) -> bool:
    """Whether the root lists choices (a charm's root states only its choices,
    FR-005a, its modes carry the amounts on their own option lines) or any
    segment it performs is a chooser API."""
    apis = (api_of(parsed[index]) for index in performed)
    return "Choices" in parsed[0] or any(api in CHOOSER_APIS for api in apis)


def _read_chain(
    parts: list[str], parsed: list[dict[str, str]], performed: list[int], amounts: _Amounts,
) -> None:
    """Each segment's amounts into ``amounts``; a coin flip's branches mask
    what they state instead, and a segment the root grants adds nothing."""
    branches = _reached(parsed, _labels(parts), _COIN_KEYS)
    for index in performed:
        if index not in branches:
            _read_amounts(parsed[index], amounts)
            continue
        branch = _Amounts()
        _read_amounts(parsed[index], branch)
        amounts.mask(branch.touched)


def _labels(parts: list[str]) -> dict[str, int]:
    """``SVn`` label -> index of the segment it opens."""
    found = {}
    for index, part in enumerate(parts):
        opener = SEGMENT_OPENER.match(part)
        if opener:
            found[opener.group(1)] = index
    return found


def _performed(parsed: list[dict[str, str]], labels: dict[str, int]) -> list[int]:
    """The indices of the segments the line performs itself: not those a
    segment grants (``Abilities$ SV3`` hands an ability to a creature, whose
    amounts are its own line's), nor what those chain to."""
    granted = _reached(parsed, labels, _GRANTED, follow=_GRANTED)
    return [index for index in range(len(parsed)) if index not in granted]


def _reached(
    parsed: list[dict[str, str]], labels: dict[str, int], keys: frozenset[str] | tuple[str, ...],
    follow: frozenset[str] = frozenset(),
) -> set[int]:
    """The indices of the segments the ``keys`` parameters name, and of what
    those chain to (``CHAIN_KEYS``, and ``follow``)."""
    pending = [index for found in parsed for index in _named(found, keys, labels)]
    reached: set[int] = set()
    while pending:
        index = pending.pop()
        if index not in reached:
            reached.add(index)
            pending += _named(parsed[index], CHAIN_KEYS | follow, labels)
    return reached


def _named(found: dict[str, str], keys: Iterable[str], labels: dict[str, int]) -> list[int]:
    """The indices of the segments the ``keys`` parameters of ``found`` name;
    a value may list several (``SV3,SV4``)."""
    values = (value for key, value in found.items() if key in keys)
    return [labels[label] for value in values for label in _LISTING.split(value) if label in labels]


def _read_amounts(found: dict[str, str], amounts: _Amounts) -> None:
    """One segment's stated amounts into ``amounts``; a non-literal one, or
    one the segment's API leaves unstated, is masked."""
    for target, keys in _AMOUNT_KEYS.items():
        for key in keys:
            if key in found:
                amounts.add(target, found[key])
    api = api_of(found)
    if api in _API_AMOUNTS:
        target, key, default = _API_AMOUNTS[api]
        amounts.add(target, found.get(key, default))
    elif api in _COUNTER_UNSTATED:
        amounts.mask(_COUNTER_TARGETS)


def _read_cost(cost: str | None, spell: bool, amounts: _Amounts) -> None:
    """The root's ``Cost$`` into the cost targets; a spell's mana, an X cost's
    and a hybrid cost's are masked."""
    if cost is None:
        amounts.mask(_COST_TARGETS)
        return
    shards = _COST_GROUP_RE.sub("", cost).split()
    amounts.values["cost_taps"] = 1.0 if "T" in shards else 0.0
    amounts.values["cost_sacrifices"] = 1.0 if "Sac<" in cost else 0.0
    mana = ManaCost.parse(" ".join(shards))
    if spell or mana is None or mana.has_x or _holds_hybrid(shards):
        amounts.mask(_MANA_TARGETS)
        return
    for target, amount in zip(_MANA_TARGETS, (
        mana.w, mana.u, mana.b, mana.r, mana.g, mana.colorless_mana,
        mana.generic_mana,
    )):
        amounts.values[target] = float(amount)


def _holds_hybrid(shards: list[str]) -> bool:
    return any(_HYBRID_SHARD.fullmatch(shard) for shard in shards)
