"""What a line's script states about amounts and costs: the value head's targets.

Read from the script text whatever the encoding surface (prose has no ``$``
parameters), summed over a chained text's segments. Two masking rules:

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
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from effects.domain.script_text import api_of, params, segments
from effects.domain.state_snapshot import COLORS
from price_predictor.domain.value_objects import ManaCost


class ValueKind(StrEnum):
    COUNT = "count"
    SIGNED = "signed"
    BINARY = "binary"


#: ``(target, kind)`` in the head's output order: append, never reorder.
VALUE_TARGETS: tuple[tuple[str, ValueKind], ...] = (
    ("damage", ValueKind.COUNT),
    ("power_change", ValueKind.SIGNED),
    ("toughness_change", ValueKind.SIGNED),
    ("counters_placed", ValueKind.COUNT),
    ("counters_removed", ValueKind.COUNT),
    ("cards_drawn", ValueKind.COUNT),
    ("mana_w", ValueKind.COUNT),
    ("mana_u", ValueKind.COUNT),
    ("mana_b", ValueKind.COUNT),
    ("mana_r", ValueKind.COUNT),
    ("mana_g", ValueKind.COUNT),
    ("mana_c", ValueKind.COUNT),
    ("mana_generic", ValueKind.COUNT),
    ("cost_taps", ValueKind.BINARY),
    ("cost_sacrifices", ValueKind.BINARY),
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


@dataclass(frozen=True, slots=True)
class ValueTargets:
    """One text's targets, aligned with :data:`VALUE_TARGETS`; ``values[i]``
    is meaningful only where ``mask[i]`` is True."""

    values: tuple[float, ...]
    mask: tuple[bool, ...]


@lru_cache(maxsize=None)
def value_targets(text: str) -> ValueTargets:
    """The value head's targets for one encoding text. Cached per text."""
    parsed = [params(segment) for segment in segments(text)]
    root = parsed[0] if parsed else {}
    values = dict.fromkeys(VALUE_TARGET_NAMES, 0.0)
    masked: set[str] = set()
    # A charm's root states only its choices (FR-005a); its modes carry the
    # amounts on their own option lines.
    if api_of(root) == "Charm" or "Choices" in root:
        masked.update(_AMOUNT_TARGETS)
    else:
        for found in parsed:
            _read_amounts(found, values, masked)
    _read_cost(root.get("Cost"), "SP" in root, values, masked)
    return ValueTargets(
        values=tuple(values[name] for name in VALUE_TARGET_NAMES),
        mask=tuple(name not in masked for name in VALUE_TARGET_NAMES),
    )


def _read_amounts(found: dict[str, str], values: dict[str, float], masked: set[str]) -> None:
    """One segment's stated amounts into ``values``; a non-literal one, or
    one the segment's API leaves unstated, into ``masked``."""

    def add(target: str, raw: str | None) -> None:
        if raw is not None and _LITERAL_RE.fullmatch(raw):
            values[target] += int(raw)
        else:
            masked.add(target)

    for target, keys in _AMOUNT_KEYS.items():
        for key in keys:
            if key in found:
                add(target, found[key])
    api = api_of(found)
    if api in _API_AMOUNTS:
        target, key, default = _API_AMOUNTS[api]
        add(target, found.get(key, default))
    elif api in _COUNTER_UNSTATED:
        masked.update(_COUNTER_TARGETS)


def _read_cost(
    cost: str | None, spell: bool, values: dict[str, float], masked: set[str],
) -> None:
    """The root's ``Cost$`` into the cost targets; a spell's mana is masked."""
    if cost is None:
        masked.update(_COST_TARGETS)
        return
    shards = _COST_GROUP_RE.sub("", cost).split()
    values["cost_taps"] = 1.0 if "T" in shards else 0.0
    values["cost_sacrifices"] = 1.0 if "Sac<" in cost else 0.0
    mana = ManaCost.parse(" ".join(shards))
    if spell or mana is None or mana.has_x:
        masked.update(_MANA_TARGETS)
        return
    for target, amount in zip(_MANA_TARGETS, (
        mana.w, mana.u, mana.b, mana.r, mana.g, mana.colorless_mana,
        mana.generic_mana,
    )):
        values[target] = float(amount)
