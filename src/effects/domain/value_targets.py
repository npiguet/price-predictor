"""What a line's script states about amounts and costs: the value head's targets.

The value head (FR-058) reads ``e`` alone and predicts numbers the script
writes down — damage, a pump, counters, cards drawn, and the line's own cost —
so the amounts a text states have to survive the bottleneck rather than being
recoverable only from the board the trunk sees beside them.

Everything is read from the encoding text itself, which is why the targets are
a pure function of it and are cached per text: a chained text (gen-2) is split
on ``[SEG]`` and summed over its segments, and a one-segment text (gen-1's
sidecars, read by the stage-0 noise pilot) is simply one segment.

Two masking rules carry the whole design:

- **A target is masked where the script states no fixed value.** ``NumDmg$ X``
  or ``NumDmg$ Count$Valid Creature`` in any segment masks damage, because the
  number depends on the board and the head reads no board. An amount no
  segment mentions is a stated zero, not a mask: a draw spell deals no damage.
  An API that reads its amount with a default reads it the way Forge does —
  ``Draw`` with no ``NumCards$`` draws one, ``PutCounter`` with no
  ``CounterNum$`` puts one — and moving counters, or choosing whether to add
  or remove them, masks both counter targets, since the script fixes neither.
- **Cost targets come from ``Cost$`` only (FR-058a).** A card's mana cost covers
  the whole card and no line's text carries it, so a spell line masks its mana
  targets even where an additional cost puts mana in ``Cost$``. Its tap and
  sacrifice targets still read from ``Cost$``.

Prior art: ``ManaCost.parse`` (``price_predictor.domain.value_objects``) reads
the mana shards and skips the rest. ``script_variants.numeric_params`` is not
reused for the amounts, because it accepts only unsigned or negative literals
and a pump's ``NumAtt$ +2`` — the common spelling — is a literal it skips.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from price_predictor.domain.value_objects import ManaCost

SEGMENT_SEPARATOR = "[SEG]"


class ValueKind(StrEnum):
    COUNT = "count"
    SIGNED = "signed"
    BINARY = "binary"


#: ``(target, kind)`` in the head's output order. Appending is safe; reordering
#: is not, because the order is the head's layout.
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
#: ``API -> (target, key, Forge's default)`` for amounts whose key means that
#: target only on that API: ``NumCards$`` elsewhere counts cards discarded,
#: milled or dug, and ``CounterNum$`` sits on a few unrelated APIs too. A
#: default of None has none, and an absent key masks the target.
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
    "mana_w", "mana_u", "mana_b", "mana_r", "mana_g", "mana_c", "mana_generic",
)
_COST_TARGETS: tuple[str, ...] = (*_MANA_TARGETS, "cost_taps", "cost_sacrifices")

#: The keys whose value names the segment's API type.
_API_KEYS: tuple[str, ...] = ("SP", "AB", "DB")
_LABEL_RE = re.compile(r"^\s*SV\d+:\s*")
_LITERAL_RE = re.compile(r"[+-]?\d+")
_COST_GROUP_RE = re.compile(r"<[^>]*>")


@dataclass(frozen=True, slots=True)
class ValueTargets:
    """One text's targets, aligned with :data:`VALUE_TARGETS`.

    ``values[i]`` is meaningful only where ``mask[i]`` is True.
    """

    values: tuple[float, ...]
    mask: tuple[bool, ...]

    def get(self, name: str) -> float | None:
        """The target by name, or None where it is masked."""
        index = VALUE_TARGET_NAMES.index(name)
        return self.values[index] if self.mask[index] else None


def segments(text: str) -> list[str]:
    """A chained text's segments, each without its ``SVn:`` opener."""
    return [_LABEL_RE.sub("", part).strip() for part in text.split(SEGMENT_SEPARATOR)]


def params(segment: str) -> dict[str, str]:
    """``key -> raw value`` of one segment's ``Key$ value`` parameters.

    Split at the first ``$`` only, so a value carrying its own ``$``-prefix
    (``Count$Valid Creature``) stays whole and reads as a non-literal.
    """
    found: dict[str, str] = {}
    for part in segment.split("|"):
        key, separator, value = part.partition("$")
        key = key.strip()
        if separator and key and key not in found:
            found[key] = value.strip()
    return found


def api_of(found: dict[str, str]) -> str | None:
    """The segment's effect API, whichever of ``SP$``/``AB$``/``DB$`` names it.

    None where none does, or where the one present is empty: an empty API
    names no effect.
    """
    for key in _API_KEYS:
        if found.get(key):
            return found[key]
    return None


@lru_cache(maxsize=None)
def value_targets(text: str) -> ValueTargets:
    """The value head's targets for one encoding text. Cached per text."""
    parsed = [params(segment) for segment in segments(text)]
    root = parsed[0] if parsed else {}
    values = dict.fromkeys(VALUE_TARGET_NAMES, 0.0)
    masked: set[str] = set()

    def add(target: str, raw: str | None) -> None:
        if raw is not None and _LITERAL_RE.fullmatch(raw):
            values[target] += int(raw)
        else:
            masked.add(target)

    # A charm's root states only its choices (FR-005a); its modes carry the
    # amounts on their own option lines.
    if api_of(root) == "Charm" or "Choices" in root:
        masked.update(_AMOUNT_TARGETS)
    else:
        for found in parsed:
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

    cost = root.get("Cost")
    if cost is None:
        masked.update(_COST_TARGETS)
    else:
        shards = _COST_GROUP_RE.sub("", cost).split()
        values["cost_taps"] = 1.0 if "T" in shards else 0.0
        values["cost_sacrifices"] = 1.0 if "Sac<" in cost else 0.0
        mana = ManaCost.parse(" ".join(shards))
        if "SP" in root or mana is None or mana.has_x:
            masked.update(_MANA_TARGETS)
        else:
            for target, amount in zip(_MANA_TARGETS, (
                mana.w, mana.u, mana.b, mana.r, mana.g, mana.colorless_mana,
                mana.generic_mana,
            )):
                values[target] = float(amount)

    return ValueTargets(
        values=tuple(values[name] for name in VALUE_TARGET_NAMES),
        mask=tuple(name not in masked for name in VALUE_TARGET_NAMES),
    )
