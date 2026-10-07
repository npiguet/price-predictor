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

#: Amount targets and the script keys each one sums.
_AMOUNT_KEYS: dict[str, tuple[str, ...]] = {
    "damage": ("NumDmg",),
    "power_change": ("NumAtt", "PowerBonus"),
    "toughness_change": ("NumDef", "ToughnessBonus"),
    "counters_placed": ("CounterNum",),
    "cards_drawn": ("NumCards",),
}
#: ``NumCards$`` means cards drawn only on a ``Draw`` segment; elsewhere it
#: counts cards discarded, milled, revealed or dug.
_DRAW_ONLY = frozenset({"cards_drawn"})
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


def _api_of(found: dict[str, str]) -> str | None:
    for key in _API_KEYS:
        if key in found:
            return found[key]
    return None


@lru_cache(maxsize=None)
def value_targets(text: str) -> ValueTargets:
    """The value head's targets for one encoding text. Cached per text."""
    parsed = [params(segment) for segment in segments(text)]
    root = parsed[0] if parsed else {}
    values = dict.fromkeys(VALUE_TARGET_NAMES, 0.0)
    masked: set[str] = set()

    # A charm's root states only its choices (FR-005a); its modes carry the
    # amounts on their own option lines.
    if _api_of(root) == "Charm" or "Choices" in root:
        masked.update(_AMOUNT_KEYS)
    else:
        for found in parsed:
            for target, keys in _AMOUNT_KEYS.items():
                if target in _DRAW_ONLY and _api_of(found) != "Draw":
                    continue
                for key in keys:
                    raw = found.get(key)
                    if raw is None:
                        continue
                    if _LITERAL_RE.fullmatch(raw):
                        values[target] += int(raw)
                    else:
                        masked.add(target)

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
