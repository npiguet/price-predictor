"""Choosing the held-out ability texts, and the cards that carry them.

Membership depends on the text's own bytes alone, so adding cards never
reassigns an existing text and a depleted corpus stays valid (FR-130).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from zlib import crc32

from effects.domain.script_text import CHAIN_KEYS, SEG, SEGMENT_OPENER


class HoldoutUnit(StrEnum):
    """What the holdout keys on.

    ``template`` (FR-039) holds out every text sharing a masked template
    together; ``text`` is feature 023's rule, kept so a corpus collected under
    it can be rebuilt exactly.
    """

    TEMPLATE = "template"
    TEXT = "text"

_WHITESPACE = re.compile(r"\s+")


def normalize_script_text(text: str) -> str:
    """The form the hash is taken over."""
    return _WHITESPACE.sub(" ", text).strip()


_DIGITS = re.compile(r"\d+")
_CHAIN_LABEL = re.compile(r"SV\d+")
_CARDNAME = re.compile(r"\bCARDNAME\b")
#: The fixed placeholder every masked span becomes.
MASK = "#"


def _mask_digits(text: str) -> str:
    return _DIGITS.sub(MASK, _CARDNAME.sub(MASK, text))


def _masked_param(param: str) -> str:
    key, dollar, value = param.partition("$")
    if not dollar:
        return _mask_digits(param.strip())
    key = key.strip()
    value = value.strip()
    if key.endswith("Description"):
        masked = MASK
    elif key in CHAIN_KEYS:
        # A chain value's digits are the segment's position, not an amount.
        masked = _label_or_masked(value)
    elif key == "Choices":
        masked = ",".join(_label_or_masked(item) for item in value.split(","))
    else:
        masked = _mask_digits(value)
    return f"{_mask_digits(key)}$ {masked}" if masked else f"{_mask_digits(key)}$"


def _label_or_masked(value: str) -> str:
    """A renamed chain label as written; anything else digit-masked.

    ``convert`` renames only labels to the ``SV<n>`` shape, so the shape is the
    test; a non-chooser ``Choices$`` value is a selector whose digits are an
    amount.
    """
    value = value.strip()
    return value if _CHAIN_LABEL.fullmatch(value) else _mask_digits(value)


def _masked_segment(segment: str) -> str:
    segment = segment.strip()
    opener = SEGMENT_OPENER.match(segment)
    prefix = ""
    if opener is not None:
        prefix = f"{opener.group(1)}: "
        segment = segment[opener.end():]
    return prefix + " | ".join(_masked_param(p) for p in segment.split("|"))


def masked_template(text: str) -> str:
    """The text with everything that varies between printings of one effect masked.

    Every run of digits becomes ``#`` wherever it sits, even inside an
    identifier (``GE3``, ``w_1_1_soldier``), as do ``CARDNAME`` and every
    ``*Description$`` value (FR-039) — except the chain labels FR-002a renames,
    so a reference still names its segment. Keys the holdout only; no encoding
    reads it.
    """
    normalized = normalize_script_text(text)
    return f" {SEG} ".join(
        _masked_segment(segment) for segment in normalized.split(SEG)
    )


def holdout_key(text: str, unit: HoldoutUnit) -> str:
    """What the holdout counts carriers of and hashes, under ``unit``."""
    if unit == HoldoutUnit.TEMPLATE:
        return masked_template(text)
    if unit == HoldoutUnit.TEXT:
        return normalize_script_text(text)
    raise ValueError(
        f"unknown holdout unit {unit!r}; "
        f"expected one of {tuple(known.value for known in HoldoutUnit)}"
    )


def text_is_held_out(text: str, *, permille: int) -> bool:
    """Whether this text's own bytes put it in the holdout.

    ``crc32`` rather than ``hash``, which is salted per process (FR-088a).
    """
    return crc32(normalize_script_text(text).encode("utf-8")) % 1000 < permille


@dataclass(frozen=True, slots=True)
class HoldoutSelection:
    """The held-out texts and every card carrying one.

    ``keys`` are the held-out units themselves: masked templates under
    ``template``, the texts again under ``text``.
    """

    texts: frozenset[str]
    cards: frozenset[str]
    keys: frozenset[str] = frozenset()


def select_holdout(
    texts_by_card: Mapping[str, Iterable[str]],
    *,
    permille: int,
    max_carriers: int,
    unit: HoldoutUnit,
) -> HoldoutSelection:
    """Pick the held-out texts, and the cards depletion has to remove.

    Carriers are counted, and eligibility and the ``crc32`` test applied, per
    :func:`holdout_key` (FR-040); every text whose key is held out is held out.
    """
    normalized: dict[str, set[str]] = {}
    for card, texts in texts_by_card.items():
        normalized[card] = {normalize_script_text(t) for t in texts if t}

    key_of: dict[str, str] = {}
    carriers: Counter[str] = Counter()
    for texts in normalized.values():
        keys = set()
        for text in texts:
            key = key_of.get(text)
            if key is None:
                key = key_of[text] = holdout_key(text, unit)
            keys.add(key)
        carriers.update(keys)

    held_keys = {
        key
        for key, count in carriers.items()
        if count <= max_carriers and text_is_held_out(key, permille=permille)
    }
    held = {text for text, key in key_of.items() if key in held_keys}

    cards = {
        card for card, texts in normalized.items() if texts & held
    }
    return HoldoutSelection(
        texts=frozenset(held), cards=frozenset(cards), keys=frozenset(held_keys),
    )
