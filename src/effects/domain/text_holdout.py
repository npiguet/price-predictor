"""Choosing the held-out ability texts, and the cards that carry them.

The holdout is keyed on the ability text rather than the card. The model never
reads a card's name — an entity reaches it as computed characteristics plus its
ability lines' embeddings — so the card was always a proxy for the text on it,
and the proxy splits functional reprints: Searing Spear and Lightning Strike
compile to the same script, and a name-keyed holdout would train on one while
holding out the other.

Membership depends on the text's own bytes alone, so adding cards never
reassigns an existing text. That is what lets a depleted collection run freeze
the holdout into the games it collects (FR-130): a selection rule that moved
when Forge gained a set would invalidate the corpus it was collected against.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal
from zlib import crc32

#: What the holdout keys on. ``template`` (gen-2, FR-039) holds out every text
#: sharing a masked template together; ``text`` is feature 023's rule, kept so a
#: corpus collected under it can be rebuilt exactly (spec Story 2 scenario 4).
HoldoutUnit = Literal["template", "text"]
HOLDOUT_UNITS: tuple[str, ...] = ("template", "text")

#: Runs of whitespace differ between scripts that are otherwise the same text,
#: and the difference is formatting rather than mechanism.
_WHITESPACE = re.compile(r"\s+")


def normalize_script_text(text: str) -> str:
    """The form the hash is taken over."""
    return _WHITESPACE.sub(" ", text).strip()


#: The parameters whose value names a chain segment (FR-002a). Their digits are
#: the segment's position, not an amount, so the mask leaves them alone: a text
#: whose reference names its second segment and one naming its first are two
#: different mechanisms.
_CHAIN_KEYS = frozenset({"Execute", "SubAbility", "RepeatSubAbility", "ReplaceWith"})
_SEGMENT_SEPARATOR = "[SEG]"
_SEGMENT_OPENER = re.compile(r"^(SV\d+):\s*")
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
    elif key in _CHAIN_KEYS:
        masked = _label_or_masked(value)
    elif key == "Choices":
        masked = ",".join(_label_or_masked(item) for item in value.split(","))
    else:
        masked = _mask_digits(value)
    return f"{_mask_digits(key)}$ {masked}" if masked else f"{_mask_digits(key)}$"


def _label_or_masked(value: str) -> str:
    """A renamed chain label as written; anything else digit-masked.

    ``convert`` renames a label at an FR-002a position to ``SV<n>`` and nothing
    else to that shape, so the label test is the shape. It matters for
    ``Choices$``: only the chooser APIs (``Charm``, ``GenericChoice``, …) list
    labels there, and on every other API the value is a selector whose digits
    are an amount like any other.
    """
    value = value.strip()
    return value if _CHAIN_LABEL.fullmatch(value) else _mask_digits(value)


def _masked_segment(segment: str) -> str:
    segment = segment.strip()
    opener = _SEGMENT_OPENER.match(segment)
    prefix = ""
    if opener is not None:
        prefix = f"{opener.group(1)}: "
        segment = segment[opener.end():]
    return prefix + " | ".join(_masked_param(p) for p in segment.split("|"))


def masked_template(text: str) -> str:
    """The text with everything that varies between printings of one effect masked.

    Every run of digits becomes ``#`` wherever it sits — standalone (``NumDmg$
    2``), inside an identifier (``GE3``, ``P1P0``, ``w_1_1_soldier``,
    ``Main1``) — as do ``CARDNAME`` and every ``*Description$`` value (FR-039).
    The one exception is the chain labels FR-002a renames: the value of
    ``Execute$``, ``SubAbility$``, ``RepeatSubAbility$`` and ``ReplaceWith$``,
    each ``Choices$`` item and each segment's opening label keep their digits,
    so a reference still names its segment. Letters stay, so mana colours and
    ``P1P1`` against ``M1M1`` stay apart.

    The template keys the holdout only. No encoding reads it.
    """
    normalized = normalize_script_text(text)
    return f" {_SEGMENT_SEPARATOR} ".join(
        _masked_segment(segment) for segment in normalized.split(_SEGMENT_SEPARATOR)
    )


def holdout_key(text: str, unit: HoldoutUnit) -> str:
    """What the holdout counts carriers of and hashes, under ``unit``."""
    if unit == "template":
        return masked_template(text)
    if unit == "text":
        return normalize_script_text(text)
    raise ValueError(f"unknown holdout unit {unit!r}; expected one of {HOLDOUT_UNITS}")


def text_is_held_out(text: str, *, permille: int) -> bool:
    """Whether this text's own bytes put it in the holdout.

    ``crc32`` rather than the built-in ``hash``, which is salted per process for
    strings: a holdout that changed between runs would not match the split the
    corpus was depleted against (FR-088a).
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
    unit: HoldoutUnit = "text",
) -> HoldoutSelection:
    """Pick the held-out texts, and the cards depletion has to remove.

    Under ``template`` (FR-040) carriers are counted per masked template — the
    cards carrying any text with that template — and eligibility and the
    ``crc32`` test apply to the template; every text whose template is held out
    is held out. Under ``text`` the same steps run on the text itself, which is
    feature 023's rule unchanged.
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
