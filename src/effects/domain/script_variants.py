"""Perturbing a Forge card script into a card that never existed.

Real cards cluster (most damage spells deal 2, 3 or 4), so a variant that
deals 7 breaks a memorized phrase without inventing a mechanic. Perturbations
are kept small: a numeric shift of up to ±3 or a doubling, floored at zero, or
a selector swapped from a whitelist.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import StrEnum

#: Selectors safe to swap between. Every one is a real Forge restriction that
#: composes with any card type, so a swap yields a script Forge can load.
SELECTOR_WHITELIST: tuple[str, ...] = (
    "Creature", "Artifact", "Enchantment", "Land", "Permanent",
    "Creature.attacking", "Creature.blocking", "Creature.tapped",
    "Creature.untapped", "Creature.YouCtrl", "Creature.OppCtrl",
    "Permanent.YouCtrl", "Permanent.OppCtrl", "Card.YouOwn",
)

#: How far a numeric shift may move a parameter.
MAX_NUMERIC_SHIFT = 3

#: Script parameter keys whose value is a number worth perturbing. Deliberately
#: a whitelist: a shifted ``CounterType`` or ``ValidCards`` would produce a
#: script that fails to load rather than a card that plays differently.
NUMERIC_PARAM_KEYS: frozenset[str] = frozenset({
    "NumDmg", "NumCards", "LifeAmount", "CounterNum", "NumAtt", "NumDef",
    "Amount", "Num", "NumCounters", "NumTimes", "PowerBonus", "ToughnessBonus",
})

#: A ``Key$ value`` pair on a Forge *source* line, not a converted chain, so
#: not :func:`effects.domain.script_text.params`: a source line opens with its
#: kind (``A:SP$``, ``SVar:DBDraw:DB$``, ``DeckHints:Type$``), and the key is
#: the word just before ``$`` -- the form :func:`_apply` rewrites by.
_PARAM_RE = re.compile(r"(\w+)\$\s*([^|]+?)(?=\s*\||\s*$)")


class PerturbationKind(StrEnum):
    SHIFT = "numeric-shift"
    DOUBLE = "numeric-double"
    SELECTOR = "selector-swap"


@dataclass(frozen=True, slots=True)
class Perturbation:
    """One change to one script line."""

    kind: PerturbationKind
    key: str
    original: str
    replacement: str

    def describe(self) -> str:
        return f"{self.kind}: {self.key}${self.original} -> {self.replacement}"


def numeric_params(script_text: str) -> dict[str, int]:
    """Whitelisted numeric parameters of a script line, by key."""
    found: dict[str, int] = {}
    for key, value in _PARAM_RE.findall(script_text):
        stripped = value.strip()
        if key in NUMERIC_PARAM_KEYS and stripped.lstrip("-").isdigit():
            found[key] = int(stripped)
    return found


def selector_params(script_text: str) -> dict[str, str]:
    """Parameters whose value is a selector this module knows how to swap."""
    found: dict[str, str] = {}
    for key, value in _PARAM_RE.findall(script_text):
        stripped = value.strip()
        if stripped in SELECTOR_WHITELIST:
            found[key] = stripped
    return found


def shift(value: int, rng: random.Random) -> int:
    """Move a number by up to ±3, floored at zero (Forge rejects a negative)."""
    delta = rng.randint(-MAX_NUMERIC_SHIFT, MAX_NUMERIC_SHIFT)
    return max(0, value + delta)


def double(value: int) -> int:
    """Double a number, floored at zero."""
    return max(0, value * 2)


def perturb(
    script_text: str, rng: random.Random,
) -> tuple[str, Perturbation] | None:
    """Apply one perturbation to ``script_text``.

    Returns the perturbed script and what changed, or None when the line has
    nothing this module knows how to perturb.
    """
    numbers = numeric_params(script_text)
    selectors = selector_params(script_text)
    choices: list[str] = []
    if numbers:
        choices += [PerturbationKind.SHIFT, PerturbationKind.DOUBLE]
    if selectors:
        choices.append(PerturbationKind.SELECTOR)
    if not choices:
        return None

    kind = rng.choice(choices)
    if kind == PerturbationKind.SELECTOR:
        key = rng.choice(sorted(selectors))
        original = selectors[key]
        alternatives = [s for s in SELECTOR_WHITELIST if s != original]
        replacement = rng.choice(alternatives)
    else:
        key = rng.choice(sorted(numbers))
        original = str(numbers[key])
        value = numbers[key]
        replacement = str(
            shift(value, rng) if kind == PerturbationKind.SHIFT else double(value)
        )
        if replacement == original:
            # A shift of zero is not a variant; try the other perturbation.
            replacement = str(double(value))
            kind = PerturbationKind.DOUBLE
            if replacement == original:
                return None

    perturbation = Perturbation(kind, key, original, replacement)
    return _apply(script_text, perturbation), perturbation


def _apply(script_text: str, perturbation: Perturbation) -> str:
    """Rewrite one ``Key$ Value`` pair, leaving the rest of the line alone."""
    pattern = re.compile(
        rf"({re.escape(perturbation.key)}\$\s*){re.escape(perturbation.original)}"
        rf"(?=\s*\||\s*$)"
    )
    return pattern.sub(rf"\g<1>{perturbation.replacement}", script_text, count=1)


def variant_name(card_name: str, index: int) -> str:
    """The name a variant script is filed under.

    Suffixed with words rather than parentheses: Forge reads a parenthesised
    suffix on a deck-list line as a set code, and no filename sanitizer strips
    brackets.
    """
    return f"{card_name} Variant {index}"


def script_card_name(lines: list[str]) -> str | None:
    """The card name a Forge script's first ``Name:`` line declares."""
    for line in lines:
        if line.startswith("Name:"):
            return line[len("Name:"):].strip()
    return None


def renamed_script(lines: list[str], name: str) -> list[str]:
    """Give the variant its own name so Forge files it as a separate card."""
    return [
        f"Name:{name}" if line.startswith("Name:") else line for line in lines
    ]


def is_multi_face(source_lines: list[str]) -> bool:
    """Whether a Forge script declares more than one card face.

    Such a card is never varied: renaming rewrites only the first ``Name:``
    line, and faces that disagree make ``StaticData`` throw while building the
    whole card database, killing every Forge startup that sees the script.
    """
    return any(
        line.strip().startswith("ALTERNATE") for line in source_lines
    )
