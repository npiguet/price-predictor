"""Perturbing a Forge card script into a card that never existed.

The corpus only ever contains texts real cards print, and real cards cluster: a
model can learn "deal 3 damage" as a memorized phrase because almost every
damage spell in the set deals 2, 3 or 4. A variant that deals 7 breaks that
correlation without inventing a mechanic — the script is a real Forge script,
Forge resolves it by its ordinary rules, and only the parameter differs.

Two perturbations, both deliberately small:

- **a numeric parameter shifted by up to ±3, or doubled**, floored at zero in
  either case. Not scaled arbitrarily, because a 40-damage spell would be a
  different card rather than the same card with a different number;
- **a selector swapped from a checked-in whitelist**, so the swap always
  produces a selector Forge recognizes. A generated selector would mostly
  produce cards that fail to load.

A variant exists on the **script surface only** and is never converted to prose:
there is no oracle text for a card nobody printed, and inventing one would put
text in the corpus that no card has.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

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
    "Amount", "Num", "NumCounters", "NumTimes", "TgtPrompt", "PowerBonus",
    "ToughnessBonus",
})

_PARAM_RE = re.compile(r"(\w+)\$\s*([^|]+?)(?=\s*\||\s*$)")


class PerturbationKind:
    SHIFT = "numeric-shift"
    DOUBLE = "numeric-double"
    SELECTOR = "selector-swap"


@dataclass(frozen=True, slots=True)
class Perturbation:
    """One change to one script line."""

    kind: str
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
    """Move a number by up to ±3, floored at zero.

    Floored because a negative count is not a smaller effect, it is a script
    Forge will reject — and a rejected script is a variant that never reaches a
    game.
    """
    delta = rng.randint(-MAX_NUMERIC_SHIFT, MAX_NUMERIC_SHIFT)
    return max(0, value + delta)


def double(value: int) -> int:
    """Double a number, floored at zero.

    Doubling rather than scaling arbitrarily: it moves the value off the
    corpus's cluster while keeping the card recognisably the same card.
    """
    return max(0, value * 2)


def perturb(
    script_text: str, rng: random.Random,
) -> tuple[str, Perturbation] | None:
    """Apply one perturbation to ``script_text``.

    Returns the perturbed script and what changed, or None when the line has
    nothing this module knows how to perturb — most script lines do not, and
    skipping them is correct.
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

    Derived from the source card so ``variant_of`` and the file name agree, and
    suffixed so several variants of one card do not collide.

    Suffixed with words rather than parentheses because Forge reads a
    parenthesised suffix on a deck-list line as a set code, and because neither
    filename sanitizer strips brackets — the file would be named
    ``lightning_bolt_(variant_0).txt``.
    """
    return f"{card_name} Variant {index}"
