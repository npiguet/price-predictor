"""Perturbing a Forge card script into a card that never existed.

Real cards cluster (most damage spells deal 2, 3 or 4), so a variant that
deals 7 breaks a memorized phrase without inventing a mechanic. Perturbations
are kept small: a numeric shift of up to ±3 or a doubling, floored at zero, or
a selector swapped from a whitelist. A signed value (``NumAtt$ +2``, ``NumDef$
-2``) keeps its sign and a magnitude of at least one, so a pump stays a pump and
a debuff a debuff. A deck-building hint (``DeckHints:``, ``DeckNeeds:``,
``DeckHas:``, an ``AIPreference`` SVar) is never perturbed: it changes what the
AI prefers to deck or pay, not what the card does.
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


#: Line kinds that only steer the AI's deck building or cost choice: a swap
#: there changes nothing in a game.
NON_GAMEPLAY_LINE_PREFIXES: tuple[str, ...] = (
    "DeckHints:", "DeckNeeds:", "DeckHas:", "SVar:AIPreference:", "AIPreference:",
)

#: A signed number as a source script writes a pump or a debuff: ``+2``, ``-2``.
_SIGNED_RE = re.compile(r"[+-]\d+")


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
    """Whitelisted numeric parameters of a script line, by key; a signed value
    (``+2``, ``-2``) reads as its number."""
    return {key: int(value) for key, value in _numeric_values(script_text).items()}


def _numeric_values(script_text: str) -> dict[str, str]:
    """Whitelisted numeric parameters of a script line, by key, as written."""
    found: dict[str, str] = {}
    for key, value in _PARAM_RE.findall(script_text):
        stripped = value.strip()
        if key in NUMERIC_PARAM_KEYS and (
            stripped.isdigit() or _SIGNED_RE.fullmatch(stripped)
        ):
            found[key] = stripped
    return found


def _is_gameplay_line(script_text: str) -> bool:
    return not script_text.lstrip().startswith(NON_GAMEPLAY_LINE_PREFIXES)


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


def _perturbed_number(written: str, kind: PerturbationKind, rng: random.Random) -> str:
    """``written`` shifted or doubled. An unsigned count is floored at zero; a
    signed one keeps its sign and a magnitude of at least one, written with
    the sign, so a +2 pump stays a pump and a -2 debuff a debuff."""
    if written[0] not in "+-":
        value = int(written)
        return str(shift(value, rng) if kind == PerturbationKind.SHIFT else double(value))
    sign, magnitude = written[0], int(written[1:])
    changed = shift(magnitude, rng) if kind == PerturbationKind.SHIFT else double(magnitude)
    return f"{sign}{max(1, changed)}"


def perturb(
    script_text: str, rng: random.Random,
) -> tuple[str, Perturbation] | None:
    """Apply one perturbation to ``script_text``.

    Returns the perturbed script and what changed, or None when the line has
    nothing this module knows how to perturb.
    """
    if not _is_gameplay_line(script_text):
        return None
    numbers = _numeric_values(script_text)
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
        original = numbers[key]
        replacement = _perturbed_number(original, kind, rng)
        if replacement == original:
            # A shift of zero is not a variant; try the other perturbation.
            kind = PerturbationKind.DOUBLE
            replacement = _perturbed_number(original, kind, rng)
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


def script_mana_cost(lines: list[str]) -> str | None:
    """The mana cost a Forge script's first ``ManaCost:`` line declares,
    brace-delimited like the converted corpus (``"2 R R"`` -> ``"{2}{R}{R}"``);
    None where it declares none.

    The deck builder reads a cost through ``convert_mana_cost``, which extracts
    ``{...}`` groups, so an unwrapped cost silently parses as costless.
    ``"no cost"`` round-trips to the literal ``ManaCost.parse`` special-cases.
    """
    cost = next(
        (line.split(":", 1)[1] for line in lines if line.startswith("ManaCost:")), "",
    )
    return "".join(f"{{{shard}}}" for shard in cost.split()) or None


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
