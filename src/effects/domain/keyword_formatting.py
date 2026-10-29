"""Filling a keyword's reminder template the way Forge fills it (FR-017, FR-018).

Forge hands a keyword line's values (``Ward:2``, ``TypeCycling:Basic:1 B``) to
the keyword's ``Keyword.type`` class, which formats them into the template: a
cost becomes ``{1}{B}``, ward's cost "pays {2}". Filling the raw values in
would put script syntax into an English sentence, so each class's ``parse``
and ``formatReminderText`` is mirrored here, keyed by the class's simple name
(the definition's ``formatter``). An unknown formatter fills the raw values in
order; an unfilled placeholder is removed.
"""

from __future__ import annotations

import re
from collections.abc import Callable

#: ``%s``, ``%d`` and their explicit-index forms ``%1$s``, ``%2$d``.
_SPECIFIER_RE = re.compile(r"%(?:(\d+)\$)?([sd])")
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:])")

#: A bracketed cost part (``Sac<1/Creature>``) or a bare shard.
_COST_PART_RE = re.compile(r"\w+<[^>]*>|\S+")
_MANA_SHARD_RE = re.compile(r"^(\d+|[XYZ]|[WUBRGCS](/[WUBRGCP2])*|2/[WUBRG])$")

_COLORS = {"white", "blue", "black", "red", "green", "colorless"}
#: After Forge's ``CardType.isACardType``. Not the effect head's frozen
#: ``CARD_TYPES``, which a change here must not move.
_CARD_TYPES = {
    "artifact", "battle", "creature", "enchantment", "instant", "kindred",
    "land", "planeswalker", "sorcery", "tribal",
}
#: ``CardType.CoreType``'s constant names, which ``isValidEnum`` compares
#: case-sensitively.
_CORE_TYPE_NAMES = {
    "Kindred", "Artifact", "Battle", "Conspiracy", "Enchantment", "Creature",
    "Dungeon", "Instant", "Land", "Phenomenon", "Plane", "Planeswalker",
    "Scheme", "Sorcery", "Vanguard",
}
#: ``Cost.NUM_NAMES``: the words ``convertIntAndTypeToWords`` counts in.
_COST_NUM_NAMES = (
    "zero", "a", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
)
_NUMERALS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def java_format(template: str, args: list[str]) -> str:
    """``String.format`` for ``%s`` and ``%d``; a specifier with no argument
    is left for :func:`strip_specifiers`."""
    position = 0

    def substitute(match: re.Match) -> str:
        nonlocal position
        if match.group(1) is not None:
            index = int(match.group(1)) - 1
        else:
            index = position
            position += 1
        if 0 <= index < len(args):
            return str(args[index])
        return match.group(0)

    return _SPECIFIER_RE.sub(substitute, template)


def strip_specifiers(text: str) -> str:
    """Remove every unfilled specifier, and the stray space it leaves behind."""
    text = _SPECIFIER_RE.sub("", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", _WHITESPACE_RE.sub(" ", text))
    return text.strip()


# ── Forge's renderings of costs, types and amounts ──────────────────────


def _article(noun: str) -> str:
    """``Lang.nounWithAmount(1, noun)``: "a" or "an" in front of the noun."""
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def _count(amount: str, noun: str) -> str:
    if amount in ("1", ""):
        return _article(noun)
    return f"{amount} {noun}"


def _numeral(n: int) -> str:
    """``Lang.getNumeral``, for the non-negative counts a cost carries."""
    if n < 20:
        return _NUMERALS[n]
    if n < 100:
        return f"{_TENS[n // 10 - 2]} {_NUMERALS[n % 10] if n % 10 else ''}"
    return str(n)


def _lang_plural(noun: str) -> str:
    """``Lang.getPlural``, whose rules differ from :func:`_plural`'s."""
    if noun.endswith("ds"):
        return noun
    if noun.endswith(("s", "x", "ch")):
        return noun + "es"
    return noun + "s"


def _amount_type_words(amount: str, noun: str) -> str:
    """``Cost.convertAmountTypeToWords``: a count in words, or a raw ``X``."""
    if not amount.isdigit():
        return f"{amount} {noun}"
    count = int(amount)
    if count == 1 and noun.startswith("another"):
        return noun
    if count >= len(_COST_NUM_NAMES):
        word = amount
    elif count == 1 and noun[:1].lower() in "aeiou":
        word = "an"
    else:
        word = _COST_NUM_NAMES[count]
    return f"{word} {noun if count == 1 else noun + 's'}"


def _sacrifice(amount: str, noun: str, fields: list[str]) -> str:
    if noun.startswith("another"):
        return f"Sacrifice {noun}"
    return f"Sacrifice {_count(amount, noun or 'permanent')}"


def _discard(amount: str, noun: str, fields: list[str]) -> str:
    if noun == "hand":
        return "Discard your hand"
    return f"Discard {_count(amount, noun or 'card')}"


def _get_counter(amount: str, noun: str, fields: list[str]) -> str:
    counter = fields[1].capitalize() if len(fields) > 1 else ""
    return f"Get {_amount_type_words(amount, counter + ' counter')}"


#: ``CostPart.toString`` for the common parts, by the part's script name; each
#: takes the part's amount, its lower-cased noun and its raw ``/`` fields.
_BRACKET_WORDINGS: dict[str, Callable[[str, str, list[str]], str]] = {
    "PayLife": lambda amount, noun, fields: f"Pay {amount} life",
    "PayEnergy": lambda amount, noun, fields: (
        "Pay " + ("X {E}" if amount == "X" else "{E}" * int(amount))
    ),
    "Behold": lambda amount, noun, fields: f"Behold {_amount_type_words(amount, fields[-1])}",
    "AddCounterYou": _get_counter,
    "Blight": lambda amount, noun, fields: f"Blight {amount}",
    "CollectEvidence": lambda amount, noun, fields: f"Collect evidence {amount}",
    "Waterbend": lambda amount, noun, fields: f"Waterbend {cost_text(amount)}",
    "Sac": _sacrifice,
    "Discard": _discard,
    "ExileFromGrave": lambda amount, noun, fields: (
        f"Exile {_count(amount, noun or 'card')} from your graveyard"
    ),
    "Exile": lambda amount, noun, fields: f"Exile {_count(amount, noun or 'card')}",
    "Return": lambda amount, noun, fields: (
        f"Return {_count(amount, noun or 'permanent')} you control to its owner's hand"
    ),
    "tapXType": lambda amount, noun, fields: (
        f"Tap {_count(amount, 'untapped ' + (noun or 'creature'))} you control"
    ),
    "AddCounter": lambda amount, noun, fields: (
        f"Put {_count(amount, (noun or 'counter') + ' counter')} on this"
    ),
    "SubCounter": lambda amount, noun, fields: (
        f"Remove {_count(amount, (noun or 'counter') + ' counter')} from this"
    ),
}


def _bracket_part(part: str) -> str:
    """A non-mana cost part, worded the way ``CostPart.toString`` words the common ones."""
    name, _, rest = part.partition("<")
    wording = _BRACKET_WORDINGS.get(name)
    if wording is None:
        return part
    fields = rest.rstrip(">").split("/")
    amount = fields[0] if fields else "1"
    noun = (fields[2] if len(fields) > 2 else fields[1] if len(fields) > 1 else "").lower()
    noun = noun.replace(".other", "").replace("card.", "")
    return wording(amount, noun, fields)


def cost_text(cost: str) -> str:
    """``Cost.toSimpleString()``: mana as symbols, the other parts in words."""
    mana: list[str] = []
    others: list[str] = []
    for part in _COST_PART_RE.findall(cost.strip()):
        if part in ("T", "Q"):
            others.append("{" + part + "}")
        elif _MANA_SHARD_RE.match(part):
            mana.append("{" + part + "}")
        elif "<" in part:
            others.append(_bracket_part(part))
        else:
            others.append(part)
    parts = (["".join(mana)] if mana else []) + others
    return ", ".join(parts)


def _only_mana(cost: str) -> bool:
    """``Cost.isOnlyManaCost``: waterbend is a ``CostPartMana`` too."""
    return all(
        _MANA_SHARD_RE.match(part) or part.startswith("Waterbend<")
        for part in _COST_PART_RE.findall(cost.strip())
    )


def _valid_desc(valid: str) -> str:
    """``Lang.buildValidDesc``, for the shapes keyword details carry."""
    words = [part.split(".")[-1] if "." in part else part for part in valid.split(",")]
    words = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", w).lower() for w in words]
    return " or ".join(words)


def _plural(noun: str) -> str:
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


# ── one formatter per Keyword.type class ────────────────────────────────


def _simple(template: str, details: str) -> str:
    return template


def _with_cost(template: str, details: str) -> str:
    cost = details.split(":")[0].split("|", 1)[0].strip()
    if "%" not in template:
        return template
    return java_format(template, [cost_text(cost)])


def _ward(template: str, details: str) -> str:
    def reminder(cost: str) -> str:
        text = cost_text(cost)
        if text.startswith("Pay "):
            return "pays " + text[len("Pay "):]
        if text.startswith("Discard "):
            return "discards " + text[len("Discard "):]
        return "pays " + text if _only_mana(cost) else text

    costs = [c for c in details.split(":") if c]
    return java_format(template, [" or ".join(reminder(c) for c in costs)])


def _equip(template: str, details: str) -> str:
    fields = details.split(":")
    kind = fields[2] if len(fields) > 2 else "creature"
    return java_format(template, [cost_text(fields[0]), kind])


def _kicker(template: str, details: str) -> str:
    fields = [f for f in details.split(":") if f]
    if len(fields) > 1:
        return (
            f"You may pay an additional {cost_text(fields[0])} and/or "
            f"{cost_text(fields[1])} as you cast this spell."
        )
    return _with_cost(template, details)


def _mayhem(template: str, details: str) -> str:
    if not details:
        return (
            "You may play this card from your graveyard if you discarded it "
            "this turn. Timing rules still apply."
        )
    return _with_cost(template, details)


def _emerge(template: str, details: str) -> str:
    fields = details.split(":")
    kind = fields[1] if len(fields) > 1 else "creature"
    if kind.lower() in _CARD_TYPES:
        kind = kind.lower()
    return java_format(template, [cost_text(fields[0]), kind])


def _ninjutsu(template: str, details: str) -> str:
    fields = details.split(":")
    zone = (
        "hand or the command zone"
        if len(fields) > 1 and fields[1] == "Commander" else "hand"
    )
    return java_format(template, [cost_text(fields[0]), zone])


def _exile_ctrl_or_grave(part: str, x_min: int) -> str:
    """``CostExile.exileMultiZoneCostString(true, xMin)``, Craft's wording of
    an ``ExileCtrlOrGrave`` part."""
    fields = part.partition("<")[2].rstrip(">").split("/", 2)
    amount = int(fields[0]) if fields[0].isdigit() else 0
    part_type = fields[1].replace(".Other", "") if len(fields) > 1 else ""
    if len(fields) > 2:
        noun = fields[2]
    elif part_type in _CORE_TYPE_NAMES or part_type == "Permanent":
        noun = part_type.lower()
    else:
        noun = part_type
    nouns = noun if " " in noun else _lang_plural(noun)
    permanent = noun == "permanent"
    other = "other " if permanent else ""
    if amount == 1:
        first = f"another {noun}" if part_type == "Artifact" or permanent else _article(noun)
        text = f"{first} you control or {_article(noun)} card from "
    elif amount > 1:
        text = (
            f"the {_numeral(amount)} from among {other}{nouns} you control "
            f"and/or {noun} cards in "
        )
    elif x_min == 1:
        text = (
            f"one or more {other}{nouns} you control and/or "
            f"{'' if permanent else noun} cards from "
        )
    else:
        text = (
            f"{'the ' if x_min > 1 else ''}{_numeral(x_min)} or more from among "
            f"{other}{nouns} you control and/or cards from "
        )
    return f"Exile {text}your graveyard"


def _craft(template: str, details: str) -> str:
    fields = details.split(":")
    parts = _COST_PART_RE.findall(fields[0])
    mana = " ".join(part for part in parts if _MANA_SHARD_RE.match(part))
    others = [part for part in parts if "<" in part]
    x_min = next((int(part[4:]) for part in parts if part.startswith("XMin")), 0)
    if len(fields) > 2:
        reminder = (
            f"Exile {fields[2]} from among permanents you control and/or cards "
            "in your graveyard"
        )
    else:
        reminder = ", ".join(
            _exile_ctrl_or_grave(part, x_min) if part.startswith("ExileCtrlOrGrave<")
            else _bracket_part(part)
            for part in others
        )
    return java_format(template, [cost_text(mana), reminder])


def _amount(details: str) -> tuple[str, bool]:
    if details.startswith("X"):
        return "X", True
    head = details.split(":")[0]
    return (head if head else "0"), False


def _with_amount(template: str, details: str) -> str:
    amount, _ = _amount(details)
    return java_format(template, [amount])


def _modular(template: str, details: str) -> str:
    if details == "Sunburst":
        return (
            "This enters with a +1/+1 counter on it for each color of mana spent "
            "to cast it. When it dies, you may put its +1/+1 counters on target "
            "artifact creature."
        )
    return _with_amount(template, details)


def _vanishing(template: str, details: str) -> str:
    if not details:
        return (
            "At the beginning of your upkeep, remove a time counter from this "
            "enchantment. When the last is removed, sacrifice it."
        )
    return _with_amount(template, details)


def _firebending(template: str, details: str) -> str:
    amount, with_x = _amount(details)
    fire = "X {R}" if with_x else "{R}" * int(amount)
    return java_format(template, [fire])


def _amplify(template: str, details: str) -> str:
    amount, _ = _amount(details)
    return java_format(template, [amount, "creature"])


def _devour(template: str, details: str) -> str:
    fields = details.split(":")
    amount, _ = _amount(details)
    kind = "creatures"
    if len(fields) > 1 and fields[1]:
        kind = _plural(_valid_desc(fields[1]))
    return java_format(template, [amount, kind])


def _with_cost_and_amount(template: str, details: str) -> str:
    fields = details.split(":")
    amount = "X" if fields[0].startswith("X") else fields[0]
    cost = fields[1].split("|", 1)[0].strip() if len(fields) > 1 else ""
    return java_format(template, [cost_text(cost), amount])


def _suspend(template: str, details: str) -> str:
    if not details:
        return (
            "At the beginning of its owner's upkeep, remove a time counter from "
            "that card. When the last is removed, the player plays it without "
            "paying its mana cost. If it's a creature, it has haste."
        )
    return _with_cost_and_amount(template, details)


def _with_cost_and_type(template: str, details: str) -> str:
    fields = details.split(":")
    kind = fields[0]
    cost = fields[1] if len(fields) > 1 else ""
    if len(fields) > 2:
        reminder = fields[2]
    elif kind == "Basic":
        reminder = "basic land"
    elif kind == "Affinity":
        reminder = "card with affinity"
    else:
        reminder = _valid_desc(kind)
    if template.lower().find("search your library") >= 0:
        reminder = (
            "a card with affinity" if kind == "Affinity"
            else _article(reminder + " card")
        )
    return java_format(template, [cost_text(cost), reminder])


def _type_desc(details: str) -> str:
    if ":" in details:
        return details.split(":")[1]
    if details.lower() in _COLORS:
        return details.lower()
    return _valid_desc(details)


def _with_type(template: str, details: str) -> str:
    return java_format(template, [_type_desc(details)])


def _hexproof(template: str, details: str) -> str:
    if not details:
        return "This can't be the target of spells or abilities your opponents control."
    return _with_type(template, details)


def _trample(template: str, details: str) -> str:
    if details:
        return (
            "This creature can deal excess combat damage to the controller of "
            "the planeswalker it's attacking."
        )
    return template


def _affinity(template: str, details: str) -> str:
    special = {
        "affinity": "permanent with affinity",
        "outlaw": "Assassin, Mercenary, Pirate, Rogue, and/or Warlock",
        "historic": "artifact, legendary, and/or Saga permanent",
    }
    if details.lower() in special:
        reminder = special[details.lower()]
    elif ":" in details:
        reminder = details.split(":")[1]
    else:
        reminder = _valid_desc(details)
    return java_format(template, [reminder])


def _partner(template: str, details: str) -> str:
    if details:
        return "You can have two commanders if both have this ability."
    return template


#: ``formatter -> (template, details) -> filled text``. Keyed by the simple
#: name of the keyword's ``Keyword.type`` class, as the definitions file
#: records it.
FORMATTERS: dict[str, Callable[[str, str], str]] = {
    "SimpleKeyword": _simple,
    "Keyword": _simple,
    "Companion": _simple,
    "Compleated": _simple,
    "Partner": _partner,
    "KeywordWithCost": _with_cost,
    "Ward": _ward,
    "Equip": _equip,
    "Kicker": _kicker,
    "Mayhem": _mayhem,
    "Emerge": _emerge,
    "Ninjutsu": _ninjutsu,
    "Craft": _craft,
    "KeywordWithAmount": _with_amount,
    "Modular": _modular,
    "Vanishing": _vanishing,
    "Firebending": _firebending,
    "Amplify": _amplify,
    "Devour": _devour,
    "KeywordWithCostAndAmount": _with_cost_and_amount,
    "Suspend": _suspend,
    "KeywordWithCostAndType": _with_cost_and_type,
    "KeywordWithType": _with_type,
    "Landwalk": _with_type,
    "Protection": _with_type,
    "Hexproof": _hexproof,
    "Trample": _trample,
    "Affinity": _affinity,
}

def format_reminder(formatter: str | None, template: str, details: str) -> str:
    """The reminder text one keyword instance reads, specifiers removed.

    ``details`` is everything after the keyword line's first colon, as Forge
    hands it to the keyword class's ``parse``.
    """
    details = details.strip()
    fill = FORMATTERS.get(formatter or "")
    if fill is None:
        values = [value for value in details.split(":") if value] if details else []
        filled = java_format(template, values)
    else:
        try:
            filled = fill(template, details)
        except (ValueError, IndexError):
            # A detail shape the mirror does not parse (Forge would throw too):
            # the generic wording is still a correct reading of the keyword.
            filled = template
    return strip_specifiers(filled)
