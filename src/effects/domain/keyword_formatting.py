"""Filling a keyword's reminder template the way Forge fills it (FR-017, FR-018).

A keyword line carries its instance values after the display name —
``Ward:2``, ``TypeCycling:Basic:1 B``, ``Equip:3`` — and Forge renders the
reminder text by handing those details to the keyword's own class, which parses
them and formats them into the template: a cost becomes ``{1}{B}``, a
typecycling type becomes "a basic land card", ward's cost becomes "pays {2}".
Filling the raw details in would put Forge's script syntax (``1 B``) into an
English sentence, so expansion mirrors that formatting here.

The classes are mirrored rather than called because expansion runs in Python.
The table is keyed by the simple name of the keyword's ``Keyword.type`` class,
which ``extract-keyword-definitions`` records as ``formatter``; prior art is
each class's ``parse`` and ``formatReminderText`` in Forge's
``forge.game.keyword`` package (``KeywordWithCost``, ``KeywordWithAmount``,
``KeywordWithCostAndAmount``, ``KeywordWithCostAndType``, ``KeywordWithType``
and the keyword-specific subclasses). An unknown formatter, or a definitions
file written before the field existed, fills the raw values in order.

Whatever placeholder no value fills is removed afterwards, so no expansion
carries a ``%`` token.
"""

from __future__ import annotations

import re
from collections.abc import Callable

#: Every format specifier Forge's templates use: ``%s``, ``%d`` and their
#: explicit-index forms ``%1$s``, ``%2$d``.
_SPECIFIER_RE = re.compile(r"%(?:(\d+)\$)?([sd])")
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:])")

#: One Forge cost part: a bracketed part (``Sac<1/Creature>``) or a bare shard.
_COST_PART_RE = re.compile(r"\w+<[^>]*>|\S+")
#: A mana shard as Forge's cost strings spell it: a number, X/Y/Z, a colour or
#: colourless, and the hybrid and Phyrexian forms built from them.
_MANA_SHARD_RE = re.compile(r"^(\d+|[XYZ]|[WUBRGCS](/[WUBRGCP2])*|2/[WUBRG])$")

_COLORS = {"white", "blue", "black", "red", "green", "colorless"}
_CARD_TYPES = {
    "artifact", "battle", "creature", "enchantment", "instant", "kindred",
    "land", "planeswalker", "sorcery", "tribal",
}


def java_format(template: str, args: list[str]) -> str:
    """``String.format`` for the two conversions Forge's templates use.

    An ordinary specifier takes the next argument in turn, an explicit-index one
    (``%1$s``) takes that argument however often it appears, and a specifier
    with no argument to take is left in place for :func:`strip_specifiers`.
    """
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


def _bracket_part(part: str) -> str:
    """A non-mana cost part, worded the way ``CostPart.toString`` words the common ones."""
    name, _, rest = part.partition("<")
    fields = rest.rstrip(">").split("/")
    amount = fields[0] if fields else "1"
    noun = (fields[2] if len(fields) > 2 else fields[1] if len(fields) > 1 else "").lower()
    noun = noun.replace(".other", "").replace("card.", "")
    if name == "PayLife":
        return f"Pay {amount} life"
    if name == "Sac":
        return f"Sacrifice {_count(amount, noun or 'permanent')}"
    if name == "Discard":
        if noun == "hand":
            return "Discard your hand"
        return f"Discard {_count(amount, noun or 'card')}"
    if name == "ExileFromGrave":
        return f"Exile {_count(amount, noun or 'card')} from your graveyard"
    if name == "Exile":
        return f"Exile {_count(amount, noun or 'card')}"
    if name == "Return":
        return f"Return {_count(amount, noun or 'permanent')} you control to its owner's hand"
    if name == "tapXType":
        return f"Tap {_count(amount, 'untapped ' + (noun or 'creature'))} you control"
    if name == "AddCounter":
        return f"Put {_count(amount, (noun or 'counter') + ' counter')} on this"
    if name == "SubCounter":
        return f"Remove {_count(amount, (noun or 'counter') + ' counter')} from this"
    return part


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
    return all(
        _MANA_SHARD_RE.match(part) for part in _COST_PART_RE.findall(cost.strip())
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


def _craft(template: str, details: str) -> str:
    fields = details.split(":")
    mana = " ".join(
        part for part in _COST_PART_RE.findall(fields[0]) if _MANA_SHARD_RE.match(part)
    )
    others = [part for part in _COST_PART_RE.findall(fields[0]) if "<" in part]
    if len(fields) > 2:
        reminder = (
            f"Exile {fields[2]} from among permanents you control and/or cards "
            "in your graveyard"
        )
    else:
        reminder = ", ".join(_bracket_part(part) for part in others)
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

    ``details`` is everything after the keyword line's first colon, exactly as
    Forge hands it to the keyword class's ``parse``. A formatter this table
    does not know fills the raw values in order, which is what an old
    definitions file (no ``formatter`` recorded) gets too.
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
