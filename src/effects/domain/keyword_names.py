"""A keyword's name, in the one spelling this package reads.

Forge writes what the card prints — ``"first strike"``, ``"ward:2"``,
``"landwalk:forest"``, ``"protection from blue"`` — while the overlay
vocabulary, gate 2's table and ``--probe-keywords`` all spell them
``first_strike``, ``ward`` and ``protection``. Every reader converts through
here: a copy that forgets the value or the protection quality matches nothing
and says so nowhere, which is how first strike and double strike once read as
unobserved on a corpus holding hundreds of each, and how a ward grant never
reached the head's ``keywords_gained`` target.

Torch-free and import-free on purpose, so the record readers, the curation
workers and the evaluator can all share it.
"""

from __future__ import annotations

#: Forge spells a protection keyword by its quality; the overlay has one slot.
PROTECTION_FROM = "protection_from_"


def normalize_keyword(raw: str) -> str:
    """Forge's spelling of a keyword, in the one this package reads.

    The parameter after ``:`` is dropped because the keyword is the same
    keyword whatever its value; the value reaches the model through the ability
    line, not through the overlay bit.
    """
    return raw.split(":", 1)[0].strip().lower().replace(" ", "_")


def overlay_keyword(raw: str) -> str:
    """:func:`normalize_keyword`, with every protection folded onto ``protection``.

    The overlay vocabulary has one protection slot, so ``protection_from_red``
    and ``protection_from_creatures`` are both that slot — on the input side
    under the gen-2 layout, and on the target side always.
    """
    name = normalize_keyword(raw)
    return "protection" if name.startswith(PROTECTION_FROM) else name
