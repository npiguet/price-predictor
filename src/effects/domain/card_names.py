"""The one spelling of a card name both sides of the Python/Java boundary share.

Torch-free, because the coverage counter's pool workers call it per record.
"""

from __future__ import annotations


def fold_card_name(name: str) -> str:
    """A card name in the one spelling both sides of the boundary agree on.

    Converted text is lowercased while Forge's ``getName()`` carries printed
    case, and comparing the two directly never matches, silently. ``lower()``
    rather than ``casefold()`` to agree with Java's
    ``toLowerCase(Locale.ROOT)``.
    """
    return name.lower()
