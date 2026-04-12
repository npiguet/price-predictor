"""TrainingSample value object — a card name + raw text + price + metadata.

Replaces the ``(name, text, price, printing_data)`` 4-tuple that train and
evaluate use cases were unpacking inline. Giving the bundle a name makes
call sites self-documenting and lets future fields be added without churning
every unpack site.
"""

from __future__ import annotations

from dataclasses import dataclass

from price_predictor.domain.value_objects import PrintingData


@dataclass(frozen=True)
class TrainingSample:
    """A single transformer training/evaluation row.

    ``text`` is the raw converted-card text (without the ``name:`` line being
    stripped — the tokenizer handles that). ``printing_data`` carries the
    side-channel features fed into the model alongside the token sequence.
    """

    name: str
    text: str
    price_eur: float
    printing_data: PrintingData
