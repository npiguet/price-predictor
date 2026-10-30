"""What a checkpoint records about the data it trained on, and when two
checkpoints describe the same split.

The game sets are enumerated rather than derived: re-deriving them against a
grown corpus would differ. Hashing the recorded files is IO and stays with the
store (:func:`effects.infrastructure.effect_model_store.verify_hashes`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any

from effects.domain.text_holdout import HoldoutUnit


def _like(default, value):
    """A stored ``value`` as its field's type, which a tuple's, a count's or an
    enum's serialized form may not keep."""
    return type(default)(value) if isinstance(default, (tuple, int, Enum)) else value


@dataclass(frozen=True, slots=True)
class SplitProvenance:
    """What a checkpoint records about the data it saw; the game sets are
    enumerated, since re-deriving them against a grown corpus would differ."""

    held_out_cards: tuple[str, ...] = ()
    card_disjoint_games: tuple[str, ...] = ()
    game_disjoint_games: tuple[str, ...] = ()
    vocab_path: str = ""
    keyword_definitions_path: str = ""
    vocab_hash: str = ""
    keyword_definitions_hash: str = ""
    withheld_keyword: str | None = None
    #: The holdout flags the split was computed under (FR-134).
    holdout_permille: int = 0
    holdout_max_carriers: int = 0
    #: The curated dataset read and its manifest digest (FR-147); ``""`` on a
    #: checkpoint saved before curated corpora existed.
    corpus_path: str = ""
    corpus_digest: str = ""
    #: FR-042. ``text`` is written as no field at all, so a checkpoint that
    #: records none reads as it trained.
    holdout_unit: HoldoutUnit = HoldoutUnit.TEXT

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # A plain string: an enum member would tie loading to its import path.
        data["holdout_unit"] = str(self.holdout_unit)
        if self.holdout_unit is HoldoutUnit.TEXT:
            del data["holdout_unit"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitProvenance:
        """The provenance ``as_dict`` wrote; a field a checkpoint predates
        takes its default, which is how such a checkpoint trained."""
        return cls(**{
            spec.name: _like(spec.default, data[spec.name])
            for spec in fields(cls) if spec.name in data
        })

    def file_hashes(self) -> dict[str, str]:
        return {
            "vocabulary": self.vocab_hash,
            "keyword definitions": self.keyword_definitions_hash,
        }

    def describes_same_split(self, other: SplitProvenance) -> bool:
        return (
            self.held_out_cards == other.held_out_cards
            and set(self.card_disjoint_games) == set(other.card_disjoint_games)
            and set(self.game_disjoint_games) == set(other.game_disjoint_games)
        )
