"""Where a curated dataset puts a record (FR-089, FR-142)."""

from __future__ import annotations

from enum import StrEnum


class Stratum(StrEnum):
    TRAINING = "training"
    CARD_DISJOINT = "card-disjoint"
    GAME_DISJOINT = "game-disjoint"
    #: A slice of card-disjoint, its records written twice: the resolutions
    #: acting through a held-out text, which ``validation_samples`` draws the
    #: card-disjoint sample's resolution classes from first and the trainer
    #: reports the size of. Its count overlaps card-disjoint's.
    GATE_ONE = "gate-one"
    #: Held-out games the card-disjoint cap declined; counted, never written.
    DROPPED_HELD_OUT = "dropped-held-out"
    #: Records of training games acting through a held-out ability text;
    #: counted, never written, while the rest of the game still trains.
    DROPPED_HELD_OUT_TEXT = "dropped-held-out-text"


#: The strata a build writes a directory for, in report order.
OUTPUT_STRATA: tuple[Stratum, ...] = (
    Stratum.TRAINING, Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT, Stratum.GATE_ONE,
)
#: The two strata the trainer validates on, each with a fixed sample.
VALIDATION_STRATA: tuple[Stratum, ...] = (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT)
