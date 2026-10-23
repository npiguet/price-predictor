"""Where a curated dataset puts a record (FR-089, FR-142).

A string enum, so a stratum keys the manifest's JSON and names its directory
exactly as the bare string always did.
"""

from __future__ import annotations

from enum import StrEnum


class Stratum(StrEnum):
    TRAINING = "training"
    CARD_DISJOINT = "card-disjoint"
    GAME_DISJOINT = "game-disjoint"
    #: A *slice* of card-disjoint rather than a fourth destination: its
    #: records are written twice on purpose, so the evaluator reads gate 1's
    #: unique-text stratum without re-filtering the whole card-disjoint output,
    #: and its count overlaps card-disjoint's rather than partitioning with it.
    GATE_ONE = "gate-one"
    #: Held-out games the card-disjoint cap declined. Accounting only: nothing
    #: is written for it, so it is counted but never carries a unique-text
    #: figure.
    DROPPED_HELD_OUT = "dropped-held-out"
    #: Records of training games acting through a held-out ability text: a
    #: token or a variant script carrying it, or its template under
    #: ``--holdout-unit template``. Accounting only, like the stratum above;
    #: the rest of the game still trains.
    DROPPED_HELD_OUT_TEXT = "dropped-held-out-text"


#: The strata a build writes a directory for, in report order.
OUTPUT_STRATA: tuple[Stratum, ...] = (
    Stratum.TRAINING, Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT, Stratum.GATE_ONE,
)
#: The two strata the trainer validates on, each with a fixed sample.
VALIDATION_STRATA: tuple[Stratum, ...] = (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT)
