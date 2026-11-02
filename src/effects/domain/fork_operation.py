"""The ten changes a directed fork applies to a copied game (FR-011).

A module of its own because both the record envelope (``records.py``, a
directed fork's ``directed`` tag) and the rules catalogue (a variant's
``operation``) name it, and the catalogue already imports the records.

Mirrored by the Java ``ForkOperation`` enum; ``ForkOperationContractTest`` and
``test_fork_operation.py`` each read the other side's source and pin the
spellings, so one is changed on both sides or neither.
"""

from __future__ import annotations

from enum import StrEnum


class ForkOperation(StrEnum):
    PT = "pt"
    KEYWORD = "keyword"
    COLOUR_TYPE = "colour_type"
    COUNTERS = "counters"
    TAP = "tap"
    LIFE = "life"
    REPLACE_CARD = "replace_card"
    REMOVE_PERMANENT = "remove_permanent"
    ADD_PERMANENT = "add_permanent"
    MOVE_CARD = "move_card"
