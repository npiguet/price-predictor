"""Lethality copies are made for a record's first natural copy only (FR-063e).

A step holds a record's natural copies and its rule-budget copies as rows of
one record id. A lethality copy of each would repeat the same redrawn board
once per row, at full weight, so the copies hang on the first natural row.
"""

from __future__ import annotations

import random
from dataclasses import replace

from effects.application.lethality_augmenter import LethalityAugmenter
from effects.domain.effect_targets import derive_targets
from effects.domain.records import Curation
from tests.unit.effects.domain.test_lethality_counterfactuals import (
    BOLT,
    _creature,
    _damage,
    _died,
    _line,
    _no_keywords,
    _record,
)


class _Batcher:
    def __init__(self):
        self.rng = random.Random(11)

    def line_of(self, key):
        return _line(BOLT)

    def text_of(self, key):
        return BOLT

    def override_text(self, *_args, **_kwargs):
        pass


def _copies_of(records) -> list:
    augmenter = LethalityAugmenter(2, _no_keywords, None)
    return augmenter.copies_of(records, [derive_targets(r) for r in records], _Batcher())


def _bolt():
    return _record([_damage("E1", 3), _died("E1")], [_creature("E1", 2)])


def test_a_records_second_natural_row_and_its_added_row_get_no_lethality_copies():
    bolt = _bolt()
    natural = replace(bolt, curation=Curation(2, 3))
    added = replace(bolt, curation=Curation(2, 3, "704.5f", ("704.5f",), (), ("E1",)))
    assert len(_copies_of([natural])) == 2
    assert len(_copies_of([natural, natural, added])) == 2
    assert len(_copies_of([added, natural, natural])) == 2


def test_an_added_row_alone_gets_none_and_each_record_keeps_its_own_first_row():
    bolt = _bolt()
    other = replace(_bolt(), record_id="r.0-0.2")
    added = replace(bolt, curation=Curation(1, 2, "704.5f", ("704.5f",), (), ("E1",)))
    assert _copies_of([added]) == []
    assert len(_copies_of([bolt, other, bolt])) == 4


def test_an_uncurated_record_is_a_natural_row():
    assert len(_copies_of([_bolt()])) == 2
