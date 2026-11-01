"""``seeded_draw``: the shared draws, pinned to the values the existing draws produced.

Corpora and probe sets on disk were chosen through these hashes, so a rebuild
has to keep choosing what they chose; each pin below is the value the draw it
replaced returned before the extraction.
"""

from __future__ import annotations

import pytest

from effects.domain.corpus_curation import game_hash, record_hash
from effects.domain.seeded_draw import id_hash, keyed_hash, smallest_k, stable_order

#: (value, keyed hash under seed 0, under seed 7, id hash), from the draws as
#: they stood before they were moved here.
_PINS = [
    ("run.0-abc.17", 2763770926404272686, 9286626237044659398, 3616614414),
    ("game-1", 16319217288563707162, 320775295290820145, 4140153980),
]


@pytest.mark.parametrize(("value", "seed0", "seed7", "crc"), _PINS)
def test_hashes_keep_the_values_the_existing_draws_produced(value, seed0, seed7, crc):
    assert keyed_hash(value, 0) == seed0
    assert keyed_hash(value, 7) == seed7
    assert id_hash(value) == crc


@pytest.mark.parametrize(("value", "seed0", "seed7", "crc"), _PINS)
def test_existing_draws_delegate_to_it(value, seed0, seed7, crc):
    assert record_hash(value, seed=0) == seed0
    assert record_hash(value, seed=7) == seed7
    assert game_hash(value) == crc


def test_stable_order_is_by_keyed_hash_and_ignores_input_order():
    ids = [f"g{i}" for i in range(50)]
    order = stable_order(ids, seed=3)
    assert order == sorted(ids, key=lambda v: keyed_hash(v, 3))
    assert stable_order(reversed(ids), seed=3) == order
    assert stable_order(ids + ids[:5], seed=3) == order


def test_an_id_keeps_its_place_relative_to_the_others_when_ids_are_added():
    base = [f"g{i}" for i in range(30)]
    grown = stable_order(base + [f"h{i}" for i in range(30)], seed=1)
    assert [v for v in grown if v in set(base)] == stable_order(base, seed=1)


def test_seeds_give_different_orders():
    ids = [f"g{i}" for i in range(50)]
    assert stable_order(ids, seed=1) != stable_order(ids, seed=2)


@pytest.mark.parametrize("k", [0, 1, 7, 50, 80])
def test_smallest_k_is_the_head_of_the_stable_order(k):
    ids = [f"g{i}" for i in range(50)]
    assert smallest_k(ids, k, seed=5) == stable_order(ids, seed=5)[:k]
