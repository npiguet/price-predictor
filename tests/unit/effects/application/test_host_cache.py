"""The control score's host cache: every form rebuilds its tensor exactly, the
smallest is kept, and one grid record's cells cost little more than one cell
(research E2; T160)."""

from __future__ import annotations

import pytest
import torch

from effects.application.host_cache import (
    _Bits,
    _Blocks,
    _Coo,
    _Narrow,
    _RowDiff,
    host_bytes,
    map_tensors,
    pack,
    unpack,
)
from tests.unit.effects.builders import fixture_run


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("host-cache"))


def _same(want: torch.Tensor, got: torch.Tensor) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)


def _sparse_cards(rows=4, slots=6, width=539, seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    features = torch.zeros((rows, slots, width))
    picks = torch.randint(0, width, (rows, slots, 20), generator=generator)
    features.scatter_(2, picks, torch.rand((rows, slots, 20), generator=generator) + 0.5)
    features[:, -1] = 0.0           # a padded slot
    return features


class TestEachForm:
    def test_a_sparse_float_tensor_is_kept_by_its_non_zero_elements(self):
        cards = _sparse_cards()
        packed = pack(cards)
        assert isinstance(packed, _Coo)
        assert host_bytes(packed) < host_bytes(cards) / 8
        _same(cards, unpack(packed))

    def test_a_boolean_tensor_and_a_zero_one_integer_one_are_kept_as_bits(self):
        mask = torch.rand((5, 9)) > 0.5
        flags = mask.to(torch.int64)
        for tensor in (mask, flags):
            packed = pack(tensor)
            assert isinstance(packed, _Bits)
            _same(tensor, unpack(packed))

    @pytest.mark.parametrize(("low", "high", "dtype"), [
        (-1, 300, torch.int16), (0, 70_000, torch.int32)])
    def test_an_integer_tensor_is_kept_in_the_narrowest_type_its_values_fit(
            self, low, high, dtype):
        tensor = torch.randint(low, high, (40, 7), dtype=torch.int64)
        tensor[0, 0], tensor[0, 1] = low, high
        packed = pack(tensor)
        assert isinstance(packed, _Narrow) and packed.values.dtype == dtype
        _same(tensor, unpack(packed))

    def test_a_tensor_of_one_dimension_is_left_as_it_is(self):
        tensor = torch.arange(5)
        assert pack(tensor) is tensor


class TestRowsOfOneRecord:
    """A grid record's cells: the first row, and where the others differ."""

    def test_near_copies_of_the_first_row_are_kept_as_their_differences(self):
        base = _sparse_cards(rows=1)
        cells = base.repeat(64, 1, 1)
        for row in range(64):
            cells[row, 0, 3] = float(row % 8 + 1)       # an edited power
            cells[row, 1, 7] = float(row // 8 + 1)      # an edited toughness
        packed = pack(cells, rows=64)
        assert isinstance(packed, _RowDiff)
        assert host_bytes(packed) < host_bytes(pack(cells)) / 10
        _same(cells, unpack(packed))

    def test_rows_that_are_not_copies_keep_their_own_form(self):
        cells = _sparse_cards(rows=8, seed=3)
        assert not isinstance(pack(cells, rows=8), _RowDiff)
        _same(cells, unpack(pack(cells, rows=8)))

    def test_relation_triples_repeated_from_row_to_row_are_kept_once(self):
        triples = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        extra = torch.tensor([[9, 9, 9]])
        blocks = [triples, triples, torch.cat([triples, extra]), torch.cat([triples, extra]),
                  triples[:0], triples]
        relations = torch.cat([
            torch.cat([torch.full((len(b), 1), row), b], dim=1) for row, b in enumerate(blocks)])
        packed = _Blocks.of(relations, rows=len(blocks))
        assert packed is not None
        assert packed.blocks.shape[0] == 3 + 4 + 0 + 3
        _same(relations, packed.dense())
        assert isinstance(pack(relations, rows=len(blocks)), _Blocks)

    def test_a_list_not_grouped_by_row_is_not_kept_by_blocks(self):
        relations = torch.tensor([[1, 2, 3, 0], [0, 4, 5, 1]])
        assert _Blocks.of(relations, rows=2) is None
        assert _Blocks.of(torch.tensor([[0, 1], [5, 2]]), rows=2) is None
        _same(relations, unpack(pack(relations, rows=2)))

    def test_rows_without_relations_and_an_empty_list_round_trip(self):
        empty = torch.zeros((0, 4), dtype=torch.long)
        packed = _Blocks.of(empty, rows=3)
        _same(empty, packed.dense())


def test_nested_values_are_packed_and_unpacked_through_and_through():
    value = {"features": [_sparse_cards(rows=2)], "mask": (torch.ones((2, 3), dtype=torch.bool),),
             "ids": torch.full((2, 4), 7, dtype=torch.int64), "name": "kept"}
    packed = pack(value, rows=2)
    assert packed["name"] == "kept"
    before, after = [], []
    map_tensors(value, lambda t: before.append(t) or t)
    map_tensors(unpack(packed), lambda t: after.append(t) or t)
    assert len(before) == len(after) == 3
    for want, got in zip(before, after):
        _same(want, got)


def test_one_grid_records_real_cells_cost_a_fraction_of_their_packed_cells(run, records):
    """A combat-free stand-in for a grid: one fixture record edited into 64
    toughness and power cells, prepared together as the control score
    prepares a grid record's cells, kept far smaller than cell by cell and
    rebuilt exactly."""
    from dataclasses import replace

    from effects.domain.board_edits import BoardEdit

    record = next(r for r in records if any(
        e.pt is not None and e.is_creature for e in r.state.entities))
    creature = next(e for e in record.state.entities if e.pt is not None and e.is_creature)
    cells = [BoardEdit(record).set_power(creature.id, p).set_toughness(creature.id, t).record
             for p in range(1, 9) for t in range(1, 9)]
    batch = run.preparer.scoring_batcher().prepare(cells)
    stripped = replace(batch, texts={}, staged=replace(batch.staged, prepared=[]), surfaces=[])
    together = pack(stripped, rows=len(cells))
    alone = pack(stripped)
    assert host_bytes(together) < host_bytes(alone) / 4
    before, after = [], []
    map_tensors(stripped, lambda t: before.append(t) or t)
    map_tensors(unpack(together), lambda t: after.append(t) or t)
    for want, got in zip(before, after, strict=True):
        _same(want, got)
