"""Token surfaces collated into the trunk's batch, and ``e`` rows scattered into it."""

from __future__ import annotations

import random

import torch

from effects.domain.effect_head_input import EffectHeadInput, Slot, SlotKind
from effects.domain.records import CombatPayload, RecordKind
from effects.domain.surface_collation import collate_surfaces, scatter_e_rows


class TestScatterERows:
    """How the ability encoder's output reaches the effect head.

    A surface built for training carries row indices rather than values, and
    this is where the rows become vectors. Both properties below are the reason
    it exists: a per-slot round trip through the host would sever the gradient,
    and the effect head's loss would then never reach the encoder at all.
    """

    def _matrix(self) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True,
        )

    def test_a_named_row_replaces_the_slots_vector(self):
        e_vectors = torch.zeros(1, 3, 2)
        rows = torch.tensor([[2, 0, -1]])
        out = scatter_e_rows(e_vectors, rows, self._matrix())
        assert out[0, 0].tolist() == [5.0, 6.0]
        assert out[0, 1].tolist() == [1.0, 2.0]

    def test_an_unnamed_slot_keeps_what_collate_put_there(self):
        """-1 means "no row named": a literal vector, or the zero padding."""
        e_vectors = torch.tensor([[[0.0, 0.0], [7.0, 8.0]]])
        rows = torch.tensor([[-1, -1]])
        out = scatter_e_rows(e_vectors, rows, self._matrix())
        assert out.tolist() == e_vectors.tolist()

    def test_row_zero_is_a_row_and_not_a_miss(self):
        e_vectors = torch.zeros(1, 1, 2)
        out = scatter_e_rows(e_vectors, torch.tensor([[0]]), self._matrix())
        assert out[0, 0].tolist() == [1.0, 2.0]

    def test_the_gradient_reaches_the_matrix(self):
        matrix = self._matrix()
        out = scatter_e_rows(torch.zeros(1, 2, 2), torch.tensor([[1, -1]]), matrix)
        out.sum().backward()
        assert matrix.grad is not None
        # Only the row that was named receives gradient.
        assert matrix.grad[1].tolist() == [1.0, 1.0]
        assert matrix.grad[0].tolist() == [0.0, 0.0]
        assert matrix.grad[2].tolist() == [0.0, 0.0]


class TestCollateSurfaces:
    """The vectorised collate against the per-slot one it replaced.

    Nothing the model reads may move, so the reference implementation is
    pasted here rather than described: an equivalence the test can only state
    in prose is one that drifts the first time either side is touched.
    """

    @staticmethod
    def _collate_reference(surfaces, *, e_dim, widths):
        """``collate_surfaces`` as it stood before it was vectorised."""
        batch = len(surfaces)
        width = max((len(s.slots) for s in surfaces), default=1)
        features = {
            kind: torch.zeros(batch, width, size)
            for kind, size in widths.items()
        }
        slot_kinds = torch.zeros(batch, width, dtype=torch.long)
        positions = torch.zeros(batch, width, dtype=torch.long)
        e_vectors = torch.zeros(batch, width, e_dim)
        e_rows = torch.full((batch, width), -1, dtype=torch.long)
        attention = torch.zeros(batch, width, dtype=torch.long)
        option_kinds = torch.zeros(batch, width, dtype=torch.long)

        for row, surface in enumerate(surfaces):
            for column, slot in enumerate(surface.slots):
                slot_kinds[row, column] = int(slot.kind)
                positions[row, column] = slot.position
                attention[row, column] = 1
                option_kinds[row, column] = int(slot.option)
                if slot.features and slot.kind in features:
                    values = torch.tensor(slot.features, dtype=torch.float32)
                    features[slot.kind][row, column, : values.shape[0]] = values
                if isinstance(slot.e, int):
                    e_rows[row, column] = slot.e
                elif slot.e is not None:
                    vector = torch.tensor(slot.e, dtype=torch.float32)
                    e_vectors[row, column, : vector.shape[0]] = vector

        return {
            "slot_features": features,
            "slot_kinds": slot_kinds,
            "positions": positions,
            "e_vectors": e_vectors,
            "e_rows": e_rows,
            "attention_mask": attention,
            "option_kinds": option_kinds,
        }

    def _assert_same(self, surfaces, *, e_dim, widths):
        got = collate_surfaces(surfaces, e_dim=e_dim, widths=widths)
        want = self._collate_reference(surfaces, e_dim=e_dim, widths=widths)
        assert got.slot_features.keys() == want["slot_features"].keys()
        for kind, tensor in want["slot_features"].items():
            mine = got.slot_features[kind]
            assert mine.dtype == tensor.dtype
            assert torch.equal(mine, tensor)
        for name in ("slot_kinds", "positions", "e_vectors", "e_rows",
                     "attention_mask", "option_kinds"):
            assert getattr(got, name).dtype == want[name].dtype, name
            assert torch.equal(getattr(got, name), want[name]), name

    def test_a_batch_of_real_surfaces_collates_identically(self):
        from tests.unit.effects.domain.test_effect_head_input import (
            _build,
            _entity,
            _record,
            _snapshot,
        )

        surfaces = [
            _build(_record()),
            _build(_record(kind=RecordKind.COMBAT, payload=CombatPayload())),
            _build(_record(state=_snapshot(entities=(
                _entity("E1"), _entity("E2"), _entity("E3", controller="P1"),
            )))),
        ]
        widths = {
            kind: max(
                (len(slot.features) for s in surfaces for slot in s.slots
                 if slot.kind is kind),
                default=1,
            )
            for kind in (SlotKind.GLOBAL, SlotKind.ACT, SlotKind.PLAYER,
                         SlotKind.CARD)
        }
        self._assert_same(surfaces, e_dim=4, widths=widths)

    def test_random_boards_of_every_slot_kind_collate_identically(self):
        # Varying board sizes, every slot kind, and all three things an `e`
        # can be: a row index, a vector read from the cache, and nothing.
        rng = random.Random(11)
        e_dim = 5
        widths = {
            SlotKind.GLOBAL: 3, SlotKind.ACT: 4, SlotKind.PLAYER: 2,
            SlotKind.CARD: 6,
        }
        surfaces = []
        for _ in range(7):
            slots = []
            for position in range(rng.randrange(1, 12)):
                kind = rng.choice(list(SlotKind))
                size = widths.get(kind, 0)
                features = tuple(
                    rng.uniform(-9.0, 9.0)
                    for _ in range(rng.randrange(0, size + 1))
                ) if size else ()
                e = rng.choice([
                    None,
                    rng.randrange(0, 40),
                    tuple(rng.uniform(-1.0, 1.0) for _ in range(e_dim)),
                    tuple(rng.uniform(-1.0, 1.0) for _ in range(e_dim - 2)),
                ])
                slots.append(Slot(
                    kind=kind, position=position, features=features, e=e,
                ))
            surfaces.append(EffectHeadInput(slots=tuple(slots)))
        self._assert_same(surfaces, e_dim=e_dim, widths=widths)

    def test_an_empty_batch_keeps_its_shapes(self):
        self._assert_same([], e_dim=3, widths={SlotKind.CARD: 2})
