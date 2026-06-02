"""The fused single-pass loader must match the reference build_state path.

``_Loader._emit_seat`` walks each seat once and materializes typed-token arrays
directly (perf). This test locks it to the tested-but-slow per-pick
``build_state`` semantics: for every (seat, pack, pick) the two must agree on
the per-type token multisets (name, packs_ago, pick_ago), the target token, and
the scalar/flag fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from draft.application.train_draft_agent import DraftExample, _Loader
from draft.domain.draft_geometry import Booster, DraftGeometry, DraftRecord, Seat
from draft.domain.draft_state import TYPE_PACK, build_state


class _FakeLocator:
    """Returns a deterministic vector per name; every card present (no drops)."""

    def __init__(self, dim: int = 6) -> None:
        self._dim = dim
        self._tbl: dict[str, np.ndarray] = {}

    def load_embedding(self, name: str) -> np.ndarray:
        if name not in self._tbl:
            self._tbl[name] = np.random.default_rng(abs(hash(name)) % 2**32).standard_normal(
                self._dim,
            ).astype(np.float32)
        return self._tbl[name]


def _make_record(pod: int, packs: int, P: int) -> DraftRecord:
    seats = [Seat("forge-full", [], 1.0) for _ in range(pod)]
    boosters = [
        Booster("TST", [f"k{k}_{j}" for j in range(P)]) for k in range(pod * packs)
    ]
    return DraftRecord("d", "r", "t", seats, boosters)


def _reference_groups(record, geo, seat, pack, pick):
    """(per-type multiset, target position in full token list) from build_state."""
    state = build_state(record, geo, seat, pack, pick)
    groups: dict[int, list[tuple[str, int, int]]] = {0: [], 1: [], 2: [], 3: []}
    for c in state.cards:
        groups[c.token_type].append((c.name, c.packs_ago, c.pick_ago))
    pack_positions = [i for i, c in enumerate(state.cards) if c.token_type == TYPE_PACK]
    target_pos = pack_positions[state.target_index] if pack_positions else -1
    return groups, target_pos


def _fused_groups(ex: DraftExample, rev: dict[int, str]):
    groups: dict[int, list[tuple[str, int, int]]] = {0: [], 1: [], 2: [], 3: []}
    for ci, t, pa, pk in zip(ex.card_idx, ex.type_idx, ex.packs_ago, ex.pick_ago):
        groups[int(t)].append((rev[int(ci)], int(pa), int(pk)))
    return groups


@pytest.mark.parametrize(
    ("pod", "packs", "P"),
    [(2, 1, 4), (4, 1, 4), (8, 3, 15), (4, 3, 6), (6, 2, 14), (3, 2, 7)],
)
def test_fused_matches_build_state(pod: int, packs: int, P: int) -> None:
    record = _make_record(pod, packs, P)
    geo = DraftGeometry.from_record(record)
    locator = _FakeLocator()
    loader = _Loader(locator, {"forge-full"})

    for seat in range(pod):
        emitted: list[DraftExample] = []
        loader._emit_seat(
            record, geo, draft_index=0, seat=seat,
            whitelisted=True, critic_active=True, reward=1.0, examples=emitted,
        )
        # One example per (pack, pick), in order.
        assert len(emitted) == packs * P
        rev = {v: k for k, v in loader._name_to_idx.items() if v is not None}

        idx = 0
        for pack in range(1, packs + 1):
            for pick in range(1, P + 1):
                ex = emitted[idx]
                idx += 1
                assert (ex.pack_number, ex.pick_number) == (pack, pick)

                ref_groups, ref_target = _reference_groups(record, geo, seat, pack, pick)
                got_groups = _fused_groups(ex, rev)
                for ttype in range(4):
                    # Order within a type group is insignificant -> compare multisets.
                    assert sorted(got_groups[ttype]) == sorted(ref_groups[ttype]), (
                        f"type {ttype} mismatch at seat={seat} pack={pack} pick={pick}"
                    )
                assert ex.target_token == ref_target
                assert ex.imitation_active is True
                assert ex.critic_active is True
                assert ex.critic_target == 1.0
