"""Gen-3 round loader: learner-only picks, per-round table, drop accounting.

Covers data-model §2/§3 and spec FR-007/FR-022.
"""

from __future__ import annotations

import numpy as np

from draft.application.train_draft_agent_online import RoundLoader
from draft.domain.draft_geometry import Booster, DraftRecord, Seat
from draft.domain.draft_state import TYPE_PACK

_POD = 4
_PACK_SIZE = 2
_DIM = 8


class _FakeLocator:
    """Deterministic vector per known name; ``None`` for names holding 'missing'."""

    def __init__(self, dim: int = _DIM) -> None:
        self._dim = dim
        self.loads: list[str] = []

    def load_embedding(self, name: str):
        self.loads.append(name)
        if "missing" in name:
            return None
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        return rng.standard_normal(self._dim).astype(np.float32)


def _boosters(tag: str = "a") -> list[Booster]:
    # pod_size=4, packs=1 (4 boosters), pack_size=2.
    return [
        Booster("TST", [f"{tag}k{k}c{j}" for j in range(_PACK_SIZE)])
        for k in range(_POD)
    ]


def _record(seats: list[Seat], tag: str = "a", draft_id: str = "d") -> DraftRecord:
    return DraftRecord(draft_id, "r", "t", seats, _boosters(tag))


def _default_seats() -> list[Seat]:
    return [
        Seat("gen-3", ["x"] * 40, 10.0),
        Seat("gen-1", ["x"] * 40, 6.0),
        Seat("gen-3", ["x"] * 40, 2.0),
        Seat("forge-r30", ["x"] * 40, 4.0),
    ]


def test_only_learner_seats_yield_examples() -> None:
    loader = RoundLoader(_FakeLocator())
    batch = loader.build([_record(_default_seats())], "gen-3")

    # Two learner seats × pack_size picks each.
    assert len(batch.seat_examples) == 2
    assert len(batch.examples) == 2 * _PACK_SIZE
    assert len(batch.learner_rewards) == 2
    assert batch.dropped_seats == 0


def test_a_learner_seat_with_a_failed_build_yields_no_examples() -> None:
    seats = _default_seats()
    seats[2] = Seat("gen-3", [], None)
    loader = RoundLoader(_FakeLocator())
    batch = loader.build([_record(seats)], "gen-3")

    assert len(batch.seat_examples) == 1
    assert len(batch.learner_rewards) == 1
    assert batch.dropped_seats == 1


def test_no_learner_label_in_the_pod_yields_an_empty_round() -> None:
    loader = RoundLoader(_FakeLocator())
    batch = loader.build([_record(_default_seats())], "gen-9")

    assert batch.examples == []
    assert batch.learner_rewards == []
    assert batch.dropped_seats == 0


def test_a_pick_whose_taken_card_is_unembeddable_is_dropped() -> None:
    """``action_position == -1`` means no usable action, so the pick is dropped."""
    seats = _default_seats()
    boosters = [
        # Seat 0 opens booster 0; its first (taken) card is un-embeddable at
        # pick 1, so that state carries no action and must not be trained on.
        Booster("TST", ["missing-1", "a0c1"]),
        Booster("TST", ["a1c0", "a1c1"]),
        Booster("TST", ["a2c0", "a2c1"]),
        Booster("TST", ["a3c0", "a3c1"]),
    ]
    record = DraftRecord("d", "r", "t", seats, boosters)

    loader = RoundLoader(_FakeLocator())
    batch = loader.build([record], "gen-3")

    assert all(ex.action_token >= 0 for ex in batch.examples)
    # Seat 0 lost exactly its first pick; seat 2 kept both.
    assert len(batch.examples) == (_PACK_SIZE - 1) + _PACK_SIZE


def test_every_example_action_token_points_at_a_pack_position() -> None:
    loader = RoundLoader(_FakeLocator())
    batch = loader.build([_record(_default_seats())], "gen-3")

    for ex in batch.examples:
        assert 0 <= ex.action_token < ex.n_tokens
        assert ex.type_idx[ex.action_token] == TYPE_PACK


# Cards one learner seat actually sees: both cards of its own booster at pick 1,
# then the single card left in the booster that wheels to it at pick 2. The card
# another seat took from that booster is never visible to this seat, so it is
# never loaded — the table holds sightings, not the whole pod's cards.
_CARDS_SEEN_PER_SEAT = _PACK_SIZE + 1
_LEARNER_SEATS = 2


def test_table_covers_only_this_rounds_cards() -> None:
    loader = RoundLoader(_FakeLocator())
    batch = loader.build([_record(_default_seats(), tag="a")], "gen-3")

    # The two learner seats open disjoint boosters, so their sightings are disjoint.
    assert batch.table.shape == (_LEARNER_SEATS * _CARDS_SEEN_PER_SEAT, _DIM)
    assert batch.table.shape[0] < _POD * _PACK_SIZE  # strictly less than every card
    assert batch.table.dtype == np.float32
    max_row = max(int(ex.card_idx.max()) for ex in batch.examples)
    assert max_row < batch.table.shape[0]


def test_each_round_builds_a_fresh_table_not_a_growing_one() -> None:
    """A per-round table keeps an hours-long run from growing without bound."""
    loader = RoundLoader(_FakeLocator())
    first = loader.build([_record(_default_seats(), tag="a")], "gen-3")
    second = loader.build([_record(_default_seats(), tag="b", draft_id="e")], "gen-3")

    assert second.table.shape == first.table.shape


def test_all_of_a_seats_picks_share_one_advantage_slot() -> None:
    loader = RoundLoader(_FakeLocator())
    batch = loader.build([_record(_default_seats())], "gen-3")

    for seat_examples in batch.seat_examples:
        assert len({ex.advantage for ex in seat_examples}) == 1


def test_multiple_records_accumulate_into_one_round() -> None:
    loader = RoundLoader(_FakeLocator())
    batch = loader.build(
        [
            _record(_default_seats(), tag="a", draft_id="d1"),
            _record(_default_seats(), tag="b", draft_id="d2"),
        ],
        "gen-3",
    )

    assert len(batch.seat_examples) == 4
    assert len(batch.learner_rewards) == 4
    # Both records' sightings land in the one round table (distinct card names).
    assert batch.table.shape[0] == 2 * _LEARNER_SEATS * _CARDS_SEEN_PER_SEAT


def test_card_embeddings_are_loaded_once_per_distinct_card() -> None:
    """The locator memoizes; the loader must not re-request a known row."""
    locator = _FakeLocator()
    loader = RoundLoader(locator)
    loader.build([_record(_default_seats())], "gen-3")

    distinct = set(locator.loads)
    assert len(locator.loads) == len(distinct)
