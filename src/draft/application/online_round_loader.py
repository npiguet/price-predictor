"""One round's records turned into learner examples over a per-round embedding
table (data-model §2/§3), with the learner seats' rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from draft.application.draft_pick_states import iter_seat_pick_states
from draft.application.draft_training_common import leave_one_out_rewards
from draft.application.online_examples import OnlineExample
from draft.application.online_log import _log
from draft.domain.draft_geometry import DraftGeometry, DraftRecord

_MISSING_WARN_CAP = 20


# --------------------------------------------------------------------------- #
# Round loader (data-model §2, research D6/D14)
# --------------------------------------------------------------------------- #

@dataclass
class RoundBatch:
    """One round's fresh, single-use data. Built, consumed once, then dropped."""

    examples: list[OnlineExample] = field(default_factory=list)
    seat_examples: list[list[OnlineExample]] = field(default_factory=list)
    learner_rewards: list[float] = field(default_factory=list)
    table: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 1), dtype=np.float32),
    )
    dropped_seats: int = 0
    index: int = 0
    records: list[DraftRecord] = field(default_factory=list)
    gen_seconds: float = 0.0


def learner_seat_rewards(
    record: DraftRecord, learner_label: str,
) -> tuple[list[tuple[int, float]], int]:
    """``(seat_index, R)`` for every surviving learner seat, plus the drop count.

    ``R`` is the pod-relative leave-one-out ``deck_score`` (FR-008). Two kinds of
    learner seat are excluded and counted as dropped (FR-022, data-model §4):

    * a failed build (``deck_score is None``) — no score, so no reward; and
    * a seat whose pod has **no other** scored seat, whose leave-one-out
      baseline is undefined. ``leave_one_out_rewards`` substitutes ``0.0`` for
      the empty mean there, which would silently turn the raw ``deck_score``
      into the reward on a different scale from every other seat's.
    """
    rewards = leave_one_out_rewards(record)
    n_scored = sum(1 for seat in record.seats if seat.deck_score is not None)
    surviving: list[tuple[int, float]] = []
    dropped = 0
    for seat_idx, seat in enumerate(record.seats):
        if seat.agent != learner_label:
            continue
        reward = rewards[seat_idx]
        if reward is None or n_scored < 2:
            dropped += 1
            continue
        surviving.append((seat_idx, float(reward)))
    return surviving, dropped


class RoundLoader:
    """Turns one round's records into learner examples over a per-round table.

    Mirrors ``rl_examples._Loader``'s shared-table memoization (each
    distinct card resolved once, examples keep int rows) minus the multi-corpus
    split. The table is rebuilt per round (research D14): a round touches only a
    few hundred cards, and a table that accumulated across an hours-long run
    would grow without bound and be re-``stack``ed every round.
    """

    def __init__(self, locator) -> None:
        self._locator = locator
        self.embedding_dim: int | None = None
        self._missing: set[str] = set()
        self._missing_total = 0
        self._table_rows: list[np.ndarray] = []
        self._name_to_idx: dict[str, int | None] = {}

    def _card_index(self, name: str) -> int | None:
        idx = self._name_to_idx.get(name, -1)
        if idx != -1:
            return idx
        emb = self._locator.load_embedding(name)
        if emb is None:
            self._name_to_idx[name] = None
            if len(self._missing) < _MISSING_WARN_CAP:
                self._missing.add(name)
            self._missing_total += 1
            return None
        if self.embedding_dim is None:
            self.embedding_dim = int(emb.shape[0])
        new_idx = len(self._table_rows)
        self._table_rows.append(np.asarray(emb, dtype=np.float32))
        self._name_to_idx[name] = new_idx
        return new_idx

    def _table(self) -> np.ndarray:
        if not self._table_rows:
            return np.zeros((0, self.embedding_dim or 1), dtype=np.float32)
        return np.stack(self._table_rows).astype(np.float32)

    def build(self, records: list[DraftRecord], learner_label: str) -> RoundBatch:
        """Walk every learner seat of every record into a fresh :class:`RoundBatch`."""
        self._table_rows = []
        self._name_to_idx = {}

        batch = RoundBatch(records=list(records))
        for record in records:
            geo = DraftGeometry.from_record(record)
            surviving, dropped = learner_seat_rewards(record, learner_label)
            batch.dropped_seats += dropped
            for seat_idx, reward in surviving:
                seat_examples = [
                    OnlineExample(
                        card_idx=state.card_idx,
                        type_idx=state.type_idx,
                        packs_ago=state.packs_ago,
                        pick_ago=state.pick_ago,
                        pack_number=state.pack_number,
                        pick_number=state.pick_number,
                        action_token=state.action_position,
                    )
                    for state in iter_seat_pick_states(
                        geo, record.boosters, seat_idx, self._card_index,
                    )
                    # action_position == -1 ⇒ the taken card has no .npz, so
                    # there is no usable action to reinforce at this pick.
                    if state.action_position >= 0
                ]
                batch.seat_examples.append(seat_examples)
                batch.learner_rewards.append(reward)
                batch.examples.extend(seat_examples)

        if self._missing_total:
            sample = ", ".join(sorted(self._missing))
            _log(
                f"Dropped {self._missing_total} cards with no .npz embedding "
                f"(e.g. {sample})"
            )
            self._missing.clear()
            self._missing_total = 0

        batch.table = self._table()
        return batch
