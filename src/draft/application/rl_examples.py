"""Gen-2 RL training states: one ``(draft, seat, pack, pick)`` example per pick,
the corpus loader that builds them and the learner seats' trajectories over a
shared embedding table (FR-013, FR-015), and the draft-disjoint split (FR-035).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from draft.application.draft_pick_states import iter_seat_pick_states
from draft.application.draft_training_common import leave_one_out_rewards
from draft.application.rl_log import _log
from draft.domain.draft_geometry import DraftGeometry, DraftRecord
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

RANDOM_SEED = 42  # hardcoded; governs init and the draft-disjoint split.
_MISSING_WARN_CAP = 20


@dataclass(slots=True)
class RLExample:
    """One ``(draft, seat, pack, pick)`` RL training state (data-model §3)."""

    draft_index: int
    card_idx: np.ndarray
    type_idx: np.ndarray
    packs_ago: np.ndarray
    pick_ago: np.ndarray
    pack_number: int
    pick_number: int
    action_token: int        # absolute index of the taken PACK token, or -1
    learner_active: bool      # feeds policy-grad + entropy + KL
    critic_active: bool       # feeds critic MSE
    reward: float             # raw (pre-standardization) terminal pod-relative R
    advantage: float = 0.0    # cached detached GAE A_t (refreshed per epoch)
    value: float = 0.0        # cached critic V(s_t) (refreshed per epoch)

    @property
    def n_tokens(self) -> int:
        return int(self.card_idx.shape[0])


class _Loader:
    """Builds ``RLExample``s + per-learner-seat trajectories from corpora.

    Mirrors ``imitation_examples._Loader``'s shared-table memoization (each
    distinct card's ``.npz`` loaded once; examples keep int rows). The on-policy
    corpus contributes both learner-seat trajectories (policy gradient + critic)
    and non-learner critic coverage; ``critic_corpus`` corpora contribute critic
    coverage only — never learner trajectories (FR-013).
    """

    def __init__(self, locator: ConvertedCardLocator, learner_agents: set[str]) -> None:
        self._locator = locator
        self._learner = learner_agents
        self.embedding_dim: int | None = None
        self.max_packs = 0
        self.max_pack_size = 0
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

    def table(self) -> np.ndarray:
        if not self._table_rows:
            return np.zeros((0, self.embedding_dim or 1), dtype=np.float32)
        return np.stack(self._table_rows).astype(np.float32)

    def build(
        self, on_policy: list[DraftRecord], critic_only: list[DraftRecord],
    ) -> tuple[list[RLExample], list[list[RLExample]]]:
        examples: list[RLExample] = []
        trajectories: list[list[RLExample]] = []
        self._ingest(on_policy, examples, trajectories, allow_learner=True)
        self._ingest(critic_only, examples, trajectories, allow_learner=False)
        if self._missing_total:
            sample = ", ".join(sorted(self._missing))
            _log(
                f"Dropped {self._missing_total} distinct cards with no .npz "
                f"embedding (e.g. {sample})"
            )
        return examples, trajectories

    def _ingest(
        self,
        records: list[DraftRecord],
        examples: list[RLExample],
        trajectories: list[list[RLExample]],
        *,
        allow_learner: bool,
    ) -> None:
        for ri, record in enumerate(records):
            geo = DraftGeometry.from_record(record)
            self.max_packs = max(self.max_packs, geo.packs)
            self.max_pack_size = max(self.max_pack_size, geo.pack_size)
            rewards = leave_one_out_rewards(record)
            for seat_idx, seat in enumerate(record.seats):
                reward = rewards[seat_idx]
                if reward is None:
                    continue  # failed build → excluded entirely (FR-015)
                is_learner = allow_learner and seat.agent in self._learner
                traj: list[RLExample] = []
                for state in iter_seat_pick_states(
                    geo, record.boosters, seat_idx, self._card_index,
                ):
                    ex = RLExample(
                        draft_index=ri,
                        card_idx=state.card_idx,
                        type_idx=state.type_idx,
                        packs_ago=state.packs_ago,
                        pick_ago=state.pick_ago,
                        pack_number=state.pack_number,
                        pick_number=state.pick_number,
                        action_token=state.action_position,
                        learner_active=is_learner and state.action_position >= 0,
                        critic_active=True,
                        reward=float(reward),
                    )
                    examples.append(ex)
                    if is_learner:
                        traj.append(ex)
                if traj:
                    trajectories.append(traj)


def split_draft_disjoint(
    examples: list[RLExample], val_fraction: float, seed: int = RANDOM_SEED,
) -> tuple[set[int], set[int]]:
    """Partition draft indices so a draft is entirely train or val (FR-035)."""
    ids = sorted({ex.draft_index for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    train_ids = {i for i in ids if i not in val_ids}
    return train_ids, val_ids
