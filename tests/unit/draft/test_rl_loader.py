"""RL loader: trajectories, learner/critic activation, failed builds, multi-corpus."""

from __future__ import annotations

import numpy as np

from draft.application.train_draft_agent_rl import _Loader
from draft.domain.draft_geometry import Booster, DraftRecord, Seat


class _FakeLocator:
    """Deterministic per-name vector; every card present (no drops)."""

    def __init__(self, dim: int = 6) -> None:
        self._dim = dim
        self._tbl: dict[str, np.ndarray] = {}

    def load_embedding(self, name: str) -> np.ndarray:
        if name not in self._tbl:
            self._tbl[name] = np.random.default_rng(
                abs(hash(name)) % 2**32,
            ).standard_normal(self._dim).astype(np.float32)
        return self._tbl[name]


def _record(
    agents: list[str], scores: list[float | None], pod: int, packs: int, P: int,
) -> DraftRecord:
    seats = [Seat(agents[s], [], scores[s]) for s in range(pod)]
    boosters = [
        Booster("TST", [f"k{k}_{j}" for j in range(P)]) for k in range(pod * packs)
    ]
    return DraftRecord("d", "r", "t", seats, boosters)


def test_learner_trajectory_and_activation() -> None:
    # pod 2, 1 pack of 4: seat0 is the learner, seat1 is Forge coverage.
    rec = _record(["gen-k", "forge-full"], [2.0, 1.0], pod=2, packs=1, P=4)
    loader = _Loader(_FakeLocator(), {"gen-k"})
    examples, trajectories = loader.build([rec], [])

    assert len(examples) == 8                 # 2 seats × 4 picks, both have rewards
    assert len(trajectories) == 1             # only the learner seat
    assert len(trajectories[0]) == 4          # one per pick
    # Trajectory is ordered by (pack, pick).
    assert [e.pick_number for e in trajectories[0]] == [1, 2, 3, 4]

    learner = [e for e in examples if e.learner_active]
    assert len(learner) == 4                  # only seat0's picks feed policy grad
    assert all(e.critic_active for e in examples)  # both seats feed the critic
    # Leave-one-out reward: seat0 = 2.0 − 1.0 = 1.0.
    assert trajectories[0][0].reward == 1.0


def test_failed_build_seat_is_excluded() -> None:
    # seat1 has a failed build (deck_score None) → excluded entirely (FR-015).
    rec = _record(["gen-k", "forge-full"], [2.0, None], pod=2, packs=1, P=4)
    loader = _Loader(_FakeLocator(), {"gen-k"})
    examples, trajectories = loader.build([rec], [])
    assert len(examples) == 4                  # only seat0
    assert len(trajectories) == 1
    # Pod mean has no other non-failed seat → seat0 reward = 2.0 − 0.0.
    assert trajectories[0][0].reward == 2.0


def test_critic_corpus_never_creates_learner_trajectories() -> None:
    on_policy = _record(["gen-k", "forge-full"], [2.0, 1.0], pod=2, packs=1, P=4)
    # A second corpus, even with a gen-k seat, contributes critic coverage only.
    critic = _record(["gen-k", "forge-full"], [3.0, 0.0], pod=2, packs=1, P=4)
    loader = _Loader(_FakeLocator(), {"gen-k"})
    examples, trajectories = loader.build([on_policy], [critic])

    assert len(trajectories) == 1              # only the on-policy learner seat
    # Critic-corpus examples exist but none are learner_active.
    assert len(examples) == 16                 # (2+2 seats) × 4 picks
    assert sum(e.learner_active for e in examples) == 4  # only on-policy seat0
