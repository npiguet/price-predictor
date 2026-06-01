"""Loss gating, leave-one-out reward, and critic z-scoring (FR-032, FR-033)."""

from __future__ import annotations

import numpy as np

from draft.application.train_draft_agent import (
    _leave_one_out_rewards,
    _Loader,
    critic_standardization,
)
from draft.domain.draft_geometry import Booster, DraftRecord, Seat


class _FakeLocator:
    """Returns a deterministic vector per known name; None for 'missing'."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def load_embedding(self, name: str):
        if name == "missing":
            return None
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        return rng.standard_normal(self._dim).astype(np.float32)


def _record() -> DraftRecord:
    # pod_size=4, packs=1, P=2.  Seats: whitelisted+ok, other+ok, whitelisted+failed, other+ok.
    seats = [
        Seat("forge-full", ["x"] * 40, 10.0),
        Seat("forge-r30", ["x"] * 40, 6.0),
        Seat("forge-full", [], None),   # failed build
        Seat("forge-r100", ["x"] * 40, 2.0),
    ]
    boosters = [
        Booster("TST", [f"k{k}c{j}" for j in range(2)]) for k in range(4)
    ]
    return DraftRecord("d", "r", "t", seats, boosters)


def test_leave_one_out_excludes_failed_seats() -> None:
    rewards = _leave_one_out_rewards(_record())
    assert rewards[0] == 10.0 - (6.0 + 2.0) / 2  # 6.0
    assert rewards[1] == 6.0 - (10.0 + 2.0) / 2  # 0.0
    assert rewards[2] is None                    # failed build, no critic
    assert rewards[3] == 2.0 - (10.0 + 6.0) / 2  # -6.0


def test_loss_gating_flags() -> None:
    loader = _Loader(_FakeLocator(), whitelist={"forge-full"})
    examples = loader.build([_record()])
    by_seat: dict[tuple[bool, bool], int] = {}
    # Group examples by (imitation_active, critic_active); every (seat,pack,pick)
    # of the same seat shares those flags.
    for ex in examples:
        by_seat[(ex.imitation_active, ex.critic_active)] = (
            by_seat.get((ex.imitation_active, ex.critic_active), 0) + 1
        )
    # seat0: whitelisted + ok -> (True, True)
    assert (True, True) in by_seat
    # seat1/seat3: not whitelisted + ok -> (False, True)
    assert (False, True) in by_seat
    # seat2: whitelisted + failed -> imitation active, critic inactive (True, False)
    assert (True, False) in by_seat


def test_whitelisted_failed_seat_has_imitation_not_critic() -> None:
    loader = _Loader(_FakeLocator(), whitelist={"forge-full"})
    examples = loader.build([_record()])
    # Critic targets are only emitted for non-failed seats.
    critic_examples = [ex for ex in examples if ex.critic_active]
    assert all(ex.critic_target is not None for ex in critic_examples)
    # The failed whitelisted seat (seat 2) still produced imitation-active examples.
    imitation_only = [ex for ex in examples if ex.imitation_active and not ex.critic_active]
    assert imitation_only


def test_critic_standardization_round_trip() -> None:
    loader = _Loader(_FakeLocator(), whitelist={"forge-full"})
    examples = [ex for ex in loader.build([_record()]) if ex.critic_active]
    mean, std = critic_standardization(examples)
    standardized = [(ex.critic_target - mean) / std for ex in examples]
    # De-standardize round-trips back to the raw targets.
    restored = [s * std + mean for s in standardized]
    raw = [ex.critic_target for ex in examples]
    assert np.allclose(restored, raw)
    # Standardized targets are ~zero mean / unit variance.
    assert abs(float(np.mean(standardized))) < 1e-6
    assert abs(float(np.std(standardized)) - 1.0) < 1e-6
