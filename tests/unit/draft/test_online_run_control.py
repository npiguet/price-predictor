"""Gen-3 run control: the shared stall counter and plateau LR annealing.

Covers data-model §7 / spec FR-034, FR-035 and research D16. Patience and
annealing key off one counter — rounds since the last new best anchor margin —
so a decay resets it and an armed patience can only fire at the LR floor.
"""

from __future__ import annotations

import pytest
import torch

from draft.application.train_draft_agent import _maybe_decay, _PlateauLR
from draft.application.train_draft_agent_online import AnchorWindow, resolve_min_lr
from draft.domain.draft_geometry import Booster, DraftRecord, Seat

LEARNER = "gen-3"
ANCHOR = "gen-1"


def _draft(learner: float | None, anchor: float | None) -> DraftRecord:
    seats = []
    for label, score in ((LEARNER, learner), (ANCHOR, anchor)):
        if score is not None:
            seats.append(Seat(label, ["x"] * 40, score))
    return DraftRecord("d", "r", "t", seats, [Booster("TST", ["c0", "c1"])])


def _window(maxlen: int = 4) -> AnchorWindow:
    return AnchorWindow(maxlen, LEARNER, ANCHOR)


def _fill(window: AnchorWindow, n: int, learner: float = 1.0) -> None:
    """Populate the window to exactly full at a known margin."""
    for _ in range(n):
        window.add(_draft(learner, 0.0))


# --------------------------------------------------------------------------- #
# The stall counter (data-model § 7)
# --------------------------------------------------------------------------- #

def test_counter_stays_put_until_the_first_best_is_recorded() -> None:
    """Before the window fills there is no best, so nothing is being stalled on."""
    window = _window(maxlen=4)
    for round_index in range(3):
        window.add(_draft(1.0, 0.0))
        window.observe_round(round_index)
        assert window.rounds_since_best == 0
        assert window.best_margin is None


def test_counter_resets_on_a_new_best() -> None:
    window = _window(maxlen=2)
    _fill(window, 2, learner=1.0)
    assert window.observe_round(0) is True          # first best
    assert window.rounds_since_best == 0

    window.add(_draft(0.0, 0.0))                    # margin falls
    assert window.observe_round(1) is False
    assert window.rounds_since_best == 1

    window.add(_draft(5.0, 0.0))                    # new high margin
    assert window.observe_round(2) is True
    assert window.rounds_since_best == 0


def test_counter_increments_once_per_stalled_round() -> None:
    window = _window(maxlen=2)
    _fill(window, 2, learner=1.0)
    window.observe_round(0)

    for expected in (1, 2, 3):
        window.add(_draft(0.5, 0.0))                # never beats the first best
        window.observe_round(expected)
        assert window.rounds_since_best == expected


def test_a_decay_resets_the_counter() -> None:
    """The cooldown that lets annealing pre-empt stopping (FR-035)."""
    window = _window(maxlen=2)
    _fill(window, 2, learner=1.0)
    window.observe_round(0)
    for r in (1, 2):
        window.add(_draft(0.5, 0.0))
        window.observe_round(r)
    assert window.rounds_since_best == 2

    window.note_lr_decay()
    assert window.rounds_since_best == 0
    assert window.best_margin is not None  # the best itself is untouched


# --------------------------------------------------------------------------- #
# Annealing (spec FR-035, gen-1's _PlateauLR reused)
# --------------------------------------------------------------------------- #

def test_decay_multiplies_the_lr_by_the_factor() -> None:
    plateau = _PlateauLR(base_lr=1e-4, factor=0.1, min_lr=1e-7)
    assert plateau.current_lr() == pytest.approx(1e-4)
    assert _maybe_decay(3, 3, plateau) == pytest.approx(1e-5)
    assert _maybe_decay(3, 3, plateau) == pytest.approx(1e-6)


def test_no_decay_before_the_patience_is_reached() -> None:
    plateau = _PlateauLR(base_lr=1e-4, factor=0.1, min_lr=1e-7)
    assert _maybe_decay(2, 3, plateau) is None
    assert plateau.decay_count == 0


def test_no_decay_lands_below_the_floor() -> None:
    plateau = _PlateauLR(base_lr=1e-4, factor=0.1, min_lr=1e-6)
    assert _maybe_decay(5, 1, plateau) == pytest.approx(1e-5)
    assert _maybe_decay(5, 1, plateau) == pytest.approx(1e-6)
    # A third decay would land at 1e-7, below the floor.
    assert _maybe_decay(5, 1, plateau) is None
    assert plateau.current_lr() == pytest.approx(1e-6)


def test_disabled_patience_never_decays() -> None:
    plateau = _PlateauLR(base_lr=1e-4, factor=0.1, min_lr=1e-9)
    assert _maybe_decay(999, None, plateau) is None
    assert plateau.decay_count == 0


def test_min_lr_defaults_to_a_thousandth_of_the_base() -> None:
    assert resolve_min_lr(lr=1e-4, min_lr=None) == pytest.approx(1e-7)
    assert resolve_min_lr(lr=1e-4, min_lr=5e-6) == pytest.approx(5e-6)


def test_the_multiplier_folds_into_the_warmup_schedule() -> None:
    """A decay scales the post-warmup constant; warmup is not re-run (FR-035)."""
    from draft.application.train_draft_agent import _make_scheduler

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": 1e-4, "name": "agent"}],
    )
    plateau = _PlateauLR(base_lr=1e-4, factor=0.1, min_lr=1e-7)
    scheduler = _make_scheduler(
        optimizer, total_steps=4, warmup_frac=1.0, controller=plateau,
    )

    def step() -> None:
        """One optimizer step then one scheduler step, as the loop does."""
        model(torch.zeros(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    # Ramp to the constant.
    for _ in range(4):
        step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)

    plateau.decay()
    step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    # Still constant afterwards — no second ramp.
    step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


# --------------------------------------------------------------------------- #
# The interaction that makes annealing worth having (SC-010)
# --------------------------------------------------------------------------- #

def test_annealing_preempts_stopping_until_the_floor() -> None:
    """An armed patience must not fire while a further decay is possible."""
    window = _window(maxlen=1)
    window.add(_draft(1.0, 0.0))
    window.observe_round(0)

    plateau = _PlateauLR(base_lr=1e-4, factor=0.1, min_lr=1e-6)
    patience, decay_patience = 6, 3
    stopped_at = None

    for round_index in range(1, 40):
        window.add(_draft(0.5, 0.0))              # permanently stalled
        window.observe_round(round_index)
        if _maybe_decay(window.rounds_since_best, decay_patience, plateau) is not None:
            window.note_lr_decay()
            continue
        if window.rounds_since_best >= patience:
            stopped_at = round_index
            break

    assert stopped_at is not None
    # Two decays were available (1e-5, 1e-6); the stop waited for the floor.
    assert plateau.decay_count == 2
    assert plateau.current_lr() == pytest.approx(1e-6)


def test_with_patience_disabled_the_run_anneals_and_continues() -> None:
    window = _window(maxlen=1)
    window.add(_draft(1.0, 0.0))
    window.observe_round(0)

    plateau = _PlateauLR(base_lr=1e-4, factor=0.1, min_lr=1e-6)
    for round_index in range(1, 30):
        window.add(_draft(0.5, 0.0))
        window.observe_round(round_index)
        if _maybe_decay(window.rounds_since_best, 3, plateau) is not None:
            window.note_lr_decay()
        assert window.rounds_since_best < 30  # nothing ever stops it

    assert plateau.current_lr() == pytest.approx(1e-6)
