"""Gen-3 per-round diagnostics: exploration + movement sweep, and the anchor window.

Covers data-model §6/§7, research D9, and spec FR-015/FR-016/FR-017/FR-019/FR-021.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from draft.application.train_draft_agent_online import (
    AnchorWindow,
    OnlineExample,
    diagnostics_sweep,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.domain.draft_geometry import Booster, DraftRecord, Seat
from draft.domain.draft_state import TYPE_PACK, TYPE_POOL

_DIM = 8


def _config() -> DraftAgentConfig:
    return DraftAgentConfig(
        embedding_dim=_DIM, packs=3, P=4, n_layers=1, n_heads=2,
    )


def _example(*, n_pool: int = 2, n_pack: int = 3, action_offset: int = 0) -> OnlineExample:
    n = n_pool + n_pack
    return OnlineExample(
        card_idx=np.arange(n, dtype=np.int32),
        type_idx=np.array([TYPE_POOL] * n_pool + [TYPE_PACK] * n_pack, dtype=np.int8),
        packs_ago=np.zeros(n, dtype=np.int8),
        pick_ago=np.zeros(n, dtype=np.int8),
        pack_number=1,
        pick_number=1,
        action_token=n_pool + action_offset,
        advantage=0.0,
    )


def _table(rows: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((rows, _DIM)).astype(np.float32)


def _models() -> tuple[DraftAgentModel, DraftAgentModel]:
    torch.manual_seed(0)
    prev = DraftAgentModel(_config())
    new = DraftAgentModel(_config())
    new.load_state_dict(prev.state_dict())
    prev.eval()
    new.eval()
    return prev, new


# --------------------------------------------------------------------------- #
# Exploration + movement sweep (research D9)
# --------------------------------------------------------------------------- #

def test_perplexity_is_the_exponential_of_entropy() -> None:
    prev, new = _models()
    examples = [_example(), _example(action_offset=1), _example(action_offset=2)]

    diag = diagnostics_sweep(
        prev, new, examples, _table(), batch_size=2, temperature=1.0,
        device=torch.device("cpu"),
    )

    assert diag.perplexity == math.exp(diag.entropy)
    assert diag.entropy > 0.0


def test_kl_is_zero_when_the_weights_did_not_move() -> None:
    prev, new = _models()  # identical state dicts
    examples = [_example(), _example(action_offset=2)]

    diag = diagnostics_sweep(
        prev, new, examples, _table(), batch_size=8, temperature=1.0,
        device=torch.device("cpu"),
    )

    assert abs(diag.kl_prev_new) < 1e-6


def test_kl_is_positive_after_a_step() -> None:
    prev, new = _models()
    with torch.no_grad():
        for p in new.policy_head.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    examples = [_example(), _example(action_offset=1)]

    diag = diagnostics_sweep(
        prev, new, examples, _table(), batch_size=8, temperature=1.0,
        device=torch.device("cpu"),
    )

    assert diag.kl_prev_new > 0.0


def test_off_argmax_rate_counts_picks_that_missed_the_argmax() -> None:
    """Measured against pi_k over the same PACK set the action came from."""
    prev, new = _models()
    examples = [_example(action_offset=o) for o in (0, 1, 2)]
    table = _table()
    device = torch.device("cpu")

    diag = diagnostics_sweep(
        prev, new, examples, table, batch_size=8, temperature=1.0, device=device,
    )

    # Recompute the argmax choice per example directly from pi_k.
    from draft.application.train_draft_agent_online import _collate

    off = 0
    for ex in examples:
        batch = _collate([ex], table, device)
        with torch.no_grad():
            logits, _ = prev(
                batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
                batch.card_mask, batch.pack_number, batch.pick_number,
            )
        masked = logits.masked_fill(~batch.pack_mask, float("-inf"))
        if int(masked.argmax(dim=-1).item()) != ex.action_token:
            off += 1
    assert diag.off_argmax_rate == off / len(examples)


def test_off_argmax_rate_is_zero_when_every_action_is_the_argmax() -> None:
    prev, new = _models()
    table = _table()
    device = torch.device("cpu")
    from draft.application.train_draft_agent_online import _collate

    # Build each example, then relabel its action as pi_k's argmax.
    examples = []
    for offset in (0, 1, 2):
        ex = _example(action_offset=offset)
        batch = _collate([ex], table, device)
        with torch.no_grad():
            logits, _ = prev(
                batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
                batch.card_mask, batch.pack_number, batch.pick_number,
            )
        masked = logits.masked_fill(~batch.pack_mask, float("-inf"))
        ex.action_token = int(masked.argmax(dim=-1).item())
        examples.append(ex)

    diag = diagnostics_sweep(
        prev, new, examples, table, batch_size=8, temperature=1.0, device=device,
    )
    assert diag.off_argmax_rate == 0.0


def test_mean_logp_averages_the_taken_actions_under_the_generating_policy() -> None:
    prev, new = _models()
    examples = [_example(action_offset=o) for o in (0, 1, 2)]
    table = _table()
    device = torch.device("cpu")

    diag = diagnostics_sweep(
        prev, new, examples, table, batch_size=2, temperature=1.0, device=device,
    )

    from draft.application.train_draft_agent_online import _collate
    from price_predictor.infrastructure.torch_training import masked_log_softmax

    batch = _collate(examples, table, device)
    with torch.no_grad():
        logits, _ = prev(
            batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
            batch.card_mask, batch.pack_number, batch.pick_number,
        )
        logp = masked_log_softmax(logits, batch.pack_mask)
        expected = float(
            logp.gather(1, batch.action_token.unsqueeze(1)).squeeze(1).mean()
        )
    assert abs(diag.mean_logp - expected) < 1e-5
    assert diag.mean_logp < 0.0


def test_sweep_is_batch_size_invariant() -> None:
    prev, new = _models()
    examples = [_example(action_offset=o % 3) for o in range(7)]
    table = _table()

    a = diagnostics_sweep(
        prev, new, examples, table, batch_size=2, temperature=1.3,
        device=torch.device("cpu"),
    )
    b = diagnostics_sweep(
        prev, new, examples, table, batch_size=16, temperature=1.3,
        device=torch.device("cpu"),
    )

    assert abs(a.entropy - b.entropy) < 1e-5
    assert abs(a.mean_logp - b.mean_logp) < 1e-5
    assert a.off_argmax_rate == b.off_argmax_rate


def test_empty_example_list_is_a_safe_zero_sweep() -> None:
    prev, new = _models()
    diag = diagnostics_sweep(
        prev, new, [], _table(), batch_size=8, temperature=1.0,
        device=torch.device("cpu"),
    )
    assert diag.entropy == 0.0
    assert diag.off_argmax_rate == 0.0
    assert diag.kl_prev_new == 0.0
    assert diag.kl_init_new == 0.0


# ── cumulative KL from the run's warm start (FR-016, research D17) ───────────

def _three_models() -> tuple[DraftAgentModel, DraftAgentModel, DraftAgentModel]:
    """(init, prev, new) — all sharing weights, all in eval()."""
    torch.manual_seed(0)
    init = DraftAgentModel(_config())
    prev, new = DraftAgentModel(_config()), DraftAgentModel(_config())
    prev.load_state_dict(init.state_dict())
    new.load_state_dict(init.state_dict())
    for m in (init, prev, new):
        m.eval()
    return init, prev, new


def _nudge(model: DraftAgentModel, scale: float = 0.5) -> None:
    with torch.no_grad():
        for p in model.policy_head.parameters():
            p.add_(torch.randn_like(p) * scale)


def _sweep(init, prev, new, examples, **kw):
    defaults = {"batch_size": 8, "temperature": 1.0, "device": torch.device("cpu")}
    defaults.update(kw)
    return diagnostics_sweep(prev, new, examples, _table(), init_model=init, **defaults)


def test_cumulative_kl_is_zero_when_nothing_has_moved() -> None:
    init, prev, new = _three_models()
    diag = _sweep(init, prev, new, [_example(), _example(action_offset=1)])
    assert abs(diag.kl_init_new) < 1e-6


def test_cumulative_kl_is_positive_once_the_learner_has_moved() -> None:
    init, prev, new = _three_models()
    _nudge(new)
    diag = _sweep(init, prev, new, [_example(), _example(action_offset=2)])
    assert diag.kl_init_new > 0.0


def test_cumulative_kl_ignores_prev_model() -> None:
    """It measures distance from pi_0, not from last round (that is kl_prev_new)."""
    init, prev, new = _three_models()
    examples = [_example(), _example(action_offset=1)]
    before = _sweep(init, prev, new, examples).kl_init_new

    _nudge(prev)  # only the previous-round reference moves
    after = _sweep(init, prev, new, examples)

    assert after.kl_init_new == pytest.approx(before, abs=1e-6)
    assert after.kl_prev_new > 0.0  # …while the per-round KL does react


def test_cumulative_kl_equals_per_round_kl_when_init_and_prev_coincide() -> None:
    """One round after the warm start, the two questions have the same answer."""
    init, prev, new = _three_models()
    _nudge(new)
    diag = _sweep(init, prev, new, [_example(action_offset=o) for o in (0, 1, 2)])
    assert diag.kl_init_new == pytest.approx(diag.kl_prev_new, abs=1e-6)


def test_cumulative_kl_can_grow_while_per_round_kl_stays_tiny() -> None:
    """The failure mode this exists for: small consistent steps that accumulate."""
    init, prev, new = _three_models()
    examples = [_example(action_offset=o) for o in (0, 1, 2)]

    per_round = []
    for _ in range(12):
        prev.load_state_dict(new.state_dict())
        _nudge(new, scale=0.02)  # a small step, always in a fresh direction
        diag = _sweep(init, prev, new, examples)
        per_round.append(diag.kl_prev_new)

    assert max(per_round) < 10 * diag.kl_init_new  # each step is small…
    assert diag.kl_init_new > max(per_round)       # …yet the total is larger


def test_cumulative_kl_is_batch_size_invariant() -> None:
    init, prev, new = _three_models()
    _nudge(new)
    examples = [_example(action_offset=o % 3) for o in range(7)]

    a = _sweep(init, prev, new, examples, batch_size=2, temperature=1.3)
    b = _sweep(init, prev, new, examples, batch_size=16, temperature=1.3)
    assert a.kl_init_new == pytest.approx(b.kl_init_new, abs=1e-5)


def test_sweep_without_an_init_model_reports_zero_cumulative_kl() -> None:
    """The parameter is optional so existing call sites keep working."""
    prev, new = _models()
    diag = diagnostics_sweep(
        prev, new, [_example()], _table(), batch_size=8, temperature=1.0,
        device=torch.device("cpu"),
    )
    assert diag.kl_init_new == 0.0


# --------------------------------------------------------------------------- #
# Anchor window (data-model §7, FR-017/FR-019/FR-021)
# --------------------------------------------------------------------------- #

def _draft(scores: dict[str, list[float | None]]) -> DraftRecord:
    seats = [
        Seat(label, ["x"] * 40 if s is not None else [], s)
        for label, values in scores.items()
        for s in values
    ]
    return DraftRecord("d", "r", "t", seats, [Booster("TST", ["c0", "c1"])])


def test_margin_is_learner_mean_minus_anchor_mean() -> None:
    window = AnchorWindow(maxlen=100, learner_label="gen-3", anchor_label="gen-1")
    window.add(_draft({"gen-3": [6.0, 8.0], "gen-1": [4.0, 6.0]}))

    assert window.label_mean("gen-3") == 7.0
    assert window.label_mean("gen-1") == 5.0
    assert window.margin == 2.0
    assert window.window_drafts == 1


def test_margin_is_none_until_both_labels_have_a_scored_seat() -> None:
    window = AnchorWindow(maxlen=100, learner_label="gen-3", anchor_label="gen-1")
    assert window.margin is None

    window.add(_draft({"gen-3": [6.0], "forge-r30": [3.0]}))
    assert window.margin is None
    assert window.label_mean("gen-1") is None

    window.add(_draft({"gen-1": [4.0]}))
    assert window.margin == 2.0


def test_failed_builds_are_excluded_from_every_mean() -> None:
    window = AnchorWindow(maxlen=100, learner_label="gen-3", anchor_label="gen-1")
    window.add(_draft({"gen-3": [6.0, None, 8.0], "gen-1": [5.0, None]}))

    assert window.label_mean("gen-3") == 7.0
    assert window.label_mean("gen-1") == 5.0


def test_every_label_is_tracked_not_just_learner_and_anchor() -> None:
    window = AnchorWindow(maxlen=100, learner_label="gen-3", anchor_label="gen-1")
    window.add(_draft({
        "gen-3": [6.0], "gen-1": [5.0], "forge-r30": [3.0], "forge-r100": [1.0],
    }))

    means = window.label_means()
    assert set(means) == {"gen-3", "gen-1", "forge-r30", "forge-r100"}
    assert means["forge-r100"] == 1.0


def test_the_window_evicts_the_oldest_draft() -> None:
    window = AnchorWindow(maxlen=2, learner_label="gen-3", anchor_label="gen-1")
    window.add(_draft({"gen-3": [1.0], "gen-1": [0.0]}))
    window.add(_draft({"gen-3": [2.0], "gen-1": [0.0]}))
    window.add(_draft({"gen-3": [3.0], "gen-1": [0.0]}))

    assert window.window_drafts == 2
    assert window.label_mean("gen-3") == 2.5  # the 1.0 draft was evicted


def test_best_margin_and_its_round_are_tracked() -> None:
    # maxlen=1 so the window is full from the first draft and every round is
    # eligible — the window-full guard is exercised separately below.
    window = AnchorWindow(maxlen=1, learner_label="gen-3", anchor_label="gen-1")

    window.add(_draft({"gen-3": [6.0], "gen-1": [5.0]}))
    window.observe_round(0)
    window.add(_draft({"gen-3": [10.0], "gen-1": [5.0]}))
    window.observe_round(1)
    window.add(_draft({"gen-3": [1.0], "gen-1": [5.0]}))
    window.observe_round(2)

    # Round 1's margin (10.0 - 5.0) is the best seen; round 2's is negative.
    assert window.best_margin == 5.0
    assert window.best_round == 1
    assert window.margin is not None and window.margin < window.best_margin


def test_observe_round_before_any_margin_leaves_the_best_unset() -> None:
    window = AnchorWindow(maxlen=100, learner_label="gen-3", anchor_label="gen-1")
    window.add(_draft({"gen-3": [6.0]}))
    window.observe_round(0)

    assert window.best_margin is None
    assert window.best_round is None


# ── window-full guard (FR-033, SC-009) ───────────────────────────────────────

def test_no_best_is_recorded_while_the_window_is_partly_filled() -> None:
    """An early lucky round must not pin the run's best (FR-033)."""
    window = AnchorWindow(maxlen=4, learner_label="gen-3", anchor_label="gen-1")

    # Three very high margins, all on a partly-filled window.
    for round_index in range(3):
        window.add(_draft({"gen-3": [99.0], "gen-1": [0.0]}))
        assert window.observe_round(round_index) is False
        assert window.best_margin is None
        assert window.margin is not None  # still printed — it is the live read


def test_the_first_best_is_recorded_on_the_first_full_window_round() -> None:
    window = AnchorWindow(maxlen=3, learner_label="gen-3", anchor_label="gen-1")
    for round_index in range(2):
        window.add(_draft({"gen-3": [5.0], "gen-1": [1.0]}))
        window.observe_round(round_index)
    assert window.best_margin is None

    window.add(_draft({"gen-3": [5.0], "gen-1": [1.0]}))
    assert window.window_drafts == 3
    assert window.observe_round(2) is True
    assert window.best_margin == 4.0
    assert window.best_round == 2


def test_a_run_shorter_than_the_window_records_no_best() -> None:
    window = AnchorWindow(maxlen=50, learner_label="gen-3", anchor_label="gen-1")
    for round_index in range(5):
        window.add(_draft({"gen-3": [7.0], "gen-1": [1.0]}))
        window.observe_round(round_index)

    assert window.best_margin is None
    assert window.best_round is None


def test_an_early_high_margin_cannot_beat_a_later_full_window_best() -> None:
    """The guard's whole point: the partial-window spike is simply not eligible."""
    window = AnchorWindow(maxlen=2, learner_label="gen-3", anchor_label="gen-1")
    window.add(_draft({"gen-3": [50.0], "gen-1": [0.0]}))   # huge, partial window
    window.observe_round(0)
    assert window.best_margin is None

    window.add(_draft({"gen-3": [1.0], "gen-1": [0.0]}))    # window now full
    window.observe_round(1)
    assert window.best_margin == pytest.approx(25.5)        # mean(50, 1) - 0
    assert window.best_round == 1
