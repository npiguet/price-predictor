"""Two full online-GRPO rounds end-to-end against a fake worker (no JVM, no GPU).

Drives :class:`TrainDraftAgentOnlineUseCase` over stub Forge transcripts using
the ``tests/integration/test_draft_supervisor_restart.py`` fake-worker pattern.
Asserts the four diagnostic axes are logged every round, the checkpoint is a
loadable gen-3 one, round 2 trained on drafts generated *after* round 1's update
(SC-008), and the corpus grew by exactly ``2 x --drafts-per-round`` records.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from draft.application import train_draft_agent_online as online
from draft.application.generate_draft_data import POD_SIZE, SENTINEL
from draft.application.train_draft_agent_online import (
    TrainDraftAgentOnlineConfig,
    TrainDraftAgentOnlineUseCase,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records

pytestmark = pytest.mark.integration

EMB_DIM = 8
PACKS = 3
PACK_SIZE = 4
DRAFTS_PER_ROUND = 2
LEARNER = "gen-3"
ANCHOR = "gen-1"


class _FakeProc:
    """A stub worker yielding transcripts, then EOF."""

    def __init__(self, lines: list[str]) -> None:
        self.stdout = iter(lines)

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def _cards() -> list[str]:
    """Every distinct card name the fake boosters use."""
    return [f"c{i}" for i in range(PACKS * POD_SIZE * PACK_SIZE)]


def _transcript(draft_id: str) -> str:
    """A full-geometry pod: 8 seats, 3 packs, 4 cards per booster."""
    names = _cards()
    boosters = []
    for k in range(PACKS * POD_SIZE):
        picks = names[k * PACK_SIZE:(k + 1) * PACK_SIZE]
        boosters.append({"set_code": "TST", "picks": picks})
    # A learner-heavy pod with the frozen anchor and a Forge bot present, so the
    # anchor margin is defined and the leave-one-out baseline is populated.
    agents = [LEARNER, ANCHOR, LEARNER, ANCHOR, LEARNER, "forge-r30", LEARNER, ANCHOR]
    return SENTINEL + json.dumps({
        "draft_id": draft_id,
        "boosters": boosters,
        "seats": [{"agent": a} for a in agents],
    }) + "\n"


class _VaryingLabeler:
    """Scores each seat differently so the round's advantages are non-degenerate."""

    def __init__(self) -> None:
        self.calls = 0

    def build_and_score(self, pool):
        self.calls += 1
        # A spread of scores across the pod keeps reward std well above the
        # degenerate-round epsilon.
        return ["Plains"] * 40, float(self.calls % 7) + 0.5


@pytest.fixture
def cards_path(tmp_path: Path) -> Path:
    """A .npz embedding cache covering every card the fake boosters use."""
    folder = tmp_path / "cardsfolder"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for name in _cards():
        np.savez(
            folder / f"{name}.npz",
            embedding=rng.standard_normal(EMB_DIM).astype(np.float32),
        )
        (folder / f"{name}.txt").write_text(f"name:{name}\n", encoding="utf-8")
    return folder


@pytest.fixture
def base_checkpoint(tmp_path: Path) -> Path:
    config = DraftAgentConfig(
        embedding_dim=EMB_DIM, packs=PACKS, P=PACK_SIZE, n_layers=1, n_heads=1,
    )
    model = DraftAgentModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "gen1.pt"
    DraftAgentStore().save_checkpoint(
        model, optimizer, 0, 0.0, config, path,
        critic_mean=2.5, critic_std=1.5,
    )
    return path


@pytest.fixture
def run(
    tmp_path: Path, cards_path: Path, base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch, capsys,
):
    """Drive two rounds and return (result, captured stdout, paths)."""
    checkpoint_dir = tmp_path / "models"
    monkeypatch.setattr(online, "CHECKPOINT_DIR", checkpoint_dir)

    launched: list[int] = []
    generated: list[str] = []

    def launch():
        launched.append(len(launched))
        # One worker serves every round: enough transcripts for both rounds.
        lines = []
        for i in range(DRAFTS_PER_ROUND * 4):
            draft_id = f"d{i}"
            generated.append(draft_id)
            lines.append(_transcript(draft_id))
        return _FakeProc(lines)

    labeler = _VaryingLabeler()
    monkeypatch.setattr(
        "draft.application.generate_draft_data.build_labeler",
        lambda config, *, locator=None: labeler,
    )
    monkeypatch.setattr(
        "draft.application.generate_draft_data.GenerateDraftDataSupervisor."
        "_default_launch_worker",
        lambda self, external_labels: launch,
    )

    corpus = tmp_path / "drafts.jsonl"
    corpus.write_text("", encoding="utf-8")
    config = TrainDraftAgentOnlineConfig(
        learner_label=LEARNER,
        learner_checkpoint=base_checkpoint,
        frozen={ANCHOR: base_checkpoint},
        mix=[(LEARNER, 5), (ANCHOR, 3), ("forge-r30", 1)],
        scorer_checkpoint=base_checkpoint,  # existence-checked only; labeler is stubbed
        build_method="greedy",
        cards_path=cards_path,
        rollout_temperature=2.0,
        drafts_per_round=DRAFTS_PER_ROUND,
        anchor_window=100,
        snapshot_every=100,   # only the run-end snapshot fires
        max_rounds=2,
        batch_size=8,
        output_path=corpus,
    )

    rounds = TrainDraftAgentOnlineUseCase().execute(config)
    out = capsys.readouterr().out
    return rounds, out, corpus, checkpoint_dir, launched


def test_two_rounds_complete(run) -> None:
    rounds, _, _, _, _ = run
    assert rounds == 2


def test_one_resident_worker_serves_both_rounds(run) -> None:
    """Forge's JVM startup is paid once per run, not once per round (FR-005)."""
    _, _, _, _, launched = run
    assert len(launched) == 1


def test_every_round_logs_all_four_axes(run) -> None:
    _, out, _, _, _ = run
    for axis in ("reward   :", "explore  :", "movement :", "progress :"):
        assert out.count(axis) == 2, f"expected {axis} once per round"
    assert "round 0 |" in out
    assert "round 1 |" in out


def test_startup_echo_precedes_the_rounds(run) -> None:
    _, out, _, _, _ = run
    assert "Online GRPO run" in out
    assert "generation 1 -> 2" in out
    assert out.index("Online GRPO run") < out.index("round 0 |")


def test_final_summary_is_printed(run) -> None:
    _, out, _, _, _ = run
    assert "Done after 2 rounds" in out
    assert "latest checkpoint :" in out
    assert "best anchor margin:" in out


def test_latest_checkpoint_is_a_loadable_gen3_agent(run) -> None:
    _, _, _, checkpoint_dir, _ = run
    latest = checkpoint_dir / "latest.pt"
    assert latest.exists()

    ckpt = DraftAgentStore().load_checkpoint(latest)
    assert ckpt.rl_metadata is not None
    assert ckpt.rl_metadata["algorithm"] == "online-grpo"
    assert ckpt.rl_metadata["generation"] == 2
    assert ckpt.rl_metadata["rollout_temperature"] == 2.0
    assert ckpt.rl_metadata["drafts_per_round"] == DRAFTS_PER_ROUND
    # No held-out metric exists, and the critic standardization is carried through.
    assert ckpt.best_val_loss == float("inf")
    assert ckpt.critic_mean == 2.5
    assert ckpt.critic_std == 1.5
    # Gen-1 payload shape: the critic head is still there.
    assert any(k.startswith("critic_head") for k in ckpt.model_state_dict)


def test_a_run_end_snapshot_is_written(run) -> None:
    _, _, _, checkpoint_dir, _ = run
    snapshots = [
        p for p in checkpoint_dir.glob("*.pt")
        if p.name != "latest.pt" and not p.name.startswith("best_")
    ]
    assert snapshots, "a timestamped snapshot is written at run end"


def test_no_best_checkpoint_when_the_window_never_fills(run) -> None:
    """Two rounds of 2 drafts cannot fill a 100-draft window (FR-033, SC-009)."""
    _, out, _, checkpoint_dir, _ = run
    assert not list(checkpoint_dir.glob("best_*.pt"))
    assert "best anchor margin: n/a" in out or "none recorded" in out


def test_movement_line_carries_both_kls_and_the_lr(run) -> None:
    _, out, _, _, _ = run
    assert out.count("KL(prev||new)=") == 2
    assert out.count("KL(init||new)=") == 2
    assert out.count("lr=") >= 2


def test_startup_echo_reports_the_window_geometry(run) -> None:
    """The operator sees the window's length in rounds and its lag (§ 6.1)."""
    _, out, _, _, _ = run
    assert "anchor window" in out
    assert "rounds," in out and "lag)" in out


def test_corpus_grows_by_drafts_per_round_times_rounds(run) -> None:
    _, _, corpus, _, _ = run
    records = list(read_records(corpus))
    assert len(records) == 2 * DRAFTS_PER_ROUND


def test_round_two_trained_on_drafts_generated_after_round_one(run) -> None:
    """No batch is ever re-shown (SC-008): the two rounds' drafts are disjoint."""
    _, out, corpus, _, _ = run
    ids = [r.draft_id for r in read_records(corpus)]
    assert len(set(ids)) == len(ids), "no draft appears twice"

    round_one = set(ids[:DRAFTS_PER_ROUND])
    round_two = set(ids[DRAFTS_PER_ROUND:])
    assert round_one.isdisjoint(round_two)
    # Round 1's checkpoint write is logged before round 2's block starts.
    assert out.index("saved") < out.index("round 1 |")


def test_best_checkpoint_is_written_once_the_window_fills(
    tmp_path: Path, cards_path: Path, base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    """With anchor_window == drafts_per_round the window fills on round 0."""
    checkpoint_dir = tmp_path / "models"
    monkeypatch.setattr(online, "CHECKPOINT_DIR", checkpoint_dir)
    labeler = _VaryingLabeler()
    monkeypatch.setattr(
        "draft.application.generate_draft_data.build_labeler",
        lambda config, *, locator=None: labeler,
    )
    monkeypatch.setattr(
        "draft.application.generate_draft_data.GenerateDraftDataSupervisor."
        "_default_launch_worker",
        lambda self, external_labels: (
            lambda: _FakeProc([_transcript(f"b{i}") for i in range(8)])
        ),
    )

    corpus = tmp_path / "drafts.jsonl"
    corpus.write_text("", encoding="utf-8")
    TrainDraftAgentOnlineUseCase().execute(TrainDraftAgentOnlineConfig(
        learner_label=LEARNER,
        learner_checkpoint=base_checkpoint,
        frozen={ANCHOR: base_checkpoint},
        mix=[(LEARNER, 5), (ANCHOR, 3)],
        scorer_checkpoint=base_checkpoint,
        cards_path=cards_path,
        rollout_temperature=2.0,
        drafts_per_round=DRAFTS_PER_ROUND,
        anchor_window=DRAFTS_PER_ROUND,   # fills on the very first round
        snapshot_every=100,
        max_rounds=2,
        batch_size=8,
        output_path=corpus,
    ))
    out = capsys.readouterr().out

    bests = list(checkpoint_dir.glob("best_*.pt"))
    assert bests, "a best checkpoint is written once the window is full"
    ckpt = DraftAgentStore().load_checkpoint(bests[0])
    assert ckpt.rl_metadata["algorithm"] == "online-grpo"
    assert "best checkpoint   :" in out
    assert "best models" in out or "best " in out


def test_the_corpus_is_appended_never_truncated(
    tmp_path: Path, cards_path: Path, base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """drafts.jsonl is the canonical shared corpus — opening it "w" destroys it."""
    monkeypatch.setattr(online, "CHECKPOINT_DIR", tmp_path / "models")
    labeler = _VaryingLabeler()
    monkeypatch.setattr(
        "draft.application.generate_draft_data.build_labeler",
        lambda config, *, locator=None: labeler,
    )
    monkeypatch.setattr(
        "draft.application.generate_draft_data.GenerateDraftDataSupervisor."
        "_default_launch_worker",
        lambda self, external_labels: (
            lambda: _FakeProc([_transcript(f"n{i}") for i in range(4)])
        ),
    )

    corpus = tmp_path / "drafts.jsonl"
    pre_existing = json.dumps({
        "draft_id": "gen1-legacy", "run_id": "old", "timestamp": "t",
        "seats": [{"agent": "forge-full", "deck": [], "deck_score": None}],
        "boosters": [{"set_code": "TST", "picks": ["c0"]}],
    })
    corpus.write_text(pre_existing + "\n", encoding="utf-8")

    TrainDraftAgentOnlineUseCase().execute(TrainDraftAgentOnlineConfig(
        learner_label=LEARNER,
        learner_checkpoint=base_checkpoint,
        frozen={ANCHOR: base_checkpoint},
        mix=[(LEARNER, 5), (ANCHOR, 3)],
        scorer_checkpoint=base_checkpoint,
        cards_path=cards_path,
        rollout_temperature=2.0,
        drafts_per_round=DRAFTS_PER_ROUND,
        max_rounds=1,
        batch_size=8,
        output_path=corpus,
    ))

    ids = [r.draft_id for r in read_records(corpus)]
    assert ids[0] == "gen1-legacy", "the pre-existing corpus must survive"
    assert len(ids) == 1 + DRAFTS_PER_ROUND
