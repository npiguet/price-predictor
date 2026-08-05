"""Gen-3 startup validation + CLI wiring (data-model §1.1, FR-024, SC-006).

Every invalid configuration must be rejected **before** the Forge worker launches
and before any update, with the contract's exit code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from draft.infrastructure.cli import build_parser


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    path = tmp_path / "gen1.pt"
    path.write_bytes(b"stub")
    return path


@pytest.fixture
def scorer(tmp_path: Path) -> Path:
    path = tmp_path / "scorer.pt"
    path.write_bytes(b"stub")
    return path


def _run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def _argv(ckpt: Path, scorer: Path, **overrides: str | None) -> list[str]:
    base: dict[str, str | None] = {
        "--learner": f"gen-3={ckpt}",
        "--frozen": f"gen-1={ckpt}",
        "--mix": "gen-3:5,gen-1:3,forge-r30:1",
        "--scorer-checkpoint": str(scorer),
        "-T": "2.0",
        "--max-rounds": "1",
    }
    base.update(overrides)
    argv = ["train-draft-agent-online"]
    for flag, value in base.items():
        if value is None:
            continue
        argv += [flag, value]
    return argv


class _Sentinel(AssertionError):
    """Raised if the use case is reached — validation should have exited first."""


@pytest.fixture(autouse=True)
def _never_run(monkeypatch: pytest.MonkeyPatch):
    """Any test in this module that reaches ``execute`` has failed to fail fast."""
    import draft.application.train_draft_agent_online as online

    def boom(*_args, **_kwargs):
        raise _Sentinel("the use case must not run on an invalid configuration")

    monkeypatch.setattr(
        online.TrainDraftAgentOnlineUseCase, "execute", boom, raising=True,
    )


# --------------------------------------------------------------------------- #
# Learner binding (rules 1 & 2)
# --------------------------------------------------------------------------- #

def test_missing_learner_is_rejected(scorer: Path, capsys) -> None:
    argv = ["train-draft-agent-online", "--scorer-checkpoint", str(scorer), "-T", "2.0"]
    assert _run(argv) == 2
    assert "learner" in capsys.readouterr().err.lower()


def test_bare_path_learner_is_rejected(ckpt: Path, scorer: Path, capsys) -> None:
    """The label is load-bearing, so ``--learner PATH`` is not accepted."""
    assert _run(_argv(ckpt, scorer, **{"--learner": str(ckpt)})) == 2
    assert "label" in capsys.readouterr().err.lower()


def test_more_than_one_learner_is_rejected(ckpt: Path, scorer: Path) -> None:
    argv = _argv(ckpt, scorer)
    argv += ["--learner", f"gen-4={ckpt}"]
    assert _run(argv) == 2


def test_missing_learner_checkpoint_file_is_rejected(
    tmp_path: Path, scorer: Path, capsys,
) -> None:
    absent = tmp_path / "nope.pt"
    assert _run(_argv(absent, scorer, **{"--frozen": None})) == 2
    assert "not found" in capsys.readouterr().err.lower()


def test_learner_label_absent_from_the_mix_is_rejected(
    ckpt: Path, scorer: Path, capsys,
) -> None:
    argv = _argv(ckpt, scorer, **{"--mix": "gen-1:3,forge-r30:1"})
    assert _run(argv) == 2
    assert "mix" in capsys.readouterr().err.lower()


def test_learner_label_also_bound_as_frozen_is_rejected(
    ckpt: Path, scorer: Path, capsys,
) -> None:
    argv = _argv(ckpt, scorer, **{"--frozen": f"gen-3={ckpt}"})
    assert _run(argv) == 2
    err = capsys.readouterr().err.lower()
    assert "frozen" in err and "gen-3" in err


# --------------------------------------------------------------------------- #
# Mix labels + anchor (rules 3 & 4)
# --------------------------------------------------------------------------- #

def test_unknown_mix_label_is_rejected(ckpt: Path, scorer: Path, capsys) -> None:
    argv = _argv(ckpt, scorer, **{"--mix": "gen-3:5,gen-1:3,mystery-bot:1"})
    assert _run(argv) == 2
    assert "mystery-bot" in capsys.readouterr().err


def test_anchor_defaults_to_the_sole_frozen_label(ckpt: Path, scorer: Path) -> None:
    """One frozen agent ⇒ ``--anchor`` may be omitted; validation passes to execute."""
    with pytest.raises(_Sentinel):
        _run(_argv(ckpt, scorer))


def test_ambiguous_anchor_with_two_frozen_labels_is_rejected(
    ckpt: Path, scorer: Path, capsys,
) -> None:
    argv = _argv(ckpt, scorer, **{"--mix": "gen-3:5,gen-1:3,gen-2:2"})
    argv += ["--frozen", f"gen-2={ckpt}"]
    assert _run(argv) == 2
    assert "anchor" in capsys.readouterr().err.lower()


def test_explicit_anchor_disambiguates_two_frozen_labels(
    ckpt: Path, scorer: Path,
) -> None:
    argv = _argv(ckpt, scorer, **{"--mix": "gen-3:5,gen-1:3,gen-2:2"})
    argv += ["--frozen", f"gen-2={ckpt}", "--anchor", "gen-1"]
    with pytest.raises(_Sentinel):
        _run(argv)


def test_anchor_naming_a_non_frozen_label_is_rejected(
    ckpt: Path, scorer: Path, capsys,
) -> None:
    argv = _argv(ckpt, scorer) + ["--anchor", "forge-r30"]
    assert _run(argv) == 2
    assert "anchor" in capsys.readouterr().err.lower()


def test_anchor_naming_the_learner_is_rejected(ckpt: Path, scorer: Path) -> None:
    assert _run(_argv(ckpt, scorer) + ["--anchor", "gen-3"]) == 2


def test_no_frozen_agent_at_all_is_rejected(ckpt: Path, scorer: Path, capsys) -> None:
    """Without a frozen anchor there is no margin baseline (FR-021)."""
    argv = _argv(ckpt, scorer, **{"--frozen": None, "--mix": "gen-3:5,forge-r30:1"})
    assert _run(argv) == 2
    assert "anchor" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# Temperature (rule 5)
# --------------------------------------------------------------------------- #

def test_missing_temperature_is_rejected(ckpt: Path, scorer: Path, capsys) -> None:
    assert _run(_argv(ckpt, scorer, **{"-T": None})) == 2
    assert "temperature" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("value", ["0", "-1.5"])
def test_non_positive_temperature_is_rejected(
    ckpt: Path, scorer: Path, value: str,
) -> None:
    assert _run(_argv(ckpt, scorer, **{"-T": value})) == 2


# --------------------------------------------------------------------------- #
# Reward inputs (rule 6)
# --------------------------------------------------------------------------- #

def test_missing_scorer_checkpoint_is_rejected(
    ckpt: Path, tmp_path: Path, capsys,
) -> None:
    argv = _argv(ckpt, tmp_path / "absent-scorer.pt")
    assert _run(argv) == 2
    assert "scorer" in capsys.readouterr().err.lower()


def test_missing_picker_is_rejected_only_with_build_method_picker(
    ckpt: Path, scorer: Path, tmp_path: Path, capsys,
) -> None:
    absent = tmp_path / "absent-picker.pt"
    argv = _argv(ckpt, scorer, **{
        "--build-method": "picker", "--picker-checkpoint": str(absent),
    })
    assert _run(argv) == 2
    assert "picker" in capsys.readouterr().err.lower()

    # …and is irrelevant under the default greedy build method.
    argv = _argv(ckpt, scorer, **{"--picker-checkpoint": str(absent)})
    with pytest.raises(_Sentinel):
        _run(argv)


# --------------------------------------------------------------------------- #
# Numeric bounds (rule 8)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "flag",
    ["--drafts-per-round", "--anchor-window", "--batch-size", "--snapshot-every",
     "--max-rounds"],
)
def test_non_positive_counts_are_rejected(
    ckpt: Path, scorer: Path, flag: str,
) -> None:
    assert _run(_argv(ckpt, scorer, **{flag: "0"})) == 2


# --------------------------------------------------------------------------- #
# Architecture mismatch (rule 7) → exit 6
# --------------------------------------------------------------------------- #

def test_architecture_error_exits_six(
    ckpt: Path, scorer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import draft.application.train_draft_agent_online as online
    from draft.domain.draft_agent_model import DraftAgentArchitectureError

    def boom(*_args, **_kwargs):
        raise DraftAgentArchitectureError("d_model must be divisible by n_heads")

    monkeypatch.setattr(
        online.TrainDraftAgentOnlineUseCase, "execute", boom, raising=True,
    )
    assert _run(_argv(ckpt, scorer)) == 6


def test_width_mismatch_exits_two(
    ckpt: Path, scorer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_check_dims`` raises ValueError for a cache/checkpoint width mismatch."""
    import draft.application.train_draft_agent_online as online

    def boom(*_args, **_kwargs):
        raise ValueError("Checkpoint expects 528-wide card embeddings")

    monkeypatch.setattr(
        online.TrainDraftAgentOnlineUseCase, "execute", boom, raising=True,
    )
    assert _run(_argv(ckpt, scorer)) == 2


# --------------------------------------------------------------------------- #
# Flags the gen-3 command deliberately does not offer (FR-006)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "flag",
    ["--resume", "--value-weight", "--gae-lambda", "--kl-coef", "--entropy-coef",
     "--val-fraction", "--epochs", "--pick-mode"],
)
def test_gen2_knobs_are_not_registered(flag: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train-draft-agent-online", flag, "1"])


# --------------------------------------------------------------------------- #
# Run control: patience + LR annealing (data-model § 1.1 rule 9, FR-034/FR-035)
# --------------------------------------------------------------------------- #
# At the defaults --anchor-window 100 / --drafts-per-round 10, window_rounds = 10,
# so every armed knob must exceed 10 and lr-decay-patience must be below patience.

def test_run_control_is_disabled_by_default() -> None:
    args = build_parser().parse_args([
        "train-draft-agent-online", "--learner", "gen-3=x.pt", "-T", "2.0",
    ])
    assert args.patience is None
    assert args.lr_decay_patience is None
    assert args.lr_decay_factor == 0.1
    assert args.min_lr is None


def test_absent_run_control_flags_leave_startup_unchanged(
    ckpt: Path, scorer: Path,
) -> None:
    """The opt-in default surface must reach execute exactly as before."""
    with pytest.raises(_Sentinel):
        _run(_argv(ckpt, scorer))


def test_patience_above_the_window_in_rounds_is_accepted(
    ckpt: Path, scorer: Path,
) -> None:
    with pytest.raises(_Sentinel):
        _run(_argv(ckpt, scorer, **{"--patience": "30"}))


@pytest.mark.parametrize("value", ["10", "5", "1"])
def test_patience_at_or_below_the_window_in_rounds_is_rejected(
    ckpt: Path, scorer: Path, value: str, capsys,
) -> None:
    assert _run(_argv(ckpt, scorer, **{"--patience": value})) == 2
    err = capsys.readouterr().err.lower()
    assert "patience" in err and "window" in err


def test_the_window_bound_follows_drafts_per_round(
    ckpt: Path, scorer: Path,
) -> None:
    """Bigger rounds shrink the window in rounds, so a smaller patience is legal."""
    argv = _argv(ckpt, scorer, **{
        "--drafts-per-round": "50", "--anchor-window": "200", "--patience": "6",
    })
    with pytest.raises(_Sentinel):   # window_rounds = 4, so 6 clears it
        _run(argv)

    argv = _argv(ckpt, scorer, **{
        "--drafts-per-round": "50", "--anchor-window": "200", "--patience": "3",
    })
    assert _run(argv) == 2


def test_lr_decay_patience_must_also_exceed_the_window(
    ckpt: Path, scorer: Path, capsys,
) -> None:
    argv = _argv(ckpt, scorer, **{"--lr-decay-patience": "8"})
    assert _run(argv) == 2
    assert "window" in capsys.readouterr().err.lower()


def test_lr_decay_patience_must_be_below_patience(
    ckpt: Path, scorer: Path, capsys,
) -> None:
    argv = _argv(ckpt, scorer, **{"--patience": "20", "--lr-decay-patience": "20"})
    assert _run(argv) == 2
    err = capsys.readouterr().err.lower()
    assert "lr-decay-patience" in err and "patience" in err


def test_annealing_is_legal_with_stopping_disabled(
    ckpt: Path, scorer: Path,
) -> None:
    """Anneal to the floor and keep running — a supported combination (FR-035)."""
    with pytest.raises(_Sentinel):
        _run(_argv(ckpt, scorer, **{"--lr-decay-patience": "15"}))


def test_both_armed_in_the_right_order_is_accepted(
    ckpt: Path, scorer: Path,
) -> None:
    argv = _argv(ckpt, scorer, **{"--patience": "30", "--lr-decay-patience": "15"})
    with pytest.raises(_Sentinel):
        _run(argv)


@pytest.mark.parametrize("value", ["0", "1", "1.5", "-0.1"])
def test_lr_decay_factor_must_be_a_proper_fraction(
    ckpt: Path, scorer: Path, value: str,
) -> None:
    argv = _argv(ckpt, scorer, **{
        "--lr-decay-patience": "15", "--lr-decay-factor": value,
    })
    assert _run(argv) == 2


@pytest.mark.parametrize("value", ["0", "-1e-6", "1e-3"])
def test_min_lr_must_be_positive_and_not_above_lr(
    ckpt: Path, scorer: Path, value: str,
) -> None:
    argv = _argv(ckpt, scorer, **{
        "--lr": "1e-4", "--lr-decay-patience": "15", "--min-lr": value,
    })
    assert _run(argv) == 2


def test_the_three_tunable_knobs_are_registered() -> None:
    args = build_parser().parse_args([
        "train-draft-agent-online",
        "--learner", "gen-3=x.pt", "-T", "1.5", "--lr", "5e-5",
        "--drafts-per-round", "20",
    ])
    assert args.rollout_temperature == 1.5
    assert args.lr == 5e-5
    assert args.drafts_per_round == 20


def test_defaults_match_the_contract() -> None:
    args = build_parser().parse_args([
        "train-draft-agent-online", "--learner", "gen-3=x.pt", "-T", "2.0",
    ])
    assert args.lr == 1e-4
    assert args.drafts_per_round == 10
    assert args.anchor_window == 100
    assert args.snapshot_every == 25
    assert args.batch_size == 32
    assert args.max_grad_norm == 1.0
    assert args.warmup_steps == 200
    assert args.seed == 42
    assert args.build_method == "greedy"
    assert args.max_rounds is None
    assert args.set_code is None
    assert args.output_path == "output/draft/drafts.jsonl"
    assert args.max_consecutive_faults == 5
