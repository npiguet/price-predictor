"""Startup validation of a gen-3 online run (data-model §1.1, FR-024): every rule
that runs before the Forge worker launches, the anchor it resolves, the
anchor window in rounds, the annealing floor, and the embedding-width check.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from draft.application.train_draft_agent_online import TrainDraftAgentOnlineConfig


# --------------------------------------------------------------------------- #
# Startup validation (data-model §1.1, FR-024)
# --------------------------------------------------------------------------- #

class OnlineConfigError(ValueError):
    """An invalid gen-3 run configuration, rejected before any update."""


def _refuse(problem: str | None) -> None:
    """Raise :class:`OnlineConfigError` with ``problem``, when there is one.

    Each rule check below returns the message of its first violation, or ``None``.
    """
    if problem is not None:
        raise OnlineConfigError(problem)


def _first_problem(problems: Iterable[str | None]) -> str | None:
    """The first violation of a lazily checked sequence, or ``None``."""
    return next((problem for problem in problems if problem is not None), None)


def resolve_anchor(config: TrainDraftAgentOnlineConfig) -> str:
    """Resolve (and validate) the anchor label — data-model §1.1 rule 4.

    The anchor is the frozen label the margin is measured against. With exactly
    one frozen agent it defaults to that agent; otherwise ``--anchor`` is
    required. It must be a frozen label present in the mix — never the learner
    (whose margin against itself is 0) and never a Forge built-in (which carries
    no checkpoint and so cannot be held fixed as a generation).
    """
    mix_labels = {label for label, _ in config.mix}
    _refuse(_anchor_binding_problem(config))
    anchor = next(iter(config.frozen)) if config.anchor is None else config.anchor
    _refuse(_anchor_in_mix_problem(anchor, mix_labels))
    return anchor


def _anchor_binding_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 4: without ``--anchor``, exactly one frozen agent; with it, a frozen label."""
    if config.anchor is None:
        return _default_anchor_problem(config.frozen)
    if config.anchor not in config.frozen:
        return (
            f"--anchor {config.anchor!r} must be a --frozen label (got "
            f"{', '.join(sorted(config.frozen)) or 'none'}); the learner "
            "and Forge built-ins cannot serve as the fixed baseline"
        )
    return None


def _default_anchor_problem(frozen: dict[str, Path]) -> str | None:
    """Rule 4 without ``--anchor``: the anchor defaults only to a single frozen agent."""
    if not frozen:
        return (
            "no anchor available: bind a reference generation with "
            "--frozen LABEL=PATH (its windowed mean deck_score is the "
            "baseline the anchor margin is measured against)"
        )
    if len(frozen) > 1:
        return (
            "--anchor is required when more than one --frozen agent is "
            f"bound (got {', '.join(sorted(frozen))}); it names "
            "which one the anchor margin is measured against"
        )
    return None


def _anchor_in_mix_problem(anchor: str, mix_labels: set[str]) -> str | None:
    """Rule 4: the anchor's seats are drafted, so the margin is defined."""
    if anchor not in mix_labels:
        return (
            f"--anchor {anchor!r} must also appear in --mix, or no anchor seats "
            "are ever drafted and the margin is never defined"
        )
    return None


def _check_picker_checkpoint(config: TrainDraftAgentOnlineConfig) -> None:
    """``--build-method picker`` needs ``--picker-checkpoint``, and the file must exist."""
    if config.build_method != "picker":
        return
    if config.picker_checkpoint is None:
        raise OnlineConfigError("--picker-checkpoint is required by --build-method picker")
    if not config.picker_checkpoint.exists():
        raise OnlineConfigError(
            f"picker checkpoint not found: {config.picker_checkpoint} "
            "(required by --build-method picker)"
        )


def validate_config(config: TrainDraftAgentOnlineConfig) -> str:
    """Run every data-model §1.1 rule in order; return the resolved anchor label.

    Raises :class:`OnlineConfigError` on the first violation. Every check here
    runs before the Forge worker launches and before any update (FR-024,
    SC-006); the checkpoint-width / geometry checks (rule 7) need the embedding
    cache and so run inside :meth:`TrainDraftAgentOnlineUseCase.execute`.
    """
    _refuse(_checkpoints_problem(config))  # rule 1
    _refuse(_learner_in_mix_problem(config))  # rule 2
    _refuse(_unknown_mix_label_problem(config))  # rule 3
    anchor = resolve_anchor(config)  # rule 4
    _refuse(_temperatures_problem(config))  # rule 5
    _refuse(_scorer_problem(config))  # rule 6
    _check_picker_checkpoint(config)  # rule 6
    _refuse(_counts_problem(config))  # rule 8
    _refuse(_run_control_problem(config))  # rule 9
    return anchor


def _checkpoints_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 1 — the learner binding resolves to an existing checkpoint, as does every frozen one."""
    if not config.learner_label or config.learner_checkpoint is None:
        return "--learner LABEL=PATH is required"
    if not config.learner_checkpoint.exists():
        return f"learner checkpoint not found: {config.learner_checkpoint}"
    return _first_problem(
        f"frozen checkpoint for {label!r} not found: {path}"
        for label, path in sorted(config.frozen.items())
        if not path.exists()
    )


def _learner_in_mix_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 2 — the learner label is in the mix and is not also frozen."""
    if config.learner_label not in {label for label, _ in config.mix}:
        return (
            f"learner label {config.learner_label!r} must appear in --mix, or "
            "no learner seats are ever drafted and there is nothing to train on"
        )
    if config.learner_label in config.frozen:
        return (
            f"label {config.learner_label!r} is bound both as the learner and "
            "as a --frozen agent; it cannot be trained and held fixed at once"
        )
    return None


def _unknown_mix_label_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 3 — every mix label is a built-in, a frozen agent, or the learner."""
    from draft.application.agent_registry import FORGE_BUILTINS

    mix_labels = {label for label, _ in config.mix}
    known = FORGE_BUILTINS | set(config.frozen) | {config.learner_label}
    return _first_problem(
        f"mix label {label!r} is neither a Forge built-in "
        f"({', '.join(sorted(FORGE_BUILTINS))}), a --frozen agent, nor the "
        "learner"
        for label in sorted(mix_labels - known)
    )


def _temperatures_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 5 — the temperature map is supplied, names the learner with a positive
    value, and names nothing that has no policy behind it.

    Both omissions are silent failures otherwise: a missing learner entry would
    train at some default temperature nobody chose, and an entry for a Forge
    built-in (whose randomisation is internal) or an unbound label would be
    quietly ignored while the operator believed the field was sampling.
    """
    temperatures = config.agent_temperatures
    if temperatures is None:
        return (
            "--agent-temp \"LABEL=T,…\" is required (no default): it is both the "
            "per-label sampling temperature and — through the learner's entry — "
            "the temperature every policy distribution in the update is "
            "evaluated at"
        )
    learner_problem = _learner_temperature_problem(config.learner_label, temperatures)
    if learner_problem is not None:
        return learner_problem
    return _first_problem(
        _temperature_entry_problem(label, temperatures[label], config)
        for label in sorted(temperatures)
    )


def _learner_temperature_problem(
    learner_label: str, temperatures: dict[str, float],
) -> str | None:
    """Rule 5, the learner's entry: present and positive."""
    if learner_label not in temperatures:
        return (
            f"--agent-temp must name the learner {learner_label!r}: "
            "sampling is its only exploration mechanism, and its temperature is "
            "also the one the update's policy distributions are evaluated at"
        )
    if temperatures[learner_label] <= 0:
        return (
            f"--agent-temp {learner_label}="
            f"{temperatures[learner_label]} must be > 0: the learner is "
            "the one agent that cannot play argmax, since sampling is its only "
            "exploration mechanism"
        )
    return None


def _temperature_entry_problem(
    label: str, temperature: float, config: TrainDraftAgentOnlineConfig,
) -> str | None:
    """Rule 5, one entry: a learner or frozen label, never a built-in, never negative."""
    from draft.application.agent_registry import FORGE_BUILTINS

    if label in FORGE_BUILTINS:
        return (
            f"--agent-temp cannot name the Forge built-in {label!r}: Forge "
            "randomises internally and ignores any temperature set here"
        )
    if label != config.learner_label and label not in config.frozen:
        return (
            f"--agent-temp label {label!r} is neither the learner nor a "
            "--frozen agent, so nothing would apply it"
        )
    if temperature < 0:
        return (
            f"--agent-temp {label}={temperature} must be >= 0 "
            "(omit the label, or give it 0, to have it play argmax)"
        )
    return None


def _scorer_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 6 — the scorer behind every reward exists."""
    if not config.scorer_checkpoint.exists():
        return (
            f"scorer checkpoint not found: {config.scorer_checkpoint} (the "
            "frozen scorer produces every seat's deck_score, i.e. the reward)"
        )
    return None


def _counts_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 8 — positive counts."""
    counts = (
        ("--drafts-per-round", config.drafts_per_round),
        ("--anchor-window", config.anchor_window),
        ("--batch-size", config.batch_size),
        ("--snapshot-every", config.snapshot_every),
    )
    count_problem = _first_problem(
        f"{name} must be >= 1, got {value}" for name, value in counts if value < 1
    )
    if count_problem is not None:
        return count_problem
    if config.max_rounds is not None and config.max_rounds < 1:
        return f"--max-rounds must be >= 1 when given, got {config.max_rounds}"
    return None


def _run_control_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 9 — run-control ordering.

    Both knobs key off "rounds since the last new best margin", and the margin
    needs a full window to reflect an improvement, so each must clear the
    window; annealing must clear before stopping does, or a decay could never
    pre-empt a stop.
    """
    rounds = window_rounds(config)
    if config.patience is not None and config.patience <= rounds:
        return (
            f"--patience ({config.patience}) must be greater than the anchor "
            f"window measured in rounds ({rounds} = ceil({config.anchor_window} "
            f"/ {config.drafts_per_round})); a shorter patience would stop the "
            "run before an improvement could reach the margin"
        )
    if config.lr_decay_patience is not None:
        return _lr_decay_problem(config, rounds)
    return None


def _lr_decay_problem(config: TrainDraftAgentOnlineConfig, rounds: int) -> str | None:
    """Rule 9, LR annealing: its patience clears the window and pre-empts the stop."""
    if config.lr_decay_patience <= rounds:
        return (
            f"--lr-decay-patience ({config.lr_decay_patience}) must be "
            f"greater than the anchor window measured in rounds ({rounds}); "
            "annealing on evidence the margin has not had time to show is "
            "just noise-chasing"
        )
    if config.patience is not None and config.lr_decay_patience >= config.patience:
        return (
            f"--lr-decay-patience ({config.lr_decay_patience}) must be < "
            f"--patience ({config.patience}) so a decay pre-empts the stop "
            "and the run only gives up at the LR floor"
        )
    return _lr_schedule_problem(config)


def _lr_schedule_problem(config: TrainDraftAgentOnlineConfig) -> str | None:
    """Rule 9, the annealing schedule: a factor in (0, 1) and a floor in (0, lr]."""
    if not 0.0 < config.lr_decay_factor < 1.0:
        return f"--lr-decay-factor must be in (0, 1), got {config.lr_decay_factor}"
    if config.min_lr is not None and not 0.0 < config.min_lr <= config.lr:
        return f"--min-lr must be > 0 and <= --lr ({config.lr}), got {config.min_lr}"
    return None


def window_rounds(config: TrainDraftAgentOnlineConfig) -> int:
    """How many rounds the anchor window spans — the bound on both patiences.

    Also the basis of the margin's ~half-window lag (data-model § 7.1), which is
    why it is echoed at startup.
    """
    return max(1, math.ceil(config.anchor_window / config.drafts_per_round))


def resolve_min_lr(lr: float, min_lr: float | None) -> float:
    """The annealing floor, defaulting to ``lr × 1e-3`` as the gen-1 trainer does."""
    return min_lr if min_lr is not None else lr * 1e-3


def probe_embedding_width(cards_path: Path) -> int | None:
    """Width of the ``.npz`` cache, read from the first cached card found.

    The trainers that read a corpus learn this from the loader after ingesting
    it; an online run has no corpus at startup, so the cache is probed directly
    to get the width *before* the Forge worker launches.
    """
    for path in sorted(cards_path.rglob("*.npz")):
        try:
            with np.load(path) as data:
                return int(data["embedding"].shape[0])
        except Exception:
            continue  # a truncated/foreign .npz — try the next one
    return None


def check_embedding_width(
    agent_config: Any, config: TrainDraftAgentOnlineConfig,
) -> int:
    """Fail fast when the checkpoint and the ``.npz`` cache disagree (rule 7).

    Without this the mismatch would surface as a torch shape error deep inside
    the first round's forward — after Forge has already started and drafted.
    """
    from draft.application.train_draft_agent import _check_dims

    embeddings = config.embeddings_path or config.cards_path
    width = probe_embedding_width(embeddings)
    if width is None:
        raise OnlineConfigError(
            f"no .npz card embeddings found under {embeddings}; run "
            "`python -m sealed encode-cards` with the encoder this agent was "
            "trained on"
        )
    _check_dims(agent_config, width, embeddings)
    return width
