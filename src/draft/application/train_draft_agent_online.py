"""Gen-3 online self-play GRPO fine-tuning of the two-headed draft agent (spec 021).

A streaming generate → update → discard → regenerate loop in one process. Each
round drives **one resident Forge draft worker** for ``drafts_per_round`` fresh
drafts whose learner seats are piloted by the live in-training policy, builds and
scores every seat's deck to get the reward, takes **one minibatch pass** of the
single critic-free term

    L = − mean( A · logπ_T(a|s) )        over learner-seat picks only

then discards the batch and drafts the next round with the updated weights. π_T is
the PACK-masked softmax at the rollout temperature ``T``; ``A`` is the round-
standardised pod-relative leave-one-out ``deck_score`` (terminal, γ=1), shared by
all of a seat's picks and detached.

There is deliberately no critic term, GAE, KL anchor, entropy bonus, validation
split, early stop, or best-checkpoint guard — see :class:`TrainDraftAgentOnlineConfig`.
The per-pick state walk is the shared :mod:`draft.application.draft_pick_states`;
the reward and batching helpers are :mod:`draft.application.draft_training_common`;
the masked-distribution math is
:mod:`price_predictor.infrastructure.torch_training`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from draft.application.online_advantages import assign_advantages
from draft.application.online_anchor_window import AnchorWindow
from draft.application.online_checkpoints import (
    CheckpointSource,
    CheckpointStamp,
    _write_checkpoint,
    base_generation,
)
from draft.application.online_diagnostics import SweepPolicies, SweepSettings, diagnostics_sweep
from draft.application.online_log import _log
from draft.application.online_round_loader import RoundBatch, RoundLoader
from draft.application.online_round_report import RoundDiagnostics, _fmt_margin, format_round_lines
from draft.application.online_run_report import (
    ResolvedStartup,
    RunOutcome,
    format_final_summary,
    format_startup_echo,
)
from draft.application.online_startup import (
    LoadedLearner,
    SeededRun,
    build_optimisation,
    build_pick_services,
    load_learner,
    seed_run,
)
from draft.application.online_update import Learner, UpdateSettings, run_update_pass
from draft.domain.draft_geometry import DraftRecord


@dataclass(kw_only=True)
class TrainDraftAgentOnlineConfig:
    """Resolved inputs to the online GRPO loop (data-model §1).

    ``learner_label``/``learner_checkpoint`` and ``agent_temperatures`` are
    required; the dataclass defaults exist only so the CLI can construct and then
    validate (:func:`~draft.application.online_config_validation.validate_config`,
    data-model §1.1). Architecture is inherited from the learner checkpoint —
    there are no architecture flags.

    **Absent by design** (spec FR-006, Out of Scope) — gen-2 knobs that do not
    exist here and must not be added: ``value_weight``, ``gae_lambda``,
    ``kl_coef``, ``entropy_coef`` and every coefficient-decay schedule,
    ``val_fraction``, ``patience``, ``epochs``, ``lr_decay_*``, ``resume``, and
    ``pick_mode`` (each label's mode follows from ``agent_temperatures``: sample
    where named, argmax where not). The operator is expected to tune exactly
    three knobs — ``lr``, ``agent_temperatures``, and ``drafts_per_round``;
    everything else carries a fixed default.
    """

    # Agent wiring (FR-002, FR-003)
    learner_label: str = ""
    learner_checkpoint: Path | None = None
    frozen: dict[str, Path] = field(default_factory=dict)
    anchor: str | None = None
    mix: list[tuple[str, int]] = field(default_factory=list)

    # Deck building & reward (FR-005)
    scorer_checkpoint: Path
    build_method: str = "greedy"
    #: Required (by the CLI) only with ``build_method == "picker"``.
    picker_checkpoint: Path | None = None
    cards_path: Path
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None

    # Rollout & optimisation (FR-004, FR-006, FR-025)
    agent_temperatures: dict[str, float] | None = None
    lr: float = 1e-4
    drafts_per_round: int = 10
    anchor_window: int = 100
    snapshot_every: int = 25
    # Run control (FR-034/FR-035) — all opt-in. These govern when a run stops or
    # anneals rather than how it learns, so the default surface stays the three
    # tunable learning knobs (FR-006). They key off the anchor margin, never a
    # held-out loss; none exists.
    patience: int | None = None
    lr_decay_patience: int | None = None
    lr_decay_factor: float = 0.1
    min_lr: float | None = None
    max_rounds: int | None = None
    set_code: str | None = None
    batch_size: int = 32
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    seed: int = 42

    # Corpus & fault handling (FR-020, FR-031)
    output_path: Path
    max_consecutive_faults: int = 5

    # Run directory (LAYOUT): checkpoints, and the worker's stderr log.
    checkpoint_dir: Path
    worker_log_dir: Path

    @property
    def rollout_temperature(self) -> float | None:
        """The learner's own temperature — derived, never separately settable.

        Sampling and every policy distribution in the update (logπ, entropy, KL)
        read this one value, so folding it into ``agent_temperatures`` removes
        the second flag that could disagree and silently desync the rollout from
        the loss. ``None`` only before validation, which requires the entry.
        """
        return (self.agent_temperatures or {}).get(self.learner_label)

    def temperature_for(self, label: str) -> float:
        """A label's sampling temperature; ``0.0`` (i.e. argmax) when unnamed."""
        return float((self.agent_temperatures or {}).get(label, 0.0))


# --------------------------------------------------------------------------- #
# The loop (FR-005, FR-029, FR-031)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _DraftField:
    """The supervisor that drafts every round, with the registry and labeler it was built from."""

    supervisor: Any
    registry: Any
    labeler: Any


@dataclass(frozen=True)
class _OnlineRun:
    """Everything startup builds that the rounds read."""

    config: TrainDraftAgentOnlineConfig
    seeded: SeededRun
    policies: SweepPolicies
    learner: Learner
    plateau: Any
    loader: RoundLoader
    checkpoints: CheckpointSource
    field: _DraftField
    window: AnchorWindow
    latest_path: Path
    records: Iterator[DraftRecord]
    update_settings: UpdateSettings
    sweep_settings: SweepSettings
    start_time: float


@dataclass
class _RunTally:
    """The loop's running counts and the checkpoints it has written so far."""

    rounds: int = 0
    total_drafts: int = 0
    total_picks: int = 0
    snapshot_path: Path | None = None
    best_path: Path | None = None


class TrainDraftAgentOnlineUseCase:
    """Owns the whole online loop in one process (FR-029).

    At startup it validates, loads the learner, builds one shared card locator,
    one deck labeler, one pick registry (learner served by the *live* model), and
    launches one resident Forge worker. Each round it pulls
    ``--drafts-per-round`` fresh records from the suspended record stream,
    appends them to the corpus, applies the one-pass GRPO update, writes
    ``latest.pt``, and drafts the next round with the updated weights.
    """

    def execute(self, config: TrainDraftAgentOnlineConfig) -> int:
        """Run until ``--max-rounds``, an interrupt, or the pick-fault abort.

        Returns the number of completed rounds.
        """
        run = self._start(config)
        tally = _RunTally()
        # Append-only, never "w": output/draft/drafts.jsonl is the canonical
        # shared corpus and truncating it would destroy the gen-1 / live-play
        # data (FR-020).
        with open(config.output_path, "a", buffering=1, encoding="utf-8") as out:
            try:
                self._run_rounds(run, tally, out)
            finally:
                run.records.close()
        self._close(run, tally)
        return tally.rounds

    # --- startup ---

    def _start(self, config: TrainDraftAgentOnlineConfig) -> _OnlineRun:
        """Validate, load the learner, build the field, echo the run, and launch the worker."""
        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

        seeded = seed_run(config)
        loaded = load_learner(config)
        # One memoizing locator for the labeler, every pick service, and the
        # trainer, so each card's .npz is decompressed once for the whole run.
        locator = ConvertedCardLocator(
            config.cards_path, embeddings_path=config.embeddings_path,
        )
        loader = RoundLoader(locator)
        generation = base_generation(loaded.base.rl_metadata)
        learner, plateau = build_optimisation(config, loaded.policies.model)
        field = self._build_field(config, loaded, locator, seeded.temperature)
        startup = ResolvedStartup(
            run_id=field.supervisor.run_id, anchor=seeded.anchor, generation=generation,
            device=loaded.device, embedding_dim=loaded.embedding_dim,
        )
        for line in format_startup_echo(config, startup=startup):
            _log(line)

        window = AnchorWindow(config.anchor_window, config.learner_label, seeded.anchor)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return _OnlineRun(
            config=config, seeded=seeded, policies=loaded.policies, learner=learner,
            plateau=plateau, loader=loader,
            checkpoints=CheckpointSource(
                store=loaded.store, model=learner.model, optimizer=learner.optimizer,
                config=config, base=loaded.base, generation=generation,
            ),
            field=field, window=window, latest_path=config.checkpoint_dir / "latest.pt",
            records=self._launch(field),
            update_settings=UpdateSettings(
                temperature=seeded.temperature, batch_size=config.batch_size,
                max_grad_norm=config.max_grad_norm, device=loaded.device,
            ),
            sweep_settings=SweepSettings(
                batch_size=config.batch_size, temperature=seeded.temperature,
                device=loaded.device,
            ),
            start_time=time.monotonic(),
        )

    def _build_field(
        self, config: TrainDraftAgentOnlineConfig, loaded: LoadedLearner, locator: Any,
        temperature: float,
    ) -> _DraftField:
        """The labeler, the pick registry over every preloaded service, and the supervisor."""
        from draft.application.agent_registry import AgentRegistry
        from draft.application.generate_draft_data import (
            GenerateDraftDataSupervisor,
            build_labeler,
        )

        gen_config = self._generation_config(config, temperature)
        labeler = build_labeler(gen_config, locator=locator)
        services = build_pick_services(config, loaded, locator, temperature)
        # Every service is preloaded, so `build`'s single pick_mode never applies;
        # it still validates the labels and geometry-checks all of them (D4).
        registry = AgentRegistry.build(
            {},
            {label for label, _ in config.mix},
            locator=locator,
            device=loaded.device,
            preloaded=services,
        )
        supervisor = GenerateDraftDataSupervisor(
            gen_config, labeler=labeler, registry=registry,
        )
        return _DraftField(supervisor=supervisor, registry=registry, labeler=labeler)

    @staticmethod
    def _launch(field: _DraftField) -> Iterator[DraftRecord]:
        """Install the signal handlers, launch the resident worker, open its record stream."""
        supervisor = field.supervisor
        supervisor._install_signal_handlers()
        launch = supervisor._default_launch_worker(field.registry.external_labels)
        return supervisor.iter_records(launch, field.labeler)

    @staticmethod
    def _generation_config(
        config: TrainDraftAgentOnlineConfig, temperature: float,
    ) -> Any:
        """The supervisor's config, every field set explicitly.

        Not relying on ``GenerateDraftDataConfig``'s defaults is deliberate: its
        ``build_method`` defaults to ``"picker"`` while gen-3's default is
        ``"greedy"``, so an omitted field would silently label every seat with a
        different builder than the operator asked for.
        """
        from draft.application.generate_draft_data import GenerateDraftDataConfig

        return GenerateDraftDataConfig(
            n_drafts=0,  # unused: the online loop consumes iter_records directly
            agent_mix=config.mix,
            set_code=config.set_code,
            scorer_checkpoint=config.scorer_checkpoint,
            build_method=config.build_method,
            picker_checkpoint=config.picker_checkpoint,
            cards_path=config.cards_path,
            embeddings_path=config.embeddings_path,
            output_path=config.output_path,
            resume=True,
            agent_checkpoints=dict(config.frozen),
            pick_mode="sample",
            temperature=temperature,
            seed=config.seed,
            max_consecutive_faults=config.max_consecutive_faults,
            # The link that actually arms the Java rule: without it, learner-free
            # pods are drafted and trained on as if valid (FR-003).
            required_agent=config.learner_label,
            worker_log_dir=config.worker_log_dir,
        )

    # --- the rounds ---

    def _run_rounds(self, run: _OnlineRun, tally: _RunTally, out: Any) -> None:
        """Play rounds until ``--max-rounds``, the stream's end, a stop, or a shutdown."""
        max_rounds = run.config.max_rounds
        while max_rounds is None or tally.rounds < max_rounds:
            if not self._play_round(run, tally, out):
                break

    def _play_round(self, run: _OnlineRun, tally: _RunTally, out: Any) -> bool:
        """Pull, train, report, save and apply run control; ``False`` when the run ends."""
        gen_start = time.monotonic()
        run.learner.model.eval()  # generation: the live policy drafts
        fresh = self._pull_round(run.records, run.config.drafts_per_round, out, run.window)
        if not fresh:
            return False  # the stream ended (shutdown requested)
        gen_seconds = time.monotonic() - gen_start
        tally.total_drafts += len(fresh)

        diag = self._train_round(run, tally, fresh, gen_seconds)
        is_best = self._report_round(run, diag, tally.rounds)
        tally.rounds += 1
        self._save_round(run, tally, is_best)
        if self._run_control_stops(run):
            return False
        return not run.field.supervisor._shutdown.is_set()

    @staticmethod
    def _pull_round(
        records: Iterator[DraftRecord], n: int, out: Any, window: AnchorWindow,
    ) -> list[DraftRecord]:
        """Pull ``n`` fresh records, persisting and windowing each as it arrives.

        Between rounds the generator stays suspended and the worker stays
        resident. Each record is appended the moment it arrives, so a round that
        later no-ops still leaves its drafts on disk.
        """
        from draft.infrastructure.draft_record_io import append_record

        fresh: list[DraftRecord] = []
        for record in records:
            append_record(out, record)
            window.add(record)
            fresh.append(record)
            if len(fresh) >= n:
                break
        return fresh

    def _train_round(
        self, run: _OnlineRun, tally: _RunTally, fresh: list[DraftRecord], gen_seconds: float,
    ) -> RoundDiagnostics:
        """Build the round's batch and advantages, and update unless the round is degenerate."""
        learner_label = run.config.learner_label
        train_start = time.monotonic()
        batch = run.loader.build(fresh, learner_label)
        batch.index = tally.rounds
        batch.gen_seconds = gen_seconds
        stats = assign_advantages(batch)

        diag = RoundDiagnostics(
            learner_label=learner_label,
            anchor_label=run.seeded.anchor,
            index=tally.rounds,
            drafts=len(fresh),
            total_drafts=tally.total_drafts,
            learner_seats=len(batch.learner_rewards),
            dropped_seats=batch.dropped_seats,
            learner_picks=len(batch.examples),
            gen_seconds=gen_seconds,
            reward=stats,
        )
        if stats.degenerate:
            diag.skipped = True
            diag.skip_reason = stats.reason
        else:
            tally.total_picks += len(batch.examples)
            self._update(run, batch, diag)
        diag.train_seconds = time.monotonic() - train_start
        diag.lr = run.plateau.current_lr()
        return diag

    @staticmethod
    def _update(run: _OnlineRun, batch: RoundBatch, diag: RoundDiagnostics) -> None:
        """The one-pass update from πₖ to πₖ₊₁, then the sweep that measures it."""
        policies = run.policies
        policies.prev_model.load_state_dict(policies.model.state_dict())
        update = run_update_pass(
            run.learner, batch, settings=run.update_settings, rng=run.seeded.rng,
        )
        # Back to eval before the sweep: with dropout active the
        # entropy/off-argmax/KL figures would measure noise.
        policies.model.eval()
        diag.policy_loss = update.policy_loss
        diag.grad_norm = update.grad_norm
        diag.sweep = diagnostics_sweep(
            policies, batch.examples, batch.table, settings=run.sweep_settings,
        )

    @staticmethod
    def _report_round(run: _OnlineRun, diag: RoundDiagnostics, round_index: int) -> bool:
        """Close the round in the anchor window and log it; ``True`` on a new best margin."""
        window = run.window
        is_best = window.observe_round(round_index)
        diag.anchor_margin = window.margin
        diag.label_means = window.label_means()
        diag.window_drafts = window.window_drafts
        for line in format_round_lines(diag):
            _log(line)
        return is_best

    def _save_round(self, run: _OnlineRun, tally: _RunTally, is_best: bool) -> None:
        """``latest.pt`` every round, a best on a new best margin, a snapshot every N rounds."""
        stamp = CheckpointStamp(rounds=tally.rounds, lr_decay_count=run.plateau.decay_count)
        _write_checkpoint(run.checkpoints, stamp, run.latest_path)
        _log(f"saved {run.latest_path} (round {tally.rounds - 1})")
        if is_best:
            tally.best_path = self._write_best(
                run.checkpoints, stamp,
                round_index=tally.rounds - 1, margin=run.window.best_margin,
            )
        if tally.rounds % run.config.snapshot_every == 0:
            tally.snapshot_path = self._snapshot(run.checkpoints, stamp)

    @staticmethod
    def _run_control_stops(run: _OnlineRun) -> bool:
        """Anneal on a stalled margin, or stop when patience runs out; ``True`` to stop.

        A stalled margin anneals before it stops: the decay resets the shared
        counter, so patience can only fire at the floor (FR-035). Both knobs are
        None unless the operator armed them.
        """
        from draft.application.draft_lr_schedule import _maybe_decay

        config, window, plateau = run.config, run.window, run.plateau
        new_lr = _maybe_decay(window.rounds_since_best, config.lr_decay_patience, plateau)
        if new_lr is not None:
            _log(
                f"LR decay #{plateau.decay_count} -> {new_lr:.2e} after "
                f"{config.lr_decay_patience} rounds without a new best "
                "margin"
            )
            window.note_lr_decay()
            return False
        if config.patience is not None and window.rounds_since_best >= config.patience:
            _log(f"Stopping: {_stop_reason(config, plateau)}")
            return True
        return False

    # --- closing ---

    def _close(self, run: _OnlineRun, tally: _RunTally) -> None:
        """The final snapshot and the closing summary, on every termination path."""
        tally.snapshot_path = self._snapshot(
            run.checkpoints,
            CheckpointStamp(rounds=tally.rounds, lr_decay_count=run.plateau.decay_count),
        )
        outcome = RunOutcome(
            rounds=tally.rounds, total_drafts=tally.total_drafts,
            learner_picks=tally.total_picks, latest_path=run.latest_path,
            snapshot_path=tally.snapshot_path, best_path=tally.best_path,
        )
        for line in format_final_summary(
            outcome, elapsed=time.monotonic() - run.start_time, window=run.window,
        ):
            _log(line)

    @staticmethod
    def _snapshot(source: CheckpointSource, stamp: CheckpointStamp) -> Path:
        path = source.config.checkpoint_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        _write_checkpoint(source, stamp, path)
        _log(f"snapshot {path} (round {stamp.rounds})")
        return path

    @staticmethod
    def _write_best(
        source: CheckpointSource, stamp: CheckpointStamp, *, round_index: int,
        margin: float | None,
    ) -> Path:
        """Write ``best_{timestamp}.pt`` for a new best anchor margin (FR-033).

        A fresh file per best rather than one rolling ``best.pt``: the margin
        lags the policy by about half the window, so the genuinely best weights
        may be a few rounds earlier and the history is worth keeping.
        """
        path = source.config.checkpoint_dir / f"best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        _write_checkpoint(source, stamp, path)
        _log(f"best {path} (round {round_index}, margin {_fmt_margin(margin)})")
        return path


def _stop_reason(config: TrainDraftAgentOnlineConfig, plateau: Any) -> str:
    """Why patience stopped the run, naming the LR floor when annealing was armed."""
    reason = f"no new best anchor margin for {config.patience} rounds"
    if config.lr_decay_patience is not None:
        reason += f" at the LR floor ({plateau.current_lr():.2e})"
    return reason
