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

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from draft.application.online_advantages import assign_advantages
from draft.application.online_anchor_window import AnchorWindow
from draft.application.online_checkpoints import _write_checkpoint, base_generation
from draft.application.online_config_validation import (
    check_embedding_width,
    resolve_min_lr,
    validate_config,
)
from draft.application.online_diagnostics import diagnostics_sweep
from draft.application.online_log import _log
from draft.application.online_round_loader import RoundLoader
from draft.application.online_round_report import RoundDiagnostics, _fmt_margin, format_round_lines
from draft.application.online_run_report import format_final_summary, format_startup_echo
from draft.application.online_update import run_update_pass
from draft.domain.draft_agent_model import DraftAgentModel
from draft.domain.draft_geometry import DraftRecord
from draft.infrastructure.draft_agent_store import DraftAgentStore

_DEFAULT_MIX = "gen-3:5,gen-1:3,forge-r30:1,forge-r100:1"


@dataclass(kw_only=True)
class TrainDraftAgentOnlineConfig:
    """Resolved inputs to the online GRPO loop (data-model §1).

    ``learner_label``/``learner_checkpoint`` and ``agent_temperatures`` are
    required; the dataclass defaults exist only so the CLI can construct and then
    validate (:func:`_validate_config`, data-model §1.1). Architecture is
    inherited from the learner checkpoint — there are no architecture flags.

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
        from draft.application.agent_pick_service import AgentPickService
        from draft.application.agent_registry import AgentRegistry
        from draft.application.generate_draft_data import (
            GenerateDraftDataSupervisor,
            build_labeler,
        )
        from draft.application.train_draft_agent import (
            _make_scheduler,
            _maybe_decay,
            _PlateauLR,
            _select_device,
        )
        from draft.infrastructure.draft_record_io import append_record
        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

        anchor = validate_config(config)
        temperature = float(config.rollout_temperature)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        rng = random.Random(config.seed)

        # --- learner: the model the loop optimises AND the model that drafts ---
        store = DraftAgentStore()
        base = store.load_checkpoint(config.learner_checkpoint)
        embedding_dim = check_embedding_width(base.config, config)
        device = _select_device()
        model = DraftAgentModel(base.config)
        model.load_state_dict(base.model_state_dict)
        model.to(device)
        model.eval()
        # πₖ for the movement diagnostics: resident, never in train mode.
        prev_model = DraftAgentModel(base.config)
        prev_model.to(device)
        prev_model.eval()
        for parameter in prev_model.parameters():
            parameter.requires_grad_(False)
        # π₀ for the cumulative-KL diagnostic: the learner's own warm start,
        # frozen for the whole run. Not the anchor — that is a different
        # checkpoint whenever --learner and --frozen differ (research D17).
        init_model = DraftAgentModel(base.config)
        init_model.load_state_dict(base.model_state_dict)
        init_model.to(device)
        init_model.eval()
        for parameter in init_model.parameters():
            parameter.requires_grad_(False)

        # One memoizing locator for the labeler, every pick service, and the
        # trainer, so each card's .npz is decompressed once for the whole run.
        locator = ConvertedCardLocator(
            config.cards_path, embeddings_path=config.embeddings_path,
        )
        loader = RoundLoader(locator)

        generation = base_generation(base.rl_metadata)

        optimizer = torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": config.lr, "name": "agent"}],
        )
        # One warmup at the start of the run, then constant — never rebuilt per
        # round, so the LR schedule and optimizer moments are continuous across
        # rounds (FR-025). An online run has no total step count, so the ramp is
        # expressed in optimizer steps rather than a fraction of a horizon (D15).
        # The plateau controller folds its multiplier into the same warmup
        # lambda, so a decay scales the post-warmup constant without re-running
        # the ramp (FR-035). Constructed even when annealing is disabled — with
        # no decay ever taken its multiplier stays 1.0.
        plateau = _PlateauLR(
            base_lr=config.lr,
            factor=config.lr_decay_factor,
            min_lr=resolve_min_lr(config.lr, config.min_lr),
        )
        scheduler = _make_scheduler(
            optimizer, total_steps=config.warmup_steps, warmup_frac=1.0,
            controller=plateau,
        )

        gen_config = self._generation_config(config, temperature)
        labeler = build_labeler(gen_config, locator=locator)
        # Per-label pick modes (research D5), read straight off --agent-temp: a
        # label it names samples at that T, a label it omits plays argmax, its
        # best. Argmax is the default because a draft allocates one fixed pool of
        # cards, so a card a frozen agent wrongly declines does not vanish — it
        # flows downstream to its podmates. A sampled field hands the learner
        # cards a properly-playing agent would have kept, making the training
        # field weak in exactly the dimension drafting is about, and unlike the
        # evaluation field, where every agent plays argmax. Naming a frozen label
        # anyway buys pod diversity at that price; the operator decides.
        services = {}
        for index, (label, path) in enumerate(sorted(config.frozen.items())):
            frozen_temperature = config.temperature_for(label)
            services[label] = AgentPickService(
                path, locator, device=device,
                pick_mode="sample" if frozen_temperature > 0 else "argmax",
                temperature=frozen_temperature or 1.0,
                # A distinct stream per sampled agent: sharing the learner's seed
                # would couple the field's exploration to the learner's.
                seed=config.seed + 1 + index if frozen_temperature > 0 else None,
            )
        services[config.learner_label] = AgentPickService.from_model(
            model, base.config, locator, device=device,
            pick_mode="sample", temperature=temperature, seed=config.seed,
        )
        # Every service is preloaded, so `build`'s single pick_mode never applies;
        # it still validates the labels and geometry-checks all of them (D4).
        registry = AgentRegistry.build(
            {},
            {label for label, _ in config.mix},
            locator=locator,
            device=device,
            preloaded=services,
        )
        supervisor = GenerateDraftDataSupervisor(
            gen_config, labeler=labeler, registry=registry,
        )

        for line in format_startup_echo(
            config, run_id=supervisor.run_id, anchor=anchor, generation=generation,
            device=device, embedding_dim=embedding_dim,
        ):
            _log(line)

        window = AnchorWindow(
            config.anchor_window, config.learner_label, anchor,
        )
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = config.checkpoint_dir / "latest.pt"

        supervisor._install_signal_handlers()
        launch = supervisor._default_launch_worker(registry.external_labels)
        records = supervisor.iter_records(launch, labeler)

        rounds = 0
        total_drafts = 0
        total_picks = 0
        snapshot_path: Path | None = None
        best_path: Path | None = None
        stop_reason: str | None = None
        start_time = time.monotonic()
        # Append-only, never "w": output/draft/drafts.jsonl is the canonical
        # shared corpus and truncating it would destroy the gen-1 / live-play
        # data (FR-020).
        with open(config.output_path, "a", buffering=1, encoding="utf-8") as out:
            try:
                while config.max_rounds is None or rounds < config.max_rounds:
                    gen_start = time.monotonic()
                    model.eval()  # generation: the live policy drafts
                    fresh = self._pull_round(
                        records, config.drafts_per_round, out, window, append_record,
                    )
                    if not fresh:
                        break  # the stream ended (shutdown requested)
                    gen_seconds = time.monotonic() - gen_start
                    total_drafts += len(fresh)

                    train_start = time.monotonic()
                    batch = loader.build(fresh, config.learner_label)
                    batch.index = rounds
                    batch.gen_seconds = gen_seconds
                    stats = assign_advantages(batch)

                    diag = RoundDiagnostics(
                        learner_label=config.learner_label,
                        anchor_label=anchor,
                        index=rounds,
                        drafts=len(fresh),
                        total_drafts=total_drafts,
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
                        total_picks += len(batch.examples)
                        prev_model.load_state_dict(model.state_dict())
                        update = run_update_pass(
                            model, optimizer, scheduler, batch,
                            temperature=temperature,
                            batch_size=config.batch_size,
                            max_grad_norm=config.max_grad_norm,
                            device=device,
                            rng=rng,
                        )
                        # Back to eval before the sweep: with dropout active the
                        # entropy/off-argmax/KL figures would measure noise.
                        model.eval()
                        diag.policy_loss = update.policy_loss
                        diag.grad_norm = update.grad_norm
                        diag.sweep = diagnostics_sweep(
                            prev_model, model, batch.examples, batch.table,
                            batch_size=config.batch_size,
                            temperature=temperature,
                            device=device,
                            init_model=init_model,
                        )
                    diag.train_seconds = time.monotonic() - train_start
                    diag.lr = plateau.current_lr()

                    is_best = window.observe_round(rounds)
                    diag.anchor_margin = window.margin
                    diag.label_means = window.label_means()
                    diag.window_drafts = window.window_drafts
                    for line in format_round_lines(diag):
                        _log(line)

                    rounds += 1
                    _write_checkpoint(
                        store, model, optimizer, config, base, generation,
                        rounds, latest_path, plateau.decay_count,
                    )
                    _log(f"saved {latest_path} (round {rounds - 1})")
                    if is_best:
                        best_path = self._write_best(
                            store, model, optimizer, config, base, generation,
                            rounds, plateau.decay_count,
                            round_index=rounds - 1, margin=window.best_margin,
                        )
                    if rounds % config.snapshot_every == 0:
                        snapshot_path = self._snapshot(
                            store, model, optimizer, config, base, generation,
                            rounds, plateau.decay_count,
                        )

                    # A stalled margin anneals before it stops: the decay resets
                    # the shared counter, so patience can only fire at the floor
                    # (FR-035). Both knobs are None unless the operator armed them.
                    new_lr = _maybe_decay(
                        window.rounds_since_best, config.lr_decay_patience, plateau,
                    )
                    if new_lr is not None:
                        _log(
                            f"LR decay #{plateau.decay_count} -> {new_lr:.2e} after "
                            f"{config.lr_decay_patience} rounds without a new best "
                            "margin"
                        )
                        window.note_lr_decay()
                    elif (
                        config.patience is not None
                        and window.rounds_since_best >= config.patience
                    ):
                        stop_reason = (
                            f"no new best anchor margin for {config.patience} rounds"
                            + (
                                f" at the LR floor ({plateau.current_lr():.2e})"
                                if config.lr_decay_patience is not None else ""
                            )
                        )
                        _log(f"Stopping: {stop_reason}")
                        break

                    if supervisor._shutdown.is_set():
                        break
            finally:
                records.close()

        snapshot_path = self._snapshot(
            store, model, optimizer, config, base, generation, rounds,
            plateau.decay_count,
        )
        for line in format_final_summary(
            rounds=rounds, total_drafts=total_drafts, learner_picks=total_picks,
            elapsed=time.monotonic() - start_time, latest_path=latest_path,
            snapshot_path=snapshot_path, best_path=best_path, window=window,
        ):
            _log(line)
        return rounds

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

    @staticmethod
    def _pull_round(
        records: Iterator[DraftRecord],
        n: int,
        out: Any,
        window: AnchorWindow,
        append_record,
    ) -> list[DraftRecord]:
        """Pull ``n`` fresh records, persisting and windowing each as it arrives.

        Between rounds the generator stays suspended and the worker stays
        resident. Each record is appended the moment it arrives, so a round that
        later no-ops still leaves its drafts on disk.
        """
        fresh: list[DraftRecord] = []
        for record in records:
            append_record(out, record)
            window.add(record)
            fresh.append(record)
            if len(fresh) >= n:
                break
        return fresh

    @staticmethod
    def _snapshot(
        store: DraftAgentStore,
        model: DraftAgentModel,
        optimizer: torch.optim.Optimizer,
        config: TrainDraftAgentOnlineConfig,
        base: Any,
        generation: int,
        rounds: int,
        lr_decay_count: int = 0,
    ) -> Path:
        path = config.checkpoint_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        _write_checkpoint(
            store, model, optimizer, config, base, generation, rounds, path,
            lr_decay_count,
        )
        _log(f"snapshot {path} (round {rounds})")
        return path

    @staticmethod
    def _write_best(
        store: DraftAgentStore,
        model: DraftAgentModel,
        optimizer: torch.optim.Optimizer,
        config: TrainDraftAgentOnlineConfig,
        base: Any,
        generation: int,
        rounds: int,
        lr_decay_count: int,
        *,
        round_index: int,
        margin: float | None,
    ) -> Path:
        """Write ``best_{timestamp}.pt`` for a new best anchor margin (FR-033).

        A fresh file per best rather than one rolling ``best.pt``: the margin
        lags the policy by about half the window, so the genuinely best weights
        may be a few rounds earlier and the history is worth keeping.
        """
        path = config.checkpoint_dir / f"best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        _write_checkpoint(
            store, model, optimizer, config, base, generation, rounds, path,
            lr_decay_count,
        )
        _log(f"best {path} (round {round_index}, margin {_fmt_margin(margin)})")
        return path
