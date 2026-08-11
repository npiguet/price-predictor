"""CLI interface for the draft module (generate-draft-data, train-draft-agent).

Mirrors ``sealed/infrastructure/cli.py``: ``build_parser`` adds one subparser
per command with ``set_defaults(func=…)`` dispatch, and ``main`` parses and
calls the resolved ``func``. Application imports are lazy (inside each ``run_*``)
so ``python -m draft --help`` and unit tests don't pay the torch import cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Architecture flags forbidden alongside --resume / --checkpoint (SC-006); they
# register with default=None so "was it supplied?" is detectable.
_TRAIN_AGENT_ARCHITECTURE_FLAGS: tuple[str, ...] = (
    "d_model", "n_layers", "n_heads", "ff_dim", "dropout",
)

# Non-architecture training flags resolved with resume precedence (explicit CLI
# > resumed train_config > dataclass default). imitation_agents is resolved
# separately (CLI string -> tuple); architecture flags are inherited from the
# checkpoint's stored config, not train_config.
_RESUMABLE_TRAIN_FLAGS: tuple[str, ...] = (
    "drafts_path", "cards_path", "imitation_weight", "critic_weight",
    "lr", "warmup_frac", "batch_size", "max_grad_norm", "epochs",
    "val_fraction", "evals_per_epoch", "patience",
    "lr_decay_patience", "lr_decay_factor", "min_lr",
)
_TRAIN_PATH_FIELDS = frozenset({"drafts_path", "cards_path"})


def _resolve_train_agent_config(
    args: argparse.Namespace, resumed_train_config: dict | None,
):
    """Build TrainDraftAgentConfig with resume precedence (CLI > resumed > default)."""
    from draft.application.train_draft_agent import TrainDraftAgentConfig
    from sealed.infrastructure.cli_resume import resolve_resumable_args

    resolved = resolve_resumable_args(
        args,
        flag_names=_RESUMABLE_TRAIN_FLAGS,
        config_cls=TrainDraftAgentConfig,
        path_fields=_TRAIN_PATH_FIELDS,
        resumed_config=resumed_train_config,
    )

    # imitation_agents: CLI is a comma string; config wants a tuple.
    if args.imitation_agents is not None:
        agents = tuple(
            a.strip() for a in args.imitation_agents.split(",") if a.strip()
        )
    elif resumed_train_config and resumed_train_config.get("imitation_agents"):
        agents = tuple(resumed_train_config["imitation_agents"])
    else:
        agents = ("forge-full",)

    return TrainDraftAgentConfig(
        # Architecture is inherited from the checkpoint on resume/checkpoint;
        # these only matter for a fresh run and are forbidden when loading.
        d_model=args.d_model,
        n_layers=args.n_layers if args.n_layers is not None else 4,
        n_heads=args.n_heads if args.n_heads is not None else 8,
        ff_dim=args.ff_dim,
        dropout=args.dropout if args.dropout is not None else 0.0,
        imitation_agents=agents,
        resume=Path(args.resume) if args.resume else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        **resolved,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="draft",
        description="Draft agent data generation and training tools",
    )
    subparsers = parser.add_subparsers(help="Available commands")
    _build_generate_draft_data_parser(subparsers)
    _build_train_draft_agent_parser(subparsers)
    _build_train_draft_agent_rl_parser(subparsers)
    _build_train_draft_agent_online_parser(subparsers)
    _build_validate_builder_parser(subparsers)
    _build_analyze_generated_decks_parser(subparsers)
    _build_play_draft_games_parser(subparsers)
    return parser


def _build_play_draft_games_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "play-draft-games",
        help=(
            "Play Forge matches between decks drafted in the same pod of a "
            "drafts.jsonl corpus, appending rows in the sealed match-outcome "
            "format. Tally them with scripts/analyze_winrates.py."
        ),
    )
    parser.set_defaults(func=run_play_draft_games)
    parser.add_argument(
        "--drafts-path", default="output/draft/drafts.jsonl",
        help="Corpus to sample from (default: output/draft/drafts.jsonl)",
    )
    parser.add_argument(
        "--run-id", action="append", dest="run_ids", default=None,
        help=(
            "Restrict sampling to records with this run_id; repeatable. "
            "Absent, the whole corpus is in scope."
        ),
    )
    parser.add_argument(
        "--output-path", default="output/draft/draft-games.txt",
        help="Append target (default: output/draft/draft-games.txt)",
    )
    parser.add_argument(
        "--n-pairings", type=int, default=None,
        help=(
            "Stop once this many matches have been recorded by this run. "
            "Absent, play until interrupted."
        ),
    )
    parser.add_argument(
        "--best-of", type=int, default=3,
        help="Games per match; must be a positive odd integer (default: 3)",
    )
    parser.add_argument(
        "--include-mirrors", action="store_true",
        help="Keep pairings whose two seats carry the same agent label",
    )
    parser.add_argument(
        "--forge-native-fraction", type=float, default=0.0,
        help=(
            "Share of forge-full seats whose deck Forge rebuilds from their "
            "drafted pool; those seats are labelled forge-native (default: 0)"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=12,
        help="Concurrent Forge worker processes (default: 12)",
    )


def _build_analyze_generated_decks_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "analyze-generated-decks",
        help=(
            "Aggregate composition stats (colors, curve, types, lands, pips, "
            "rarity) over the per-seat decks in a drafts.jsonl corpus, plus a "
            "per-label deck-score summary (agent vs Forge). Per-label breakdowns "
            "follow the global report."
        ),
    )
    parser.set_defaults(func=run_analyze_generated_decks)
    parser.add_argument(
        "--agent", required=True,
        help=(
            "Restrict the analysis to seats piloted by this agent/mix label "
            "(e.g. draft-agent, forge-full). Run once per agent to compare."
        ),
    )
    parser.add_argument(
        "--drafts-path", default="output/draft/drafts.jsonl",
        help="Drafts corpus to analyze (default: output/draft/drafts.jsonl).",
    )
    parser.add_argument(
        "--cards-path", default="output/cardsfolder/",
        help="Directory with converted card files (default: output/cardsfolder/).",
    )
    parser.add_argument(
        "--no-rarity", action="store_true",
        help="Skip the rarity-distribution section (no MTGJSON load).",
    )


_ONLINE_DEFAULT_MIX = "gen-3:5,gen-1:3,forge-r30:1,forge-r100:1"


def _build_train_draft_agent_online_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-draft-agent-online",
        help=(
            "Online self-play GRPO fine-tuning (gen-3): generate fresh drafts "
            "from the current policy, take one critic-free "
            "-A*logpi_T(a|s) pass over them, discard, regenerate"
        ),
    )
    parser.set_defaults(func=run_train_draft_agent_online)

    # -- Agent wiring (FR-003) -------------------------------------------------
    parser.add_argument(
        "--learner", action="append", default=None, metavar="LABEL=PATH",
        help="The agent under training (required, exactly one). LABEL is its mix "
             "label; PATH warm-starts the policy at round 0. A bare PATH is not "
             "accepted — the label is load-bearing.",
    )
    parser.add_argument(
        "--frozen", action="append", default=None, metavar="LABEL=PATH",
        help="An untrained reference bound to a checkpoint (repeatable), e.g. the "
             "gen-1 anchor. Never enters the gradient.",
    )
    parser.add_argument(
        "--anchor", default=None, metavar="LABEL",
        help="Which --frozen label is the anchor-margin baseline. Required only "
             "when more than one --frozen is given.",
    )
    parser.add_argument(
        "--mix", default=_ONLINE_DEFAULT_MIX, metavar="LABEL:WEIGHT,...",
        help=f"Per-seat categorical over learner + frozen + Forge built-ins "
             f"(default: {_ONLINE_DEFAULT_MIX}).",
    )

    # -- Deck building & reward (same meaning as generate-draft-data) ----------
    parser.add_argument(
        "--scorer-checkpoint", default="models/sealed/scorer/latest.pt",
        help="Frozen scorer producing each seat's deck_score (the reward). Must exist.",
    )
    parser.add_argument(
        "--build-method", choices=("greedy", "picker"), default="greedy",
        help="Pool -> deck before scoring (default: greedy).",
    )
    parser.add_argument(
        "--picker-checkpoint", default="models/sealed/picker/latest.pt",
        help="Used only with --build-method picker; must exist then.",
    )
    parser.add_argument(
        "--cards-path", default="output/cardsfolder/",
        help=".npz embedding cache; fixes the model/scorer input width.",
    )

    # -- Rollout & optimisation ------------------------------------------------
    parser.add_argument(
        "--agent-temp", default=None, dest="agent_temp", metavar="LABEL=T,...",
        help="Per-label sampling temperature; an omitted label plays argmax. "
             "Must name the --learner with a value > 0 -- that value is also "
             "what EVERY policy distribution (logpi, entropy, KL) is evaluated "
             "at. Naming a Forge built-in or an unbound label is an error. "
             "Required, no default.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="AdamW learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--drafts-per-round", type=int, default=10, dest="drafts_per_round",
        help="Fresh drafts generated and trained on (one pass) per round (default: 10).",
    )
    parser.add_argument(
        "--anchor-window", type=int, default=100, dest="anchor_window",
        help="Sliding window (drafts) backing the anchor margin (default: 100).",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=25, dest="snapshot_every",
        help="Rounds between timestamped snapshots; latest.pt every round and "
             "best_*.pt on each new best margin (default: 25).",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Stop after N consecutive rounds with no new best anchor margin. "
             "Off by default. Must exceed --anchor-window / --drafts-per-round.",
    )
    parser.add_argument(
        "--lr-decay-patience", type=int, default=None, dest="lr_decay_patience",
        help="Anneal the LR after N rounds without a new best margin, resetting "
             "the stall counter so a decay pre-empts --patience. Off by default. "
             "Must sit between the window in rounds and --patience.",
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=0.1, dest="lr_decay_factor",
        help="LR multiplier per plateau decay (default: 0.1).",
    )
    parser.add_argument(
        "--min-lr", type=float, default=None, dest="min_lr",
        help="Annealing floor (default: lr * 1e-3); an armed --patience only "
             "fires here.",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None, dest="max_rounds",
        help="Optional round budget; omitted => runs until Ctrl-C.",
    )
    parser.add_argument(
        "--set", dest="set_code", default=None,
        help="Restrict every draft to one set; else a random sealed-legal set per draft.",
    )
    parser.add_argument(
        "--output-path", default="output/draft/drafts.jsonl", dest="output_path",
        help="Corpus appended with every generated draft (shared file).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Torch/numpy init, batch shuffling, pick-sampling RNG. Forge-side "
             "randomness is NOT seeded (default: 42).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, dest="batch_size",
        help="States per gradient step (default: 32).",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, dest="max_grad_norm",
        help="Per-parameter-group gradient-norm cap (default: 1.0).",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=200, dest="warmup_steps",
        help="Linear LR ramp over the first N optimizer steps of the run, then "
             "constant (default: 200).",
    )
    parser.add_argument(
        "--max-consecutive-faults", type=int, default=5,
        dest="max_consecutive_faults",
        help="Abort after this many consecutive pick-fault-abandoned drafts "
             "(default: 5).",
    )


def _build_validate_builder_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-builder",
        help=(
            "Picker-vs-SA builder-validation diagnostic (FR-042): build each pool "
            "with the picker and SA, score both, and print the gating "
            "picker-vs-SA Spearman, the SA-picker score-gap median/IQR, and the "
            "SA-vs-SA reference ceiling. Run once per picker/scorer pair to choose "
            "--build-method."
        ),
    )
    parser.set_defaults(func=run_validate_builder)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pools-from",
        help="drafts.jsonl whose per-seat drafted pools are the eval pools.",
    )
    source.add_argument(
        "--fresh-pools", action="store_true",
        help=(
            "Run a fresh draft session and use its drafted pools (the 45-card "
            "shape the picker labels), instead of an existing corpus. Needs Forge."
        ),
    )
    parser.add_argument(
        "--set", dest="set_code", default=None,
        help="Set code for --fresh-pools drafts (random sealed-legal set if omitted).",
    )
    parser.add_argument(
        "--n-pools", type=int, default=300,
        help="Number of pools to evaluate (default: 300).",
    )
    parser.add_argument(
        "--scorer-checkpoint", default="models/sealed/scorer/latest.pt",
        help="Frozen scorer (default: models/sealed/scorer/latest.pt).",
    )
    parser.add_argument(
        "--picker-checkpoint", default="models/sealed/picker/latest.pt",
        help="Picker (default: models/sealed/picker/latest.pt).",
    )
    parser.add_argument(
        "--cards-path", default="output/cardsfolder/",
        help=".npz cache (default: output/cardsfolder/).",
    )


def _build_generate_draft_data_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "generate-draft-data",
        help=(
            "Drive Forge's draft AI for all pod seats, build+score a deck per "
            "seat, and append one self-contained JSON record per draft"
        ),
    )
    parser.set_defaults(func=run_generate_draft_data)
    parser.add_argument(
        "--n-drafts", type=int, required=True,
        help="Number of complete drafts to generate (counts pre-existing under --resume).",
    )
    parser.add_argument(
        "--set", default=None, dest="set_code",
        help=(
            "Restrict all drafts to one set code; else each draft independently "
            "picks a random sealed-legal set (FR-009)."
        ),
    )
    parser.add_argument(
        "--agent-mix", default="forge-full:6,forge-r30:1,forge-r100:1",
        help=(
            "Categorical agent weights, 'label:weight,...'; each of pod_size "
            "seats is sampled independently (FR-006). "
            "Default: forge-full:6,forge-r30:1,forge-r100:1"
        ),
    )
    parser.add_argument(
        "--scorer-checkpoint", default="models/sealed/scorer/latest.pt",
        help="Frozen scorer labeling decks (default: models/sealed/scorer/latest.pt).",
    )
    parser.add_argument(
        "--build-method", default="picker", choices=["picker", "greedy"],
        help="Deck builder for labeling (default: picker).",
    )
    parser.add_argument(
        "--picker-checkpoint", default="models/sealed/picker/latest.pt",
        help=(
            "Picker used when --build-method picker; ignored otherwise "
            "(default: models/sealed/picker/latest.pt)."
        ),
    )
    parser.add_argument(
        "--cards-path", default="output/cardsfolder/",
        help=".npz embedding cache (default: output/cardsfolder/).",
    )
    parser.add_argument(
        "--output-path", default="output/draft/drafts.jsonl",
        help="Destination JSONL; appended, created if missing.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append + count pre-existing drafts toward --n-drafts (FR-012).",
    )
    # Live model-pilot flags (opt-in; absent ⇒ byte-for-byte gen-1, SC-004).
    parser.add_argument(
        "--agent-checkpoint", action="append", default=[], dest="agent_checkpoints",
        metavar="LABEL=PATH",
        help=(
            "Bind a mix label to a trained agent checkpoint (repeatable). A bare "
            "PATH is shorthand for label 'draft-agent'. A mix label that is "
            "neither a Forge built-in nor bound here fails fast (FR-011)."
        ),
    )
    parser.add_argument(
        "--pick-mode", default="argmax", choices=["argmax", "sample"],
        help=(
            "argmax = the policy's strongest legal card (default); sample = "
            "temperature-scaled softmax over PACK logits (rollout diversity)."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Softmax temperature for --pick-mode sample; ignored for argmax (default 1.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed the pick-sampling RNG for reproducible sampled rollouts (SC-007).",
    )
    parser.add_argument(
        "--max-consecutive-faults", type=int, default=5,
        help=(
            "Abort the run with a nonzero exit after this many consecutive "
            "pick-fault-abandoned drafts (FR-016; default 5)."
        ),
    )


# RL trainer resumable scalar flags (resolved CLI > resumed train_config >
# dataclass default). learner_agents / critic_corpus / paths handled specially.
_RL_RESUMABLE_TRAIN_FLAGS: tuple[str, ...] = (
    "drafts_path", "cards_path", "rollout_temperature", "gae_lambda",
    "kl_coef", "entropy_coef", "entropy_decay_after", "value_weight",
    "lr", "warmup_frac", "batch_size", "max_grad_norm", "epochs",
    "val_fraction", "evals_per_epoch", "patience",
    "lr_decay_patience", "lr_decay_factor", "min_lr",
)
_RL_TRAIN_PATH_FIELDS = frozenset({"drafts_path", "cards_path"})


def _resolve_train_agent_rl_config(
    args: argparse.Namespace, resumed_train_config: dict | None,
):
    """Build TrainDraftAgentRLConfig with resume precedence (CLI > resumed > default)."""
    from draft.application.train_draft_agent_rl import TrainDraftAgentRLConfig
    from sealed.infrastructure.cli_resume import resolve_resumable_args

    resolved = resolve_resumable_args(
        args,
        flag_names=_RL_RESUMABLE_TRAIN_FLAGS,
        config_cls=TrainDraftAgentRLConfig,
        path_fields=_RL_TRAIN_PATH_FIELDS,
        resumed_config=resumed_train_config,
    )

    # learner_agents: CLI is a comma string; config wants a tuple.
    if args.learner_agents is not None:
        learner = tuple(
            a.strip() for a in args.learner_agents.split(",") if a.strip()
        )
    elif resumed_train_config and resumed_train_config.get("learner_agents"):
        learner = tuple(resumed_train_config["learner_agents"])
    else:
        learner = ()

    # critic_corpus: repeatable paths; CLI overrides, else inherit, else none.
    if args.critic_corpus:
        critic = tuple(Path(p) for p in args.critic_corpus)
    elif resumed_train_config and resumed_train_config.get("critic_corpus"):
        critic = tuple(Path(p) for p in resumed_train_config["critic_corpus"])
    else:
        critic = ()

    return TrainDraftAgentRLConfig(
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        resume=Path(args.resume) if args.resume else None,
        learner_agents=learner,
        critic_corpus=critic,
        **resolved,
    )


def _build_train_draft_agent_rl_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-draft-agent-rl",
        help=(
            "RL self-play fine-tuning (gen-2): on-policy actor-critic "
            "(REINFORCE + GAE + KL anchor) over a corpus the reference "
            "checkpoint generated"
        ),
    )
    parser.set_defaults(func=run_train_draft_agent_rl)
    parser.add_argument(
        "--checkpoint", default=None,
        help="Reference checkpoint (warm-start actor+critic + KL anchor); the "
             "policy that generated --drafts-path. Required; xor --resume.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Continue an RL run (weights+optimizer+epoch+best+anneal); inherits "
             "its settings, CLI flags override. Xor --checkpoint.",
    )
    parser.add_argument(
        "--drafts-path", default=None,
        help="On-policy corpus generated by --checkpoint; the sole policy-gradient "
             "source (default: output/draft/drafts.jsonl). Resumable.",
    )
    parser.add_argument(
        "--critic-corpus", action="append", default=[], dest="critic_corpus",
        metavar="PATH",
        help="Extra off-policy corpus for critic regression/coverage only "
             "(repeatable); never the policy gradient.",
    )
    parser.add_argument(
        "--cards-path", default=None,
        help=".npz embedding cache (default: output/cardsfolder/). Resumable.",
    )
    parser.add_argument(
        "--learner-agents", default=None,
        help="Comma-separated whitelist of mix labels whose seats feed the policy "
             "gradient (the generation's own label). Required. Resumable.",
    )
    parser.add_argument(
        "--rollout-temperature", type=float, default=None, dest="rollout_temperature",
        help="Temperature the corpus was sampled at; all policy distributions use "
             "it. Required (no default) and must match generate-draft-data "
             "--temperature. Resumable.",
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=None, dest="gae_lambda",
        help="GAE λ (default: 0.95; near 1 leans on Monte-Carlo). Resumable.",
    )
    parser.add_argument(
        "--kl-coef", type=float, default=None, dest="kl_coef",
        help="Initial KL-anchor coefficient (val-driven schedule; default 0.1). Resumable.",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=None, dest="entropy_coef",
        help="Initial entropy-bonus coefficient (val-driven schedule; default 0.01). Resumable.",
    )
    parser.add_argument(
        "--entropy-decay-after", type=int, default=None, dest="entropy_decay_after",
        help="Evals of monotone val improvement before coef decay arms (default 5). Resumable.",
    )
    parser.add_argument(
        "--value-weight", type=float, default=None, dest="value_weight",
        help="Critic-regression coefficient (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="AdamW LR (default: 3e-4). Resumable (re-applied on resume).",
    )
    parser.add_argument(
        "--warmup-frac", type=float, default=None,
        help="LR linear-warmup fraction (default: 0.05). Resumable.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="States per gradient step (default: 32). Resumable.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=None,
        help="Per-group gradient-norm cap (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max epochs (default: 100). Resumable.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=None,
        help="Draft-disjoint validation fraction (default: 0.0025). Resumable.",
    )
    parser.add_argument(
        "--evals-per-epoch", type=int, default=None,
        help="Validate + checkpoint this many times per epoch (default: 100). Resumable.",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early-stop after this many evals without val improvement (default: 30). Resumable.",
    )
    parser.add_argument(
        "--lr-decay-patience", type=int, default=None, dest="lr_decay_patience",
        help="Enable plateau LR annealing; must be < --patience. Default: disabled. Resumable.",
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=None, dest="lr_decay_factor",
        help="LR multiplier per plateau (default: 0.1). Resumable.",
    )
    parser.add_argument(
        "--min-lr", type=float, default=None, dest="min_lr",
        help="Annealing floor (default: lr * 1e-3). Resumable.",
    )


def _build_train_draft_agent_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-draft-agent",
        help="Train the imitation policy + MC-regression critic jointly on a recorded corpus",
    )
    parser.set_defaults(func=run_train_draft_agent)
    # Resumable flags register with default=None so a value resolves as:
    # explicit CLI > resumed checkpoint's train_config > dataclass default.
    parser.add_argument(
        "--drafts-path", default=None,
        help="Recorded corpus (default: output/draft/drafts.jsonl). Resumable.",
    )
    parser.add_argument(
        "--cards-path", default=None,
        help=".npz cache (default: output/cardsfolder/). Resumable.",
    )
    # Architecture flags: default=None so they can be rejected with --resume/--checkpoint.
    parser.add_argument(
        "--d-model", type=int, default=None, dest="d_model",
        help="Internal width (default: derived); a non-default value inserts a Linear.",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None, dest="n_layers",
        help="SAB layers (default: 4).",
    )
    parser.add_argument(
        "--n-heads", type=int, default=None, dest="n_heads",
        help="Attention heads (default: 8); d_model %% n_heads == 0 (fail fast).",
    )
    parser.add_argument(
        "--ff-dim", type=int, default=None, dest="ff_dim",
        help="Feed-forward width (default: 4 * d_model).",
    )
    parser.add_argument(
        "--dropout", type=float, default=None, dest="dropout",
        help="Transformer dropout (default: 0.0).",
    )
    parser.add_argument(
        "--imitation-weight", type=float, default=None,
        help="CE coefficient; 0 => critic-only (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--critic-weight", type=float, default=None,
        help="MSE coefficient; 0 => imitation-only (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--imitation-agents", default=None,
        help="Comma-separated whitelist of imitation-target agents (default: "
             "forge-full). Resumable.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="AdamW LR (default: 3e-4). Resumable (also re-applied to a resumed "
             "optimizer, so --lr changes the LR on resume).",
    )
    parser.add_argument(
        "--warmup-frac", type=float, default=None,
        help="LR linear-warmup fraction (default: 0.05). Resumable.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="States per gradient step (default: 32). Resumable.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=None,
        help="Per-group gradient-norm cap (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max epochs (default: 100). Resumable.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=None,
        help="Draft-disjoint validation fraction (default: 0.0025 — a small "
             "held-out monitor; the huge train set makes overfit unlikely). Resumable.",
    )
    parser.add_argument(
        "--evals-per-epoch", type=int, default=None,
        help="Validate + checkpoint this many times per epoch (mini-epochs; "
             "default: 100). Resumable.",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early-stop after this many mini-epochs without val improvement "
             "(default: 30). Resumable.",
    )
    parser.add_argument(
        "--lr-decay-patience", type=int, default=None, dest="lr_decay_patience",
        help="Enable plateau LR annealing: after this many mini-epochs without a "
             "new best val_loss, LR *= --lr-decay-factor (down to --min-lr). Must "
             "be < --patience. Default: disabled (constant LR). Resumable.",
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=None, dest="lr_decay_factor",
        help="LR multiplier applied on each plateau (default: 0.1). Resumable.",
    )
    parser.add_argument(
        "--min-lr", type=float, default=None, dest="min_lr",
        help="Annealing floor; no decay below this (default: lr * 1e-3). Resumable.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Continue a run: restore weights+optimizer+epoch+best-val and "
             "inherit its training settings; CLI flags override them; arch flags "
             "forbidden; xor --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Bootstrap fresh run from weights only; architecture flags forbidden; xor --resume.",
    )


def run_generate_draft_data(args: argparse.Namespace) -> int:
    """Execute the generate-draft-data command."""
    from draft.application.agent_mix import AgentMixError, parse_agent_mix
    from draft.application.agent_registry import (
        AgentRegistryError,
        parse_agent_checkpoints,
    )
    from draft.application.generate_draft_data import (
        GenerateDraftDataConfig,
        GenerateDraftDataSupervisor,
        MaxConsecutiveFaultsError,
    )

    if args.n_drafts < 1:
        print("Error: --n-drafts must be >= 1.", file=sys.stderr)
        return 2

    try:
        agent_mix = parse_agent_mix(args.agent_mix)
    except AgentMixError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        agent_checkpoints = parse_agent_checkpoints(args.agent_checkpoints)
    except AgentRegistryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.pick_mode == "sample" and args.temperature <= 0:
        print(
            "Error: --temperature must be > 0 with --pick-mode sample.",
            file=sys.stderr,
        )
        return 2

    config = GenerateDraftDataConfig(
        n_drafts=args.n_drafts,
        set_code=args.set_code,
        agent_mix=agent_mix,
        scorer_checkpoint=Path(args.scorer_checkpoint),
        build_method=args.build_method,
        picker_checkpoint=Path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
        output_path=Path(args.output_path),
        resume=args.resume,
        agent_checkpoints=agent_checkpoints,
        pick_mode=args.pick_mode,
        temperature=args.temperature,
        seed=args.seed,
        max_consecutive_faults=args.max_consecutive_faults,
    )

    try:
        GenerateDraftDataSupervisor(config).run()
    except AgentRegistryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except MaxConsecutiveFaultsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


def run_train_draft_agent(args: argparse.Namespace) -> int:
    """Execute the train-draft-agent command."""
    from draft.application.train_draft_agent import TrainDraftAgentUseCase
    from draft.domain.draft_agent_model import DraftAgentArchitectureError
    from draft.infrastructure.draft_agent_store import DraftAgentStore

    if args.resume is not None and args.checkpoint is not None:
        print(
            "Error: --resume and --checkpoint are mutually exclusive: --resume "
            "continues an existing run; --checkpoint bootstraps a fresh run from "
            "another agent's weights.",
            file=sys.stderr,
        )
        return 2

    loading = args.resume is not None or args.checkpoint is not None
    if loading:
        for flag in _TRAIN_AGENT_ARCHITECTURE_FLAGS:
            if getattr(args, flag) is not None:
                print(
                    f"Error: architecture flag --{flag.replace('_', '-')} "
                    "conflicts with --resume / --checkpoint; architecture is "
                    "inherited from the checkpoint's stored config. Omit the flag.",
                    file=sys.stderr,
                )
                return 2

    # On --resume, inherit the run's settings from its checkpoint; CLI flags
    # override. (--checkpoint is a fresh run from weights only, so no inherit.)
    resumed_train_config: dict | None = None
    if args.resume is not None:
        try:
            resumed = DraftAgentStore().load_checkpoint(Path(args.resume))
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        resumed_train_config = resumed.train_config

    config = _resolve_train_agent_config(args, resumed_train_config)

    try:
        TrainDraftAgentUseCase().execute(config)
    except DraftAgentArchitectureError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def run_train_draft_agent_rl(args: argparse.Namespace) -> int:
    """Execute the train-draft-agent-rl command (spec 020)."""
    from draft.application.train_draft_agent_rl import TrainDraftAgentRLUseCase
    from draft.domain.draft_agent_model import DraftAgentArchitectureError
    from draft.infrastructure.draft_agent_store import DraftAgentStore

    if (args.checkpoint is None) == (args.resume is None):
        print(
            "Error: exactly one of --checkpoint or --resume is required: "
            "--checkpoint warm-starts a fresh RL run from a reference agent; "
            "--resume continues an existing RL run.",
            file=sys.stderr,
        )
        return 2

    # On --resume, inherit settings from the checkpoint's train_config.
    resumed_train_config: dict | None = None
    if args.resume is not None:
        try:
            resumed = DraftAgentStore().load_checkpoint(Path(args.resume))
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        resumed_train_config = resumed.train_config

    config = _resolve_train_agent_rl_config(args, resumed_train_config)

    if not config.learner_agents:
        print("Error: --learner-agents must be a non-empty whitelist.", file=sys.stderr)
        return 2
    if config.rollout_temperature is None or config.rollout_temperature <= 0:
        print(
            "Error: --rollout-temperature must be supplied and positive (it must "
            "match the generate-draft-data --temperature the corpus was sampled at).",
            file=sys.stderr,
        )
        return 2

    try:
        TrainDraftAgentRLUseCase().execute(config)
    except DraftAgentArchitectureError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def run_train_draft_agent_online(args: argparse.Namespace) -> int:
    """Execute the train-draft-agent-online command (spec 021)."""
    from draft.application.agent_mix import AgentMixError, parse_agent_mix
    from draft.application.agent_registry import AgentRegistryError
    from draft.application.generate_draft_data import MaxConsecutiveFaultsError
    from draft.application.train_draft_agent_online import (
        OnlineConfigError,
        TrainDraftAgentOnlineConfig,
        TrainDraftAgentOnlineUseCase,
        parse_agent_temperatures,
        parse_frozen_specs,
        parse_learner_spec,
        validate_config,
    )
    from draft.domain.draft_agent_model import DraftAgentArchitectureError

    try:
        learner_label, learner_checkpoint = parse_learner_spec(args.learner)
        frozen = parse_frozen_specs(args.frozen)
        mix = parse_agent_mix(args.mix)
        agent_temperatures = parse_agent_temperatures(args.agent_temp)
    except (OnlineConfigError, AgentMixError, AgentRegistryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    config = TrainDraftAgentOnlineConfig(
        learner_label=learner_label,
        learner_checkpoint=learner_checkpoint,
        frozen=frozen,
        anchor=args.anchor,
        mix=mix,
        scorer_checkpoint=Path(args.scorer_checkpoint),
        build_method=args.build_method,
        picker_checkpoint=Path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
        agent_temperatures=agent_temperatures,
        lr=args.lr,
        drafts_per_round=args.drafts_per_round,
        anchor_window=args.anchor_window,
        snapshot_every=args.snapshot_every,
        patience=args.patience,
        lr_decay_patience=args.lr_decay_patience,
        lr_decay_factor=args.lr_decay_factor,
        min_lr=args.min_lr,
        max_rounds=args.max_rounds,
        set_code=args.set_code,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        output_path=Path(args.output_path),
        max_consecutive_faults=args.max_consecutive_faults,
    )

    # Every rule runs before the Forge worker launches and before any update
    # (FR-024, SC-006); execute() re-runs it so the use case is safe standalone.
    try:
        validate_config(config)
    except OnlineConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        TrainDraftAgentOnlineUseCase().execute(config)
    except DraftAgentArchitectureError as exc:  # a ValueError subclass — catch first
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    except MaxConsecutiveFaultsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # The loop installs SIGINT handlers and shuts down cleanly, printing the
        # final summary; reaching here means the interrupt landed during startup.
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (AgentRegistryError, AgentMixError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def run_validate_builder(args: argparse.Namespace) -> int:
    """Execute the validate-builder diagnostic (FR-042)."""
    from draft.application.validate_builder import (
        ValidateBuilderConfig,
        format_diagnostic,
        run_validate,
    )

    config = ValidateBuilderConfig(
        pools_from=Path(args.pools_from) if args.pools_from else None,
        fresh_pools=args.fresh_pools,
        set_code=args.set_code,
        n_pools=args.n_pools,
        scorer_checkpoint=Path(args.scorer_checkpoint),
        picker_checkpoint=Path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
    )
    try:
        diagnostic = run_validate(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    print(format_diagnostic(diagnostic))
    return 0


def run_analyze_generated_decks(args: argparse.Namespace) -> int:
    """Composition + per-label deck-score stats over a drafts.jsonl corpus."""
    from draft.application.analyze_generated_decks import (
        available_agents,
        deck_score_summary,
        format_deck_score_section,
        generated_decks_from_drafts,
    )
    from draft.infrastructure.draft_record_io import read_records
    from sealed.application.analyze_generated_decks import analyze_decks

    drafts_path = Path(args.drafts_path)
    records = list(read_records(drafts_path))
    if not records:
        print(f"Error: no records in {drafts_path}", file=sys.stderr)
        return 2

    decks = generated_decks_from_drafts(records, agent=args.agent)
    if not decks:
        available = available_agents(records)
        print(
            f"Error: no built decks for agent {args.agent!r} in {drafts_path}. "
            f"Available agents: {', '.join(available) or '(none)'}",
            file=sys.stderr,
        )
        return 2

    print(format_deck_score_section(deck_score_summary(records, agent=args.agent)))
    print()

    analyze_decks(
        decks, Path(args.cards_path),
        no_rarity=args.no_rarity,
        source_summary=[(f"{drafts_path} [agent={args.agent}]", len(decks))],
    )
    return 0


def run_play_draft_games(args: argparse.Namespace) -> int:
    """Play same-pod matches and append them in the sealed match-outcome format."""
    from draft.application.play_draft_games import (
        PlayDraftGamesConfig,
        PlayDraftGamesUseCase,
    )
    from draft.domain.seat_table import project
    from draft.infrastructure.draft_record_io import read_records

    drafts_path = Path(args.drafts_path)
    run_ids = tuple(args.run_ids or ())

    # ── Startup validation: everything checkable before a game is played ──
    if args.best_of < 1 or args.best_of % 2 == 0:
        print(
            f"Error: --best-of must be a positive odd integer, got {args.best_of}",
            file=sys.stderr,
        )
        return 2
    if not 0.0 <= args.forge_native_fraction <= 1.0:
        print(
            "Error: --forge-native-fraction must be between 0 and 1, got "
            f"{args.forge_native_fraction}",
            file=sys.stderr,
        )
        return 2
    if args.n_pairings is not None and args.n_pairings < 1:
        print(
            f"Error: --n-pairings must be a positive integer, got {args.n_pairings}",
            file=sys.stderr,
        )
        return 2
    if args.workers < 1:
        print(
            f"Error: --workers must be a positive integer, got {args.workers}",
            file=sys.stderr,
        )
        return 2
    if not drafts_path.exists():
        print(f"Error: corpus not found: {drafts_path}", file=sys.stderr)
        return 2

    records = [r for r in read_records(drafts_path) if not run_ids or r.run_id in run_ids]
    if not records:
        scope = f" matching --run-id {', '.join(run_ids)}" if run_ids else ""
        print(f"Error: no records in {drafts_path}{scope}", file=sys.stderr)
        return 2

    rows = list(project(records))
    labels = sorted({row.label for row in rows})
    if not _has_playable_pair(rows, include_mirrors=args.include_mirrors):
        print(
            "Error: no pair survives the --run-id and mirror filters. "
            f"Labels present: {', '.join(labels) or '(none)'}. "
            "Pass --include-mirrors to play same-label pairings.",
            file=sys.stderr,
        )
        return 2

    # ── Startup echo ──
    scope = ", ".join(run_ids) if run_ids else "whole corpus"
    print(f"Corpus:      {drafts_path} ({len(records)} records, {len(rows)} seats)")
    print(f"Run ids:     {scope}")
    print(f"Labels:      {', '.join(labels)}")
    print(
        "Pairings:    "
        + (f"{args.n_pairings}" if args.n_pairings is not None else "unbounded (Ctrl-C to stop)")
    )
    print(
        f"Matches:     best-of-{args.best_of}, {args.workers} workers, "
        f"mirrors {'included' if args.include_mirrors else 'excluded'}"
    )
    print(f"Output:      {args.output_path}")

    config = PlayDraftGamesConfig(
        drafts_path=drafts_path,
        output_path=Path(args.output_path),
        run_ids=run_ids,
        n_pairings=args.n_pairings,
        best_of=args.best_of,
        include_mirrors=args.include_mirrors,
        forge_native_fraction=args.forge_native_fraction,
        workers=args.workers,
    )

    summary = PlayDraftGamesUseCase().execute(config)
    # 130 only for an interrupt that arrived before the first match completed.
    # A clean bounded finish, and an interrupt after any match, are both 0.
    if summary.interrupted and summary.matches_played == 0:
        return 130
    return 0


def _has_playable_pair(rows, *, include_mirrors: bool) -> bool:
    """True when some pod holds two seats that could legally be paired."""
    by_pod: dict[str, list[str]] = {}
    for row in rows:
        by_pod.setdefault(row.draft_id, []).append(row.label)
    for labels in by_pod.values():
        if len(labels) < 2:
            continue
        if include_mirrors or len(set(labels)) > 1:
            return True
    return False


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
