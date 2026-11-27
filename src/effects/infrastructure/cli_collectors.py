"""The two collectors, ``collect-coverage`` and ``collect-variants``: their
parsers, the caps and config they share, and the refusals each checks before
any worker starts.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from effects.application.collect_coverage import CollectCoverageConfig
from effects.application.collect_variants import CollectVariantsConfig
from effects.application.collector_common import (
    CollectionFlags,
    collection_target,
)
from effects.domain.collection_caps import DEFAULT_RANDOM_SEAT_PROBABILITY, CollectionCaps
from effects.domain.shard_generation import UnacceptedGenerationError
from effects.infrastructure.cli_common import (
    _LAYOUT_REFUSED,
    _add_cards_folder,
    _add_effect_records,
    _add_holdout_flags,
    _add_holdout_sources,
    _Config,
    _optional_path,
    add_config_flags,
    config_from_args,
    exits_with,
    holdout_rule,
)
from sealed.infrastructure.effect_collection_flags import (
    add_cap_flags,
    add_directed_fork_flags,
    announce_probe_state,
    candidate_rate_error,
    random_seat_error,
    rule_list,
)


def _collect_coverage_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "collect-coverage",
        help=(
            "Play weighted decks over the whole converted corpus until every "
            "card is satisfied or retired"
        ),
    )
    parser.set_defaults(func=run_collect_coverage)
    _add_effect_records(parser, required=False, help_text=(
        "Destination shard directory, a records run; this or --training-corpus "
        "names it. Unlike on match-outcomes, collection is the point of this command"
    ))
    _add_cards_folder(parser)
    _add_holdout_sources(parser)
    parser.add_argument(
        "--only-cards", type=_optional_path, default=None,
        help=(
            "The held-out coverage round over the list 'holdout-cards' writes: "
            "decks from these cards alone, counted per held-out text. Refused "
            "beside any other holdout source"
        ),
    )
    # The held-out round credits a game to a held-out text under this rule,
    # so it has to be the one build-corpus is given.
    _add_holdout_flags(parser)
    add_config_flags(parser, CollectCoverageConfig, only=(
        "wording_cards", "min_text_games", "target_records", "no_progress_rounds",
        "decks_per_round", "workers", "stop_timeout", "recycle_seconds", "seed",
        "allow_degraded",
    ))
    parser.add_argument(
        "--count-from", type=Path, action="append", default=None, metavar="DIR",
        help=(
            "An earlier run's records directory whose coverage counts toward "
            "satisfaction; read once at the start and never written. "
            "Repeatable; refused when it overlaps --effect-records or another"
        ),
    )
    parser.add_argument(
        "--directed-fork-rules", type=rule_list, default=(),
        help=(
            "Comma-separated rule ids written to every record's aimed_at "
            "(FR-009a, FR-012); a label only: this command runs no directed fork"
        ),
    )
    add_cap_flags(parser)


def _collect_variants_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "collect-variants",
        help=(
            "Perturb Forge card scripts by one parameter and play the results, "
            "so the corpus contains texts no real card prints"
        ),
    )
    parser.set_defaults(func=run_collect_variants)
    _add_effect_records(
        parser, required=False,
        help_text="Destination shard directory, a records run; this or --training-corpus names it",
    )
    add_config_flags(parser, CollectVariantsConfig, only=(
        "forge_cards_path", "variant_scripts", "round", "variant_games",
        "no_progress_rounds", "rule_counts", "steer_floor", "steer_variants",
        "steer_min_gain", "steer_iterations", "seed",
    ))
    _add_holdout_sources(parser, excluded="held-out cards, and their variants,")
    # A variant printing a held-out text is dropped under this rule, so it has
    # to be the one build-corpus is given.
    _add_cards_folder(parser)
    _add_holdout_flags(parser)
    add_config_flags(parser, CollectVariantsConfig, only=(
        "decks_per_round", "workers", "stop_timeout", "recycle_seconds",
        "allow_degraded",
    ))
    parser.add_argument(
        "--continues", action="append", default=None, metavar="RUN_ID",
        help=(
            "The run id of an interrupted collect-variants run into the same "
            "--effect-records (its run.json's invocations name it): its shards' "
            "games count before the first round, so no finished variant is "
            "played again. Pass the same --seed. Repeatable; untargeted round only"
        ),
    )
    add_cap_flags(parser)
    # Directed forks run here as on match-outcomes, startup check included
    # (collect_variants.run); the rules also aim every record.
    add_directed_fork_flags(parser)
    parser.set_defaults(takes_directed_forks=True)


def collection_caps(args: argparse.Namespace) -> CollectionCaps:
    """The caps of a ``collect-coverage`` or ``collect-variants`` run.

    Every record of either is aimed (FR-009a): ``aimed_at`` lists the run's
    ``--directed-fork-rules``, and is empty, never absent, where it names none.
    Only ``collect-variants`` takes directed forks for them (FR-012); on
    ``collect-coverage`` the list is a label and no fork is taken.
    """
    rules = tuple(getattr(args, "directed_fork_rules", ()))
    # An unset --random-seat-probability is None so a share above 0 without
    # one can be refused; with the share at 0 the worker never reads it.
    probability = args.random_seat_probability
    return config_from_args(
        CollectionCaps, args,
        random_seat_probability=(
            DEFAULT_RANDOM_SEAT_PROBABILITY if probability is None else probability
        ),
        aimed_at=rules,
        directed_fork_rules=rules if getattr(args, "takes_directed_forks", False) else (),
    )


def collector_config(
    config_type: type[_Config], args: argparse.Namespace, **overrides: Any,
) -> _Config:
    """A collector's config, with its corpus location, deck holdout and
    holdout rule resolved; raises ValueError on a contradictory holdout,
    before any worker starts."""
    effect_records, deck_holdout = collection_target(CollectionFlags(
        effect_records=args.effect_records,
        exclude_cards=args.exclude_cards,
        split_from=args.split_from,
        training_corpus=args.training_corpus,
        only_cards=getattr(args, "only_cards", None),
    ))
    return config_from_args(
        config_type, args,
        effect_records=effect_records, deck_holdout=deck_holdout,
        caps=collection_caps(args), holdout=holdout_rule(args),
        **overrides,
    )


class CollectorRefusal(ValueError):
    """A collector's flags contradict each other: refused, naming the command,
    before any worker starts."""


def holdout_source_error(args: argparse.Namespace) -> str | None:
    """Why a collector cannot run without a holdout source, or None (FR-058g).

    Without ``--exclude-cards``, ``--split-from`` or ``--training-corpus`` a
    coverage round decks the held-out cards and a variant round varies them,
    so their texts reach training. The held-out round (``--only-cards``) decks
    them on purpose and needs none.
    """
    if getattr(args, "only_cards", None) is not None:
        return None
    if any(
        getattr(args, name, None) is not None
        for name in ("exclude_cards", "split_from", "training_corpus")
    ):
        return None
    return (
        "no holdout source: pass --exclude-cards, --split-from or "
        "--training-corpus, or this run decks the held-out cards and their "
        "texts reach training"
    )


def steered_round_error(args: argparse.Namespace) -> str | None:
    """Why a steered variant round cannot start, or None (FR-058g).

    It aims at the pairs the counts of the corpus being collected show short,
    so it needs ``--rule-counts``, written by ``rule-counts`` over gen-3 records.
    """
    from effects.domain.variant_rounds import VariantRound

    if getattr(args, "round", None) is not VariantRound.STEERED:
        return None
    if args.rule_counts is None:
        return "--round steered needs --rule-counts, the corpus's own gen-3 counts"
    try:
        document = json.loads(Path(args.rule_counts).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"--rule-counts {args.rule_counts}: {exc}"
    generation = document.get("generation")
    if generation is None:
        return (
            f"--rule-counts {args.rule_counts} names no generation; rerun "
            "rule-counts over the corpus being collected"
        )
    if generation != "gen-3":
        return (
            f"--rule-counts {args.rule_counts} counts {generation} records; the "
            "steered round needs the gen-3 counts of the corpus being collected"
        )
    return None


def continues_error(config: Any) -> str | None:
    """Why ``collect-variants --continues`` cannot resume, or None.

    Only the untargeted round resumes: a steered run's iterations aim at the
    counts of the corpus as it stands, so it starts again from fresh
    ``rule-counts``. A run id naming no shard under ``--effect-records`` is a
    typo or another directory, and would resume nothing.
    """
    from effects.domain.variant_rounds import VariantRound
    from effects.infrastructure.shard_io import iter_shards

    continues = getattr(config, "continues", ())
    if not continues:
        return None
    if config.round is VariantRound.STEERED:
        return (
            "--continues resumes the untargeted round only; rerun rule-counts "
            "and start a new steered run"
        )
    names = [shard.name for shard in iter_shards(config.effect_records)]
    for run_id in continues:
        if not any(name.startswith(f"{run_id}.") for name in names):
            return f"--continues {run_id}: no shard of that run under {config.effect_records}"
    return None


def _flag_error(args: argparse.Namespace) -> str | None:
    """The first refusal the collector's flags earn before any config is built."""
    return (
        random_seat_error(args) or candidate_rate_error(args)
        or holdout_source_error(args) or steered_round_error(args)
    )


def _check_workers_can_start(command: str, config: Any) -> None:
    """The connector JAR, ``java`` and the list files the run reads, checked
    before any worker starts, so a missing one exits 1 naming it
    rather than in a traceback."""
    import price_predictor.infrastructure.forge_classpath as forge_classpath

    holdout = config.deck_holdout
    try:
        forge_classpath.check_worker_prerequisites(files={
            "--exclude-cards": holdout.exclude_cards, "--only-cards": holdout.only_cards,
            "--split-from": holdout.split_from,
            "--wording-cards": getattr(config, "wording_cards", None),
        })
    except FileNotFoundError as exc:
        raise CollectorRefusal(f"{command}: {exc}") from exc


def _check_conversion(command: str, config: Any) -> None:
    """The conversion the ``--cards-folder`` trees come from, checked against
    the connector JAR before any worker starts and kept in the run's
    ``run.json``: a tree naming no conversion, or one converted
    at another Forge commit than the JAR's, or a records directory already
    recording another conversion, exits 1 naming both."""
    from price_predictor.domain.converted_provenance import ConversionRefused
    from price_predictor.infrastructure.records_conversion import record_collection

    try:
        record_collection(config.effect_records, config.source_trees.conversion())
    except ConversionRefused as exc:
        raise CollectorRefusal(f"{command}: {exc}") from exc


def _collector_runner(command: str, config_type: type, module: str) -> Callable:
    """The ``run_*`` of one collector: its config, refused as a message on a
    bad flag combination, then ``module.run`` over it.

    ``module`` is imported when the command runs, as every other ``run_*``
    imports its application module, so ``--help`` never pays for it. A
    ValueError out of the collection itself is not a refusal and keeps its
    traceback; a record of an older generation met counting the shards
    already there is a refusal, as every reader's, and so is a run stopped
    for its workers' engine (``WorkerEngineRefused``).
    """

    @exits_with({
        CollectorRefusal: 1, _LAYOUT_REFUSED: 1, UnacceptedGenerationError: 1,
        "price_predictor.infrastructure.forge_worker_pool.WorkerStartError": 1,
        "price_predictor.domain.worker_status.WorkerEngineRefused": 1,
    })
    def run(args: argparse.Namespace) -> int:
        error = _flag_error(args)
        if error is not None:
            raise CollectorRefusal(f"{command}: {error}")
        try:
            config = collector_config(config_type, args)
        except ValueError as exc:
            raise CollectorRefusal(f"{command}: {exc}") from exc
        error = continues_error(config)
        if error is not None:
            raise CollectorRefusal(f"{command}: {error}")
        _check_workers_can_start(command, config)
        _check_conversion(command, config)
        print(announce_probe_state(args.probe_keywords, args.probes_per_game))
        return importlib.import_module(module).run(config)

    run.__name__ = run.__qualname__ = f"run_{command.replace('-', '_')}"
    return run


run_collect_coverage = _collector_runner(
    "collect-coverage", CollectCoverageConfig, "effects.application.collect_coverage",
)
run_collect_variants = _collector_runner(
    "collect-variants", CollectVariantsConfig, "effects.application.collect_variants",
)
