"""``validate-corpus``: a shard directory measured against the corpus invariants."""

from __future__ import annotations

import argparse

from effects.application.corpus_validation.findings import Thresholds
from effects.domain.shard_generation import UnacceptedGenerationError
from effects.infrastructure.cli_common import (
    _NO_SHARDS,
    _add_cards_folder,
    _add_effect_records,
    _add_variant_scripts,
    add_config_flags,
    config_from_args,
    exits_with,
)


def _validate_corpus_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-corpus",
        help=(
            "Check a shard directory against the corpus invariants and fail "
            "loudly on any that a collector is breaking"
        ),
    )
    parser.set_defaults(func=run_validate_corpus)
    _add_effect_records(parser, help_text="Shard directory to read")
    parser.add_argument(
        "--limit", type=int, default=0,
        help=(
            "Stop after this many records, 0 reading everything; every record "
            "id is held in memory (default: %(default)s)"
        ),
    )
    add_config_flags(parser, Thresholds)
    _add_cards_folder(parser)
    _add_variant_scripts(parser, required=True, help_text=(
        "The perturbed-script tree whose keys a collect-variants corpus names; "
        "a key in a tree not given fails the keyword-join check as unchecked; "
        "empty reads none"
    ))


@exits_with({_NO_SHARDS: 1, UnacceptedGenerationError: 1})
def run_validate_corpus(args: argparse.Namespace) -> int:
    """Every invariant prints its measurement whether it passed or not: the
    number is what tells an operator whether a fix worked or merely moved."""
    from effects.application.validate_corpus import (
        CorpusPorts,
        DirectoryOptions,
        validate_directory,
    )

    root = args.effect_records
    source, record_renderer = validation_ports()
    report = validate_directory(
        root, ports=CorpusPorts(source, record_renderer),
        options=DirectoryOptions(
            limit=args.limit, thresholds=config_from_args(Thresholds, args),
            sidecars=_sidecars_for_validation(args),
        ),
    )
    findings = report.findings
    window = f" (first {args.limit} records)" if args.limit else ""
    print(f"{report.shards} shards under {root}{window}\n")
    for finding in findings:
        print("\n".join(finding.lines()))
    print("\n" + "\n".join(findings.summary()))
    return 1 if findings.broken() else 0


def validation_ports():
    """The shard reader and record renderer the validator runs on: the
    application layer imports no infrastructure, so this composition root
    hands them in."""
    from effects.application.corpus_validation.ports import ShardSource
    from effects.infrastructure import shard_io
    from effects.infrastructure.record_codec import record_to_dict

    tally = shard_io.DecodeTally()

    def read_window(root, limit, observe, *, shards):
        return shard_io.read_window(
            root, shard_io.ReadWindow(limit, observe), shards=shards, tally=tally,
        )

    return ShardSource(
        require_shards=shard_io.require_shards,
        require_generation=shard_io.require_generation,
        decode_tally=tally,
        read_window=read_window,
        damage=shard_io.SHARD_DAMAGE,
    ), record_to_dict


def _sidecars_for_validation(args: argparse.Namespace):
    """A ``SidecarCache`` over the converted trees, or None where none exist.

    None rather than an empty cache: a cache whose every lookup fails reads as
    a corpus that does not join, when this machine merely has no tree to join
    against, and the keyword-join check reports None as "not checked".
    """
    from effects.infrastructure.converted_trees import SourceTrees

    variants = getattr(args, "variant_scripts", None)
    trees = SourceTrees.of((path for path in args.cards_folders if path.is_dir()),
                           variants if variants is not None and variants.is_dir() else None)
    return trees.sidecar_cache() if trees.cards_folders else None
