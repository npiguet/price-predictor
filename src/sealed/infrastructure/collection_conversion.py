"""``match-outcomes --effect-records``: the conversion its records join against.

An effect record names its abilities by provenance keys that join against a
converted tree's sidecars, so a collection names that tree, ``--cards-folder``,
and refuses to start when the tree's conversion was made at another Forge
commit than the connector JAR its workers run
(``price_predictor.infrastructure.converted_provenance``). A plain
``match-outcomes`` run writes no keyed record, so it names no tree and
nothing is checked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CARDS_FOLDER_HELP = (
    "The converted card tree the effect records' provenance keys join against"
    " (output/<module>/<gen>/converted/cardsfolder/): the conversion its"
    " run.json records must be at the connector JAR's Forge commit, or the run"
    " exits 1 before any worker starts. Required with --effect-records, and"
    " only with it"
)


def add_cards_folder_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cards-folder", type=Path, default=None, help=CARDS_FOLDER_HELP)


def cards_folder_error(args: argparse.Namespace) -> str | None:
    """Why ``--cards-folder`` and ``--effect-records`` cannot run as given, or None."""
    collecting = bool(getattr(args, "effect_records", None))
    named = getattr(args, "cards_folder", None) is not None
    if collecting and not named:
        return (
            "--effect-records needs --cards-folder: the converted tree the records'"
            " provenance keys join against, whose conversion is checked against the"
            " connector's Forge commit"
        )
    if named and not collecting:
        return (
            "--cards-folder needs --effect-records: without it no record is keyed,"
            " so nothing joins against a conversion"
        )
    return None


def check_conversion(args: argparse.Namespace, effect_records_dir: Path | None) -> None:
    """Refuse a collection whose ``--cards-folder`` names no conversion, or one
    made at another Forge commit than the connector's, or into a records
    directory already recording another conversion (``ConversionRefused``);
    keep the conversion in the run's ``run.json``.
    Nothing for a run collecting no effect records."""
    if effect_records_dir is None:
        return
    from price_predictor.infrastructure.converted_provenance import collection_conversion
    from price_predictor.infrastructure.records_conversion import record_collection

    record_collection(effect_records_dir, collection_conversion([Path(args.cards_folder)]))
