"""The converted-card corpus commands of the ``price_predictor`` CLI: ``vocabulary``,
``convert`` and ``check-convert``, parsers and handlers."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

from price_predictor.infrastructure.cli_common import require_path_exists

logger = logging.getLogger(__name__)


#: ``check-convert``'s report, by default in the converted tree it checks.
CHECK_REPORT_NAME = "check_report.txt"


def add_corpus_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register ``vocabulary``, ``convert`` and ``check-convert``."""
    _add_vocabulary(subparsers)
    _add_convert(subparsers)
    _add_check_convert(subparsers)


def _add_vocabulary(subparsers: argparse._SubParsersAction) -> None:
    vocab_parser = subparsers.add_parser(
        "vocabulary", help="Build custom MTG tokenizer vocabulary from card corpus"
    )
    vocab_parser.add_argument(
        "--output-dir", type=str, default="models/price-predictor/transformer/",
        help="Directory where vocab.txt is written",
    )
    vocab_parser.add_argument(
        "--cards-path", type=str, default="./output/cardsfolder",
        help="Path to converted card corpus (directory of .txt files)",
    )
    vocab_parser.add_argument(
        "--freq-threshold", type=int, default=5,
        help="Minimum corpus occurrences for a word to be included as a token",
    )
    vocab_parser.add_argument(
        "--printings-path", type=str, default="resources/AllPrintings.json",
        help=(
            "Path to AllPrintings.json. When present, alphabetic fragments of "
            "every set code are seeded into the vocabulary so training texts "
            "carrying printing data never produce UNK for set codes. Silently skipped if absent."
        ),
    )
    vocab_parser.set_defaults(func=run_vocabulary)


def _add_convert(subparsers: argparse._SubParsersAction) -> None:
    convert_parser = subparsers.add_parser(
        "convert", help="Convert Forge card scripts to converted cards"
    )
    convert_parser.add_argument(
        "--cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder/",
        help="Path to Forge cardsfolder directory",
    )
    convert_parser.add_argument(
        "--output-path", type=str, required=True,
        help=(
            "Output directory for the converted cards: a generation's "
            "output/<module>/<gen>/converted/cardsfolder"
        ),
    )
    convert_parser.add_argument(
        "--tokens-path", type=str,
        default="../forge/forge-gui/res/tokenscripts/",
        help="Path to Forge tokenscripts directory (skipped if absent)",
    )
    convert_parser.add_argument(
        "--tokens-output-path", type=str, required=True,
        help=(
            "Output directory for converted token scripts, beside "
            "--output-path. Kept out of the card "
            "tree because converted token and card filenames collide and the "
            "sealed pipeline reads the card tree as its corpus."
        ),
    )
    convert_parser.add_argument(
        "--replace", action="store_true",
        help=(
            "Sweep converted pairs (texts with a sidecar, their temp files) out of "
            "both output trees first; other files (check_report.txt, run.json) stay. "
            "Without it, an output tree that already holds converted files is refused."
        ),
    )
    convert_parser.set_defaults(func=run_convert)


def _add_check_convert(subparsers: argparse._SubParsersAction) -> None:
    check_parser = subparsers.add_parser(
        "check-convert",
        help="Check converted files against Oracle text from Forge scripts",
    )
    check_parser.add_argument(
        "--cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder/",
        help="Path to Forge cardsfolder directory",
    )
    check_parser.add_argument(
        "--output-path", type=str, default="./output/cardsfolder",
        help="Path to converted output directory, as 'convert' writes it",
    )
    check_parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Similarity threshold below which cards are flagged (0.0-1.0)",
    )
    check_parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of results to show (0 = all)",
    )
    check_parser.add_argument(
        "--report", type=str, default=None,
        help=(
            "The full report's file, its directory the run directory "
            f"(default: {CHECK_REPORT_NAME} in --output-path)"
        ),
    )
    check_parser.set_defaults(func=run_check_convert)


def run_vocabulary(args: argparse.Namespace) -> int:
    """Execute the vocabulary command — build MTG tokenizer vocabulary from card corpus."""
    from price_predictor.application.build_vocabulary import build_vocabulary
    from price_predictor.infrastructure.tokenizer_store import save_vocabulary

    cards_path = Path(args.cards_path)
    if not require_path_exists(cards_path, "Cards path"):
        return 1

    txt_files = list(cards_path.rglob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in %s", cards_path)
        return 1

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Could not create output directory %s: %s", output_dir, e)
        return 2

    printings_path_raw = getattr(args, "printings_path", "resources/AllPrintings.json")
    printings_path: Path | None = Path(printings_path_raw)
    if not printings_path.exists():
        logger.warning(
            "AllPrintings.json not found at %s — set-code tokens will not be seeded.",
            printings_path,
        )
        printings_path = None

    result = build_vocabulary(
        cards_path,
        freq_threshold=args.freq_threshold,
        printings_path=printings_path,
    )
    vocab_path = output_dir / "vocab.txt"
    save_vocabulary(result.vocab, vocab_path)

    output = {
        "vocab_path": str(vocab_path),
        "vocab_size": result.vocab_size,
        "domain_token_count": result.domain_token_count,
        "freq_threshold_token_count": result.freq_threshold_token_count,
        "coverage_pct": result.coverage_pct,
        "unk_pct": result.unk_pct,
    }
    print(json.dumps(output, indent=2))
    return 0


def run_convert(args: argparse.Namespace) -> int:
    """Execute the convert command — launch Java batch converter.

    Writes three things rather than one: converted card text, a
    ``<name>.provenance.json`` sidecar beside each of them joining every
    rendered line back to the runtime traits that produced it, and the
    converted token scripts under their own tree.

    The JVM's stdout and stderr are deliberately left attached to the
    operator's terminal rather than captured: ``ConvertMain`` prints its
    undefined-SVar report — every chain parameter that named an SVar its
    script never defines, whose segment was left out of ``script_text`` —
    as one block on stderr at the end of the run, and the operator is the
    one who has to read it.

    Returns ``ConvertMain``'s exit status: non-zero when an output tree
    already holds converted files and ``--replace`` was not given, and when
    any script failed to convert.
    """
    import subprocess

    from price_predictor.infrastructure.forge_jvm import (
        build_forge_classpath,
        build_jvm_command,
    )

    try:
        # ConvertMain calls ForgeEnvironmentInitializer.initialize(), which
        # uses GuiBase / GuiHeadless / FModel — all in forge-gui (and FModel
        # pulls in forge-ai subsystems), so the full runtime is required.
        classpath = build_forge_classpath(
            include_full_runtime=True,
            include_dependency_glob=True,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    cmd = build_jvm_command(
        main_class="com.pricepredictor.connector.ConvertMain",
        classpath=classpath,
        main_args=[
            "--cards-path", args.cards_path,
            "--output-path", args.output_path,
            "--tokens-path", args.tokens_path,
            "--tokens-output-path", args.tokens_output_path,
            *(["--replace"] if args.replace else []),
        ],
    )

    try:
        code = _converted(args, lambda: subprocess.run(cmd, check=False).returncode)
    except FileNotFoundError:
        logger.error("Java not found. Ensure java is on PATH.")
        return 2
    return code


#: ``ConvertMain``'s exit when it refused before touching either tree
#: (``ExitCode.BAD_CONFIG``: a flag, an unsafe or occupied output tree).
_UNTOUCHED_EXIT = 2


def _converted(args: argparse.Namespace, convert: Callable[[], int]) -> int:
    """Run ``convert`` and keep, in this invocation's ``run.json`` entry, the
    conversion it left (``converted_provenance``): the Forge checkout's commit
    as it stood at the start and the sidecars' digest once it finished whole,
    ``null`` when it may have changed the trees without finishing (a failed
    script, a crash, Ctrl-C), nothing when it refused before touching them."""
    from price_predictor.infrastructure.converted_provenance import forge_checkout
    from price_predictor.infrastructure.forge_jvm import forge_dir

    commit, dirty = forge_checkout(forge_dir())
    try:
        code = convert()
    except KeyboardInterrupt:
        _record_conversion(args, None)
        raise
    if code == 0:
        _record_conversion(args, _conversion_record(args, commit, dirty))
    elif code != _UNTOUCHED_EXIT:
        _record_conversion(args, None)
    return code


def _conversion_record(args: argparse.Namespace, commit: str | None, dirty: bool) -> dict:
    from price_predictor.infrastructure.converted_provenance import (
        CARD_TREE,
        TOKEN_TREE,
        sidecar_digest,
    )

    roots = {CARD_TREE: Path(args.output_path), TOKEN_TREE: Path(args.tokens_output_path)}
    return {"forge_commit": commit, "forge_dirty": dirty, "sidecar_digest": sidecar_digest(roots)}


def _record_conversion(args: argparse.Namespace, record: dict | None) -> None:
    from price_predictor.domain.converted_provenance import CONVERSION_MEMBER
    from price_predictor.infrastructure.run_manifest import annotate_run

    annotate_run(_converted_root(args), **{CONVERSION_MEMBER: record})


def _check_report(args: argparse.Namespace) -> Path:
    """``--report``, else :data:`CHECK_REPORT_NAME` in ``--output-path``."""
    if args.report:
        return Path(args.report)
    return Path(args.output_path) / CHECK_REPORT_NAME


def run_check_convert(args: argparse.Namespace) -> int:
    """Execute the check-convert command."""
    from price_predictor.application.check_convert import check_all, format_report

    output_dir = Path(args.output_path)
    cards_dir = Path(args.cards_path)

    if not require_path_exists(output_dir, "Output directory"):
        return 1
    if not require_path_exists(cards_dir, "Cards directory"):
        return 1

    logger.info("Checking converted files in %s against %s...", output_dir, cards_dir)

    results = check_all(output_dir, cards_dir, threshold=args.threshold)

    report_path = _check_report(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_report(results), encoding="utf-8")
    logger.info("Report written to %s (%d cards with issues)", report_path, len(results))
    print(format_report(results, limit=args.limit))
    return 0


def _converted_root(args: argparse.Namespace) -> Path:
    """``convert``'s run directory: the one holding both converted trees
    (``converted/`` beside ``cardsfolder/`` and ``tokenscripts/``), else the
    card tree's own when the two trees are not siblings."""
    cards, tokens = Path(args.output_path), Path(args.tokens_output_path)
    return cards.parent if cards.parent.resolve() == tokens.parent.resolve() else cards
