"""Command lines for the effects CLI, with the flags a command requires given placeholders.

Most commands have no default for the paths they read and write
(artifact-layout rule 2), so a test of some other flag would otherwise
open with the same four path flags. :func:`cli_args` appends each required
flag the arguments leave out; a test of a required flag itself builds its
arguments by hand.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from effects.infrastructure.cli import build_parser

#: Each command's required flags and a placeholder for each; a collector's
#: corpus source is handled in :func:`cli_args`.
REQUIRED_FLAGS: dict[str, dict[str, str]] = {
    "build-vocab": {"--vocab-path": "vocab.txt", "--keyword-definitions": "keywords.json"},
    "extract-keyword-definitions": {"--output": "keywords.json"},
    "collect-variants": {"--variant-scripts": "variants"},
    "build-corpus": {
        "--records-dir": "records", "--output": "corpus", "--vocab-path": "vocab.txt",
        "--keyword-definitions": "keywords.json",
    },
    "rule-counts": {"--records-dir": "records", "--output": "counts.json", "--variant-scripts": ""},
    "field-coverage": {"--effect-records": "records"},
    "validate-corpus": {"--effect-records": "records", "--variant-scripts": ""},
    "train-effect-model": {
        "--vocab-path": "vocab.txt", "--keyword-definitions": "keywords.json",
        "--reports-dir": "reports", "--model-output": "model",
    },
    "encode-abilities": {"--output-root": "cache", "--checkpoint": "latest.pt"},
    "evaluate-effect-model": {
        "--checkpoint": "latest.pt", "--abilities-root": "cache",
        "--win-rates": "rates.txt", "--reports-dir": "reports",
    },
}
_COLLECTORS = ("collect-coverage", "collect-variants")


def _needs_records(command: str, argv: list[str]) -> bool:
    """A collector names its records unless ``--training-corpus`` names the corpus."""
    return command in _COLLECTORS and "--training-corpus" not in argv


def cli_args(argv: Sequence[str]) -> list[str]:
    """``argv`` with each flag its command requires, and ``argv`` leaves out, appended."""
    argv = list(argv)
    command = argv[0] if argv else ""
    required = dict(REQUIRED_FLAGS.get(command, {}))
    if _needs_records(command, argv):
        required["--effect-records"] = "records"
    for flag, value in required.items():
        if flag not in argv:
            argv += [flag, value]
    return argv


def parse_cli(argv: Sequence[str]) -> argparse.Namespace:
    """The effects parser's reading of ``argv``, required flags filled in."""
    return build_parser().parse_args(cli_args(argv))
