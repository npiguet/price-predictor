"""Entry point for python -m price_predictor."""

import logging
import sys

from price_predictor.infrastructure.cli import RUN_MANIFESTS, build_parser
from price_predictor.infrastructure.cli_run_manifest import dispatch, tag_subcommands


def _configure_logging() -> None:
    """Configure logging to send INFO messages to stderr."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(message)s",
    )


def main() -> int:
    _configure_logging()
    parser = build_parser()
    tag_subcommands(parser)
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return dispatch(args, RUN_MANIFESTS, module="price_predictor")


if __name__ == "__main__":
    sys.exit(main())
