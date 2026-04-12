"""Entry point for python -m price_predictor."""

import logging
import sys

from price_predictor.infrastructure.cli import build_parser


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
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
