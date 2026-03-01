"""Entry point for python -m price_predictor."""

import logging
import sys

from price_predictor.infrastructure.cli import (
    build_parser,
    run_evaluate,
    run_predict,
    run_serve,
    run_train,
)


def _configure_logging() -> None:
    """Configure logging to send INFO messages to stderr."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(message)s",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "predict":
        return run_predict(args)
    elif args.command == "train":
        _configure_logging()
        return run_train(args)
    elif args.command == "evaluate":
        _configure_logging()
        return run_evaluate(args)
    elif args.command == "serve":
        _configure_logging()
        return run_serve(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
