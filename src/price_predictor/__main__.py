"""Entry point for python -m price_predictor."""

import logging
import sys

from price_predictor.infrastructure.cli import (
    build_parser,
    run_check_convert,
    run_convert,
    run_eval,
    run_evaluate,
    run_evaluate_transformer,
    run_predict,
    run_serve,
    run_train,
    run_train_transformer,
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
    elif args.command == "eval":
        return run_eval(args)
    elif args.command == "convert":
        return run_convert(args)
    elif args.command == "train-transformer":
        _configure_logging()
        return run_train_transformer(args)
    elif args.command == "evaluate-transformer":
        _configure_logging()
        return run_evaluate_transformer(args)
    elif args.command == "check-convert":
        return run_check_convert(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
