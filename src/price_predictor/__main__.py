"""Entry point for python -m price_predictor."""

import logging
import sys

from price_predictor.infrastructure.cli import (
    build_parser,
    run_check_convert,
    run_convert,
    run_evaluate_sklearn,
    run_evaluate_transformer_new,
    run_predict_sklearn,
    run_predict_transformer,
    run_serve,
    run_train_sklearn,
    run_train_transformer_new,
    run_vocabulary,
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

    if args.command == "train":
        if getattr(args, "model", None) is None:
            parser.parse_args(["train", "--help"])
            return 1
        _configure_logging()
        if args.model == "sklearn":
            return run_train_sklearn(args)
        elif args.model == "transformer":
            return run_train_transformer_new(args)
    elif args.command == "predict":
        if getattr(args, "model", None) is None:
            parser.parse_args(["predict", "--help"])
            return 1
        if args.model == "sklearn":
            return run_predict_sklearn(args)
        elif args.model == "transformer":
            return run_predict_transformer(args)
    elif args.command == "evaluate":
        if getattr(args, "model", None) is None:
            parser.parse_args(["evaluate", "--help"])
            return 1
        _configure_logging()
        if args.model == "sklearn":
            return run_evaluate_sklearn(args)
        elif args.model == "transformer":
            return run_evaluate_transformer_new(args)
    elif args.command == "serve":
        _configure_logging()
        return run_serve(args)
    elif args.command == "vocabulary":
        return run_vocabulary(args)
    elif args.command == "convert":
        return run_convert(args)
    elif args.command == "check-convert":
        return run_check_convert(args)
    else:
        parser.print_help()
        return 1

    # Should not be reached, but just in case
    return 1


if __name__ == "__main__":
    sys.exit(main())
