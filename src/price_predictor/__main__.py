"""Entry point for python -m price_predictor."""

import sys

from price_predictor.infrastructure.cli import (
    build_parser,
    run_evaluate,
    run_predict,
    run_train,
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
        return run_train(args)
    elif args.command == "evaluate":
        return run_evaluate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
