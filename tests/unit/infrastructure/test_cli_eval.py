"""Tests verifying the 'eval' subcommand was removed (FR-010)."""

from __future__ import annotations

import pytest


class TestEvalSubcommandRemoved:
    """The 'eval' subcommand was removed in feature 008 (FR-010)."""

    def test_eval_subcommand_no_longer_registered(self) -> None:
        from price_predictor.infrastructure.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["eval", "card.txt"])
