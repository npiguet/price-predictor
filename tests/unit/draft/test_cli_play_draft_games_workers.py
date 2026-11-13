"""``draft play-draft-games --workers``: the Forge pool's size, 1 to 12, as
on every other command that starts Forge workers (V4 #9)."""

from __future__ import annotations

import pytest

from draft.infrastructure.cli import build_parser


def parse(*argv: str):
    return build_parser().parse_args(["play-draft-games", *argv])


@pytest.mark.parametrize("workers", ["0", "13", "-2"])
def test_an_out_of_range_worker_count_is_a_usage_error(workers):
    with pytest.raises(SystemExit):
        parse("--workers", workers)


def test_the_default_and_the_edges_are_accepted():
    assert parse().workers == 12
    assert parse("--workers", "1").workers == 1
    assert parse("--workers", "12").workers == 12
