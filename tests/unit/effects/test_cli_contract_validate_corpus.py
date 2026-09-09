"""`validate-corpus`'s thresholds, and the flag every one of them needs.

A threshold with no flag is a threshold nobody can move, and a number nobody
can move is one an operator either accepts or ignores — there is no third
option at 3 a.m. with eight hours of collection already spent. So the contract
here is not "these flags exist" but "the flag set and the threshold set are the
same set", which is the shape of the defect this whole pass keeps finding: a
value one side reads and the other side cannot write.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from effects.application.validate_corpus import Thresholds
from effects.infrastructure.cli import build_parser


def parse(*argv: str):
    return build_parser().parse_args(["validate-corpus", *argv])


class TestEveryThresholdHasAFlag:
    def test_the_two_sets_are_the_same_set(self):
        args = parse()
        for spec in fields(Thresholds):
            assert hasattr(args, spec.name), spec.name

    def test_the_parsed_defaults_rebuild_the_default_thresholds(self):
        """A flag whose default drifts from the dataclass's makes two runs
        that measured different things and report the same words."""
        args = parse()
        assert Thresholds(**{
            spec.name: getattr(args, spec.name) for spec in fields(Thresholds)
        }) == Thresholds()


class TestTheWatchedRatesTakeAFloor:
    """Passing a floor turns a measurement into a verdict and changes nothing
    else about what is measured."""

    @pytest.mark.parametrize(
        "flag,name",
        [
            ("--min-zone-change-from-zone-rate", "min_zone_change_from_zone_rate"),
            ("--min-attributed-rate", "min_attributed_rate"),
            ("--min-fork-attributed-rate", "min_fork_attributed_rate"),
            ("--min-cause-rate", "min_cause_rate"),
        ],
    )
    def test_unset_it_watches_and_set_it_judges(self, flag, name):
        assert getattr(parse(), name) is None
        assert getattr(parse(flag, "0.8"), name) == 0.8


class TestTheMirrorTurnCheckIsJudgedAtZero:
    """Unlike the other new checks: a fork is taken *at* the moment it
    mirrors, so there is no rate at which the two may disagree about which
    turn it is. The flag exists to widen it for a corpus under repair, not
    because zero is a guess."""

    def test_it_defaults_to_zero_rather_than_to_unset(self):
        assert parse().max_mirror_turn_disagreement_rate == 0.0

    def test_it_can_be_widened(self):
        assert parse(
            "--max-mirror-turn-disagreement-rate", "0.05",
        ).max_mirror_turn_disagreement_rate == 0.05
