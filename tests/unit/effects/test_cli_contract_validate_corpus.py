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

from effects.application.corpus_validation.findings import Thresholds
from tests.unit.effects.cli_args import parse_cli


def parse(*argv: str):
    return parse_cli(["validate-corpus", *argv])


class TestEveryThresholdHasAFlag:
    def test_the_parsed_defaults_rebuild_the_default_thresholds(self):
        """A flag whose default drifts from the dataclass's makes two runs
        that measured different things and report the same words."""
        args = parse()
        assert Thresholds(**{
            spec.name: getattr(args, spec.name) for spec in fields(Thresholds)
        }) == Thresholds()


class TestTheMirrorTurnCheckIsJudgedAtZero:
    """Unlike the other new checks: a fork is taken *at* the moment it
    mirrors, so there is no rate at which the two may disagree about which
    turn it is. The flag exists to widen it for a corpus under repair, not
    because zero is a guess."""

    def test_it_defaults_to_zero_rather_than_to_unset(self):
        assert parse().max_mirror_turn_disagreement_rate == 0.0

