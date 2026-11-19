"""What the collectors count of a run's directed forks, and the probabilities
that follow (D8, task DF01b2).

The records are the played gone-planeswalker forks of ``DirectedForkPlayedForwardTest``
(``directed-702.19ef-combats.jsonl``), re-gamed; the sides come from the real
catalogue under ``rules/``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from effects.domain.directed_fork_probabilities import directed_fork_probability
from effects.domain.directed_sides import (
    DirectedForkGames,
    directed_rule_probabilities,
    forkable_sides,
)
from effects.domain.rule_catalogue import Polarity
from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.rules_catalogue_io import load_registered_variants, load_scenarios

_PLAYED = Path(__file__).parents[3] / "fixtures" / "effects" / "directed-702.19ef-combats.jsonl"
RULES = Path(__file__).parents[4] / "rules"


def fork_record(rule: str, game: str):
    """The fixture's real directed record of ``rule``, moved into ``game``."""
    for line in _PLAYED.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        if data.get("aimed_at") == [rule]:
            return replace(record_from_dict(data), game_id=game)
    raise AssertionError(f"no directed record of {rule} in {_PLAYED}")


def live_record(game: str):
    """The fixture's live (not forked) combat record, moved into ``game``."""
    for line in _PLAYED.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        if not data.get("fork"):
            return replace(record_from_dict(data), game_id=game)
    raise AssertionError("no live record")


@pytest.fixture(scope="module")
def catalogue():
    return load_scenarios(RULES)


@pytest.fixture(scope="module")
def registered():
    """The real registry's pairs, and 702.19e's plain_trample: the fixture's
    played rule, in the catalogue but not in the Java registry."""
    return {**load_registered_variants(RULES), "702.19e": frozenset({"plain_trample"})}


class TestForkableSides:
    def test_a_rule_s_forkable_variants_are_the_registered_classified_ones(
        self, catalogue, registered,
    ):
        sides = forkable_sides(catalogue, ("702.19e", "702.19f"), registered)
        assert sides == {
            "702.19e": {"plain_trample": Polarity.DOES_NOT_FIRE},
            "702.19f": {"trample_over_planeswalkers": Polarity.DOES_NOT_FIRE},
        }

    def test_a_rule_the_catalogue_lacks_has_no_variant(self, catalogue, registered):
        assert forkable_sides(catalogue, ("999.99z",), registered) == {"999.99z": {}}

    def test_a_rule_counts_only_the_side_its_registered_variant_makes(self, catalogue, registered):
        """702.19c's catalogue lists plain_trample (does_not_fire) and
        thrasta_in_the_dreadmaws_place (fires); only the second is registered."""
        sides = forkable_sides(catalogue, ("702.19c",), registered)
        assert sides == {"702.19c": {"thrasta_in_the_dreadmaws_place": Polarity.FIRES}}
        assert forkable_sides(catalogue, ("702.19c",), {}) == {"702.19c": {}}

    def test_the_catalogue_unregistered_side_is_left_out_of_the_counts(self, catalogue):
        both = forkable_sides(catalogue, ("702.19c",),
                              {"702.19c": {"plain_trample", "thrasta_in_the_dreadmaws_place"}})
        assert set(both["702.19c"].values()) == {Polarity.FIRES, Polarity.DOES_NOT_FIRE}

    def test_every_listed_rule_has_a_registered_side(self, catalogue, registered):
        listed = (RULES / "directed-fork-rules.txt").read_text(encoding="utf-8").split()
        sides = forkable_sides(catalogue, listed, registered)
        assert [rule for rule, variants in sides.items() if not variants] == []


class TestDirectedForkGames:
    def test_a_directed_record_credits_its_game_to_its_rule_and_variant(self):
        found = DirectedForkGames()
        found.add(fork_record("702.19e", "g1"))
        assert found.games == {("702.19e", "plain_trample"): {"g1"}}

    def test_a_live_record_credits_nothing(self):
        found = DirectedForkGames()
        found.add(live_record("g1"))
        assert found.games == {}

    def test_a_game_counts_once_however_many_records_it_holds(self):
        found = DirectedForkGames()
        for _ in range(3):
            found.add(fork_record("702.19e", "g1"))
        assert found.count("702.19e", {"plain_trample"}) == 1

    def test_cut_games_are_left_out(self):
        found = DirectedForkGames()
        found.add(fork_record("702.19e", "g1"))
        found.add(fork_record("702.19e", "g2"))
        found.drop_games(frozenset({"g2"}))
        assert found.games == {("702.19e", "plain_trample"): {"g1"}}

    def test_a_game_drops_to_nothing_leaves_no_entry(self):
        found = DirectedForkGames()
        found.add(fork_record("702.19e", "g2"))
        found.drop_games(frozenset({"g2"}))
        assert found.games == {}

    def test_merging_shards_unions_their_games(self):
        first, second = DirectedForkGames(), DirectedForkGames()
        first.add(fork_record("702.19e", "g1"))
        second.add(fork_record("702.19e", "g1"))
        second.add(fork_record("702.19e", "g2"))
        first.merge(second)
        assert first.count("702.19e", {"plain_trample"}) == 2


class TestProbabilities:
    SIDES = {
        "702.19e": {"plain_trample": Polarity.DOES_NOT_FIRE},
        "702.19f": {"trample_over_planeswalkers": Polarity.DOES_NOT_FIRE},
    }

    def forks(self, rule: str, games: int) -> DirectedForkGames:
        found = DirectedForkGames()
        for n in range(games):
            found.add(fork_record(rule, f"g{n}"))
        return found

    def test_a_rule_with_no_fork_yet_is_at_one(self):
        assert directed_rule_probabilities(self.SIDES, DirectedForkGames(), 100) == {
            "702.19e": 1.0, "702.19f": 1.0}

    def test_the_formula_is_the_shared_one_over_the_forked_side(self):
        got = directed_rule_probabilities(self.SIDES, self.forks("702.19e", 40), 100)
        assert got["702.19e"] == directed_fork_probability([40], 100) == pytest.approx(0.6)
        assert got["702.19f"] == 1.0

    def test_a_rule_with_one_registered_side_reaches_zero_when_full(
        self, catalogue, registered,
    ):
        sides = forkable_sides(catalogue, ("702.19c",), registered)
        found = DirectedForkGames()
        found.games = {("702.19c", "thrasta_in_the_dreadmaws_place"): {f"g{n}" for n in range(100)}}
        assert directed_rule_probabilities(sides, DirectedForkGames(), 100) == {"702.19c": 1.0}
        assert directed_rule_probabilities(sides, found, 100) == {"702.19c": 0.0}

    def test_a_rule_at_its_count_is_at_zero(self):
        got = directed_rule_probabilities(self.SIDES, self.forks("702.19e", 100), 100)
        assert got["702.19e"] == 0.0

    def test_a_rule_one_fork_short_keeps_the_floor(self):
        got = directed_rule_probabilities(self.SIDES, self.forks("702.19e", 99), 100)
        assert got["702.19e"] == 0.05

    def test_both_sides_of_a_rule_count_and_the_shorter_decides(self):
        sides = {"r": {"a": Polarity.FIRES, "b": Polarity.DOES_NOT_FIRE}}
        found = DirectedForkGames()
        found.games = {("r", "a"): {f"a{n}" for n in range(100)},
                       ("r", "b"): {f"b{n}" for n in range(25)}}
        assert directed_rule_probabilities(sides, found, 100)["r"] == pytest.approx(0.75)

    def test_a_variant_of_the_other_polarity_counts_for_no_side(self):
        sides = {"r": {"a": Polarity.FIRES, "o": Polarity.OTHER}}
        found = DirectedForkGames()
        found.games = {("r", "o"): {f"o{n}" for n in range(100)}}
        assert directed_rule_probabilities(sides, found, 100)["r"] == 1.0

    def test_a_rule_with_no_side_to_count_is_left_out(self):
        sides = {"r": {"o": Polarity.OTHER}, "s": {}}
        assert directed_rule_probabilities(sides, DirectedForkGames(), 100) == {}
