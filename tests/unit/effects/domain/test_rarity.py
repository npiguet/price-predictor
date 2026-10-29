"""``rarity_bucket`` at every edge (T012), and where a text's games come from."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from effects.domain.rarity import RARITY_BUCKETS, TextGames, effective_games, rarity_bucket


@pytest.mark.parametrize(
    ("games", "bucket"),
    [
        (0, "1"), (1, "1"),
        (2, "2-4"), (4, "2-4"),
        (5, "5-19"), (19, "5-19"),
        (20, "20+"), (10_000, "20+"),
    ],
)
def test_every_edge_lands_in_its_bucket(games, bucket):
    assert rarity_bucket(games) == bucket


def test_the_buckets_are_in_ascending_order():
    assert RARITY_BUCKETS == ("1", "2-4", "5-19", "20+")


def test_a_fractional_effective_count_rounds_down_into_its_bucket():
    assert rarity_bucket(4.9) == "2-4"


def _records(*pairs):
    """``(game_id, text)`` pairs as records carrying both."""
    return [SimpleNamespace(game_id=game, text=text) for game, text in pairs]


def _text(record):
    return record.text


def test_effective_games_counts_distinct_games_not_records():
    """One long game producing many records is still one game."""
    records = _records(*[("g1", "bolt")] * 50, ("g2", "bolt"))
    assert effective_games(records, _text) == {"bolt": 2}


def test_a_record_with_no_text_is_not_counted():
    assert effective_games(_records(("g1", None)), _text) == {}


class TestTextGames:
    def test_the_table_is_preferred_to_the_records_own_count(self):
        games = TextGames.of(_records(("g1", "a"), ("g2", "b")), _text, {"a": 40})
        assert games.games("a") == 40
        assert games.bucket("a") == "20+"

    def test_a_text_the_table_does_not_name_falls_back_to_the_records(self):
        """Not to zero, which would weigh it heaviest of all."""
        games = TextGames.of(_records(("g1", "b"), ("g2", "b")), _text, {"a": 40})
        assert games.games("b") == 2

    def test_no_table_and_an_empty_table_count_alike(self):
        records = _records(("g1", "a"), ("g2", "a"), ("g3", "a"))
        assert TextGames.of(records, _text).games("a") == 3
        assert TextGames.of(records, _text, {}).games("a") == 3


class TestCoverage:
    """How much of a shard the rarity table actually names.

    Reported once per run because a table that matches nothing is invisible
    otherwise: every weight silently falls back to the resident shard's count,
    and the run looks exactly like a healthy one.
    """

    def test_it_counts_distinct_texts_rather_than_records(self):
        records = _records(*[("g1", "a")] * 3, ("g1", "b"), ("g1", None), ("g2", None))
        assert TextGames.of(records, _text, {"a": 3}).coverage() == (1, 2)

    def test_a_table_naming_nothing_reads_as_zero_of_the_texts_present(self):
        records = _records(("g1", "a"), ("g1", "b"))
        assert TextGames.of(records, _text, {"other": 1}).coverage() == (0, 2)
        assert TextGames.of(records, _text, {}).coverage() == (0, 2)

    def test_a_shard_with_no_acting_text_at_all_looks_up_nothing(self):
        records = _records(("g1", None), ("g2", None))
        assert TextGames.of(records, _text, {"a": 1}).coverage() == (0, 0)

    def test_without_a_table_there_is_nothing_to_report(self):
        assert TextGames.of(_records(("g1", "a")), _text).coverage() is None
