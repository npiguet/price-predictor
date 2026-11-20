"""Unit tests for ``train-encoder``'s counters: aggregation, color extraction
and missing-card handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sealed.application.encoder_counters import (
    CardCounters,
    _aggregate,
    _colors_from_mana_cost,
    _drop_missing_cards,
)
from sealed.domain.encoder_model import COLOR_ORDER
from sealed.domain.match import Side
from sealed.infrastructure.cards_played_reader import CardsPlayedRow
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

# Per-color counter lists on CardCounters are indexed by WUBRG position.
_CI: dict[str, int] = {c: i for i, c in enumerate(COLOR_ORDER)}


@dataclass(frozen=True)
class _Game:
    """One game's card lists, who won and who played first."""

    cp_a: list[str]
    cp_b: list[str]
    cnp_a: list[str]
    cnp_b: list[str]
    winner: str = "A"
    starter: str = "A"


def _row(game: _Game) -> CardsPlayedRow:
    return CardsPlayedRow(
        timestamp="2026-05-03T14:22:01Z",
        run_id="run",
        set_code="BLB",
        method_a="forge-best",
        method_b="forge-3sub",
        cards_played_a=game.cp_a,
        cards_played_b=game.cp_b,
        cards_not_played_a=game.cnp_a,
        cards_not_played_b=game.cnp_b,
        winner=Side(game.winner),
        starter=Side(game.starter),
    )


def _seed_card(folder: Path, card_name: str, mana_cost: str | None) -> None:
    sanitized = card_name.lower().replace(" ", "_").replace(",", "")
    letter = sanitized[0]
    (folder / letter).mkdir(parents=True, exist_ok=True)
    cost_line = f"mana cost: {mana_cost}\n" if mana_cost is not None else ""
    (folder / letter / f"{sanitized}.txt").write_text(
        f"name: {card_name}\n{cost_line}types: instant\n",
        encoding="utf-8",
    )


# ── Aggregation (primary + @play + per-color, single pass) ────────────


class TestAggregate:
    def test_winning_and_losing_sides_both_counted(self, tmp_path: Path):
        loc = ConvertedCardLocator(tmp_path)
        rows = [
            _row(_Game(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[],
                 winner="A", starter="A")),
            _row(_Game(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[],
                 winner="B", starter="A")),
        ]
        counters = _aggregate(rows, loc)
        # LB: A side, won game 1 / lost game 2
        assert counters["LB"].wins_when_played == 1
        assert counters["LB"].losses_when_played == 1
        assert counters["LB"].wins_when_in_deck == 1
        assert counters["LB"].losses_when_in_deck == 1
        # GB: B side, lost game 1 / won game 2
        assert counters["GB"].wins_when_played == 1
        assert counters["GB"].losses_when_played == 1

    def test_in_deck_includes_not_played(self, tmp_path: Path):
        rows = [
            _row(_Game(cp_a=["LB"], cp_b=[], cnp_a=["GB"], cnp_b=[],
                 winner="A", starter="A")),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        assert counters["LB"].wins_when_played == 1
        assert counters["LB"].wins_when_in_deck == 1
        assert counters["GB"].wins_when_played == 0
        assert counters["GB"].wins_when_in_deck == 1

    def test_starter_drives_at_play_subset(self, tmp_path: Path):
        # Card on side A is starter (at play); same card later on side A
        # but starter is B (at draw).
        rows = [
            _row(_Game(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="A")),
            _row(_Game(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="B")),
        ]
        c = _aggregate(rows, ConvertedCardLocator(tmp_path))["LB"]
        assert c.wins_when_played_at_play == 1   # only the first row
        assert c.wins_when_in_deck_at_play == 1
        # On-draw counterparts: derivable by subtraction (FR-010a).
        assert c.wins_when_played - c.wins_when_played_at_play == 1
        assert c.wins_when_in_deck - c.wins_when_in_deck_at_play == 1

    def test_multi_game_match_accumulates(self, tmp_path: Path):
        rows = [
            _row(_Game(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="A")),
            _row(_Game(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="A")),
            _row(_Game(cp_a=["LB"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="B", starter="A")),
        ]
        c = _aggregate(rows, ConvertedCardLocator(tmp_path))["LB"]
        assert c.wins_when_played == 2
        assert c.losses_when_played == 1
        assert c.wins_when_in_deck == 2
        assert c.losses_when_in_deck == 1

    def test_per_color_counters_built_from_deck_color_set(self, tmp_path: Path):
        # Side A deck: LB ({R}) + GB ({G}). Deck colors = {R, G}.
        # Both LB and GB get +1 wins_when_in_deck_with_{R,G}; LB is
        # "played", GB is "not played".
        _seed_card(tmp_path, "LB", "{R}")
        _seed_card(tmp_path, "GB", "{G}")
        rows = [
            _row(_Game(cp_a=["LB"], cp_b=[], cnp_a=["GB"], cnp_b=[],
                 winner="A", starter="A")),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        c_lb = counters["LB"]
        assert c_lb.wins_when_played_with[_CI["R"]] == 1
        assert c_lb.wins_when_played_with[_CI["G"]] == 1
        assert c_lb.wins_when_in_deck_with[_CI["R"]] == 1
        assert c_lb.wins_when_in_deck_with[_CI["G"]] == 1
        assert c_lb.wins_when_played_with[_CI["W"]] == 0
        c_gb = counters["GB"]
        assert c_gb.wins_when_in_deck_with[_CI["R"]] == 1
        assert c_gb.wins_when_in_deck_with[_CI["G"]] == 1
        assert c_gb.wins_when_played_with[_CI["R"]] == 0  # GB was not played

    def test_losing_side_per_color_counters_increment(self, tmp_path: Path):
        _seed_card(tmp_path, "LB", "{R}")
        _seed_card(tmp_path, "GB", "{G}")
        rows = [
            _row(_Game(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[],
                 winner="A", starter="A")),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        # GB (loser): losses_when_played_with_G should be 1.
        assert counters["GB"].losses_when_played_with[_CI["G"]] == 1
        assert counters["GB"].losses_when_in_deck_with[_CI["G"]] == 1

    def test_card_with_no_mana_cost_contributes_no_color(self, tmp_path: Path):
        # Land-like card: no "mana cost:" line; deck-color set ignores it.
        _seed_card(tmp_path, "Plains-ish", None)
        _seed_card(tmp_path, "LB", "{R}")
        rows = [
            _row(_Game(cp_a=["LB"], cp_b=[], cnp_a=["Plains-ish"], cnp_b=[],
                 winner="A", starter="A")),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        # Plains-ish is "not played" but still in deck running color R.
        assert counters["Plains-ish"].wins_when_in_deck_with[_CI["R"]] == 1
        # It contributed no color of its own.
        assert counters["Plains-ish"].wins_when_in_deck_with[_CI["W"]] == 0

    def test_card_missing_from_corpus_is_still_counted(self, tmp_path: Path):
        # A card with no .txt: still gets primary/@play counters (it's
        # dropped later by _drop_missing_cards), and an empty color set.
        rows = [
            _row(_Game(cp_a=["Mystery Card"], cp_b=[], cnp_a=[], cnp_b=[],
                 winner="A", starter="A")),
        ]
        counters = _aggregate(rows, ConvertedCardLocator(tmp_path))
        c = counters["Mystery Card"]
        assert c.wins_when_played == 1
        assert all(v == 0 for v in c.wins_when_in_deck_with)


# ── Color extraction ─────────────────────────────────────────────────


class TestColorsFromManaCost:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("{R}", {"R"}),
            ("{1}{G}", {"G"}),
            ("{2}{U}{U}", {"U"}),
            ("{W/U}", {"W", "U"}),         # hybrid
            ("{W/P}", {"W"}),               # Phyrexian
            ("{2/W}", {"W"}),               # mono-hybrid
            ("{2}", set()),
            ("{C}", set()),
            ("{X}", set()),
            ("{2}{R}{G}", {"R", "G"}),
            ("", set()),
            (None, set()),
        ],
    )
    def test_dispatches_correctly(self, line, expected):
        assert _colors_from_mana_cost(line) == expected


# ── Missing-card handling (report-but-don't-block) ───────────────────


class TestDropMissingCards:
    def test_noop_when_all_cards_present(self, tmp_path: Path):
        for name in ("Lightning Bolt", "Grizzly Bears"):
            _seed_card(tmp_path, name, "{R}")
        counters = {n: CardCounters() for n in ("Lightning Bolt", "Grizzly Bears")}
        dropped = _drop_missing_cards(counters, ConvertedCardLocator(tmp_path))
        assert dropped == 0
        assert set(counters) == {"Lightning Bolt", "Grizzly Bears"}

    def test_drops_missing_cards_and_reports(self, tmp_path: Path, capsys):
        _seed_card(tmp_path, "Lightning Bolt", "{R}")
        counters = {
            n: CardCounters()
            for n in ("Lightning Bolt", "Counterspell", "Llanowar Elves")
        }
        dropped = _drop_missing_cards(counters, ConvertedCardLocator(tmp_path))
        assert dropped == 2
        # Missing cards are removed; the resolvable one survives.
        assert set(counters) == {"Lightning Bolt"}
        # ... and they're reported on stdout (not raised).
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "2 card(s)" in out
        assert "Counterspell" in out
        assert "Llanowar Elves" in out
        assert "python -m price_predictor convert" in out

    def test_caps_displayed_card_names_at_20(self, tmp_path: Path, capsys):
        counters = {f"Card {i}": CardCounters() for i in range(30)}
        dropped = _drop_missing_cards(counters, ConvertedCardLocator(tmp_path))
        assert dropped == 30
        assert counters == {}
        assert "and 10 more" in capsys.readouterr().out
