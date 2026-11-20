"""Unit tests for ``train-encoder``'s labels: label arithmetic, the
``cards-win-rates.txt`` writer and the stratified split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sealed.application.encoder_counters import (
    CardCounters,
)
from sealed.application.encoder_labels import CardLabels, _build_label_map
from sealed.application.encoder_split import _split_cards
from sealed.application.encoder_win_rates import _write_win_rates
from sealed.domain.encoder_model import COLOR_ORDER

# Per-color counter lists on CardCounters are indexed by WUBRG position.
_CI: dict[str, int] = {c: i for i, c in enumerate(COLOR_ORDER)}


# ── Label arithmetic ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CounterSpec:
    """The counter values ``_make_counter`` sets; everything else stays zero."""

    wp: int = 0
    lp: int = 0
    wd: int = 0
    ld: int = 0
    wp_play: int = 0
    lp_play: int = 0
    wd_play: int = 0
    ld_play: int = 0
    color_in_deck: dict[str, int] | None = None
    color_played: dict[str, int] | None = None
    color_loss_in_deck: dict[str, int] | None = None
    color_loss_played: dict[str, int] | None = None


def _make_counter(spec: CounterSpec) -> CardCounters:
    c = CardCounters()
    c.wins_when_played = spec.wp
    c.losses_when_played = spec.lp
    c.wins_when_in_deck = spec.wd
    c.losses_when_in_deck = spec.ld
    c.wins_when_played_at_play = spec.wp_play
    c.losses_when_played_at_play = spec.lp_play
    c.wins_when_in_deck_at_play = spec.wd_play
    c.losses_when_in_deck_at_play = spec.ld_play
    for letter_map, slots in (
        (spec.color_in_deck, c.wins_when_in_deck_with),
        (spec.color_played, c.wins_when_played_with),
        (spec.color_loss_in_deck, c.losses_when_in_deck_with),
        (spec.color_loss_played, c.losses_when_played_with),
    ):
        for letter, value in (letter_map or {}).items():
            slots[_CI[letter]] = value
    return c


class TestBuildLabelMap:
    def test_excludes_zero_in_deck(self):
        counters = {"X": _make_counter(CounterSpec(wd=0, ld=0))}
        labels = _build_label_map(counters, shrinkage_k=20.0)
        assert "X" not in labels

    def test_score_play_raw_and_shrunk_with_k_zero_match(self):
        # 4 in-deck @play games: 3 wins (all played), 1 loss (played).
        counters = {"X": _make_counter(CounterSpec(
            wp=3, lp=1, wd=3, ld=1,
            wp_play=3, lp_play=1, wd_play=3, ld_play=1,
        ))}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        # raw_score_play = (3 - 1) / (3 + 1) = 0.5
        assert lbl.raw_score_play == pytest.approx(0.5)
        # k=0 → shrunk == raw.
        assert lbl.shrunk_score_play == pytest.approx(0.5)

    def test_score_play_shrinkage_pulls_toward_zero(self):
        counters = {"X": _make_counter(CounterSpec(
            wp=2, lp=0, wd=2, ld=0,
            wp_play=2, lp_play=0, wd_play=2, ld_play=0,
        ))}
        labels = _build_label_map(counters, shrinkage_k=20.0)
        lbl = labels["X"]
        # raw = 2/2 = 1.0; shrunk = 2 / (2 + 20) ≈ 0.0909
        assert lbl.raw_score_play == pytest.approx(1.0)
        assert lbl.shrunk_score_play == pytest.approx(2.0 / 22.0)

    def test_zero_at_play_denominator_yields_none(self):
        # Card never observed on the play.
        counters = {"X": _make_counter(CounterSpec(
            wp=2, lp=1, wd=2, ld=1,
            wp_play=0, lp_play=0, wd_play=0, ld_play=0,
        ))}
        labels = _build_label_map(counters, shrinkage_k=20.0)
        lbl = labels["X"]
        assert lbl.raw_score_play is None
        assert lbl.shrunk_score_play is None
        # score_draw cell should be populated from the @draw subset.
        assert lbl.raw_score_draw is not None

    def test_played_rate_always_present_when_card_kept(self):
        counters = {"X": _make_counter(CounterSpec(wp=1, lp=0, wd=2, ld=2))}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        # raw_played_rate = 1 / 4 = 0.25
        assert lbl.raw_played_rate == pytest.approx(0.25)
        assert lbl.shrunk_played_rate == pytest.approx(0.25)

    def test_cast_lift_empty_when_never_played(self):
        # Card is in deck 4 times but never played.
        counters = {"X": _make_counter(CounterSpec(wp=0, lp=0, wd=2, ld=2))}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        assert lbl.raw_cast_lift is None

    def test_cast_lift_empty_when_always_played(self):
        # Card is in deck and played every time.
        counters = {"X": _make_counter(CounterSpec(wp=2, lp=2, wd=2, ld=2))}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        assert lbl.raw_cast_lift is None

    def test_color_lift_empty_when_no_observations_with_color(self):
        # Card in deck once with color R and never with W.
        counters = {"X": _make_counter(CounterSpec(
            wp=1, lp=0, wd=1, ld=0,
            color_in_deck={"R": 1}, color_played={"R": 1},
        ))}
        labels = _build_label_map(counters, shrinkage_k=0.0)
        lbl = labels["X"]
        assert lbl.raw_color_lift["R"] is not None
        assert lbl.raw_color_lift["W"] is None


# ── cards-win-rates.txt writer ───────────────────────────────────────


def _label(name: str, **overrides) -> CardLabels:
    base = dict(
        card_name=name,
        counters=CardCounters(),
        raw_score_play=0.5, shrunk_score_play=0.5,
        raw_score_draw=0.4, shrunk_score_draw=0.4,
        raw_played_rate=0.6, shrunk_played_rate=0.6,
        raw_cast_lift=0.1, shrunk_cast_lift=0.1,
        raw_color_lift={c: 0.0 for c in COLOR_ORDER},
        shrunk_color_lift={c: 0.0 for c in COLOR_ORDER},
    )
    base.update(overrides)
    return CardLabels(**base)


class TestWriteWinRates:
    def test_header_matches_fr013a(self, tmp_path: Path):
        labels = {"X": _label("X")}
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == (
            "card_name;wins_when_played;wins_when_in_deck;"
            "losses_when_played;losses_when_in_deck;"
            "raw_score_play;shrunk_score_play;"
            "raw_score_draw;shrunk_score_draw;"
            "raw_played_rate;shrunk_played_rate;"
            "raw_cast_lift;shrunk_cast_lift;"
            "raw_color_lift_W;shrunk_color_lift_W;"
            "raw_color_lift_U;shrunk_color_lift_U;"
            "raw_color_lift_B;shrunk_color_lift_B;"
            "raw_color_lift_R;shrunk_color_lift_R;"
            "raw_color_lift_G;shrunk_color_lift_G"
        )
        # 1 (card_name) + 4 (primary counters) + 9 heads × 2 (raw+shrunk)
        # = 23 columns. (Supporting docs count "24" — typo; spec.md
        # FR-013a's column list adds to 23.)
        assert len(first.split(";")) == 23

    def test_row_has_23_fields(self, tmp_path: Path):
        labels = {"X": _label("X")}
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        rows = path.read_text(encoding="utf-8").splitlines()
        assert len(rows[1].split(";")) == 23

    def test_empty_cells_render_as_empty_string(self, tmp_path: Path):
        no_play = _label(
            "Y",
            raw_score_play=None, shrunk_score_play=None,
            raw_cast_lift=None, shrunk_cast_lift=None,
            raw_color_lift={"W": None, "U": 0.1, "B": None, "R": None, "G": None},
            shrunk_color_lift={"W": None, "U": 0.1, "B": None, "R": None, "G": None},
        )
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates({"Y": no_play}, path)
        row = path.read_text(encoding="utf-8").splitlines()[1].split(";")
        # raw_score_play (idx 5) and shrunk_score_play (idx 6) should be empty.
        assert row[5] == ""
        assert row[6] == ""
        # raw_color_lift_U (idx 15) is non-empty; raw_color_lift_W (idx 13) empty.
        assert row[13] == ""
        assert row[15] == "0.10000"

    def test_sorted_by_shrunk_score_play_desc_with_empties_at_end(
        self, tmp_path: Path,
    ):
        labels = {
            "Top": _label("Top", shrunk_score_play=0.9),
            "Mid": _label("Mid", shrunk_score_play=0.5),
            "Bot": _label("Bot", shrunk_score_play=-0.4),
            "Empty": _label("Empty", shrunk_score_play=None),
        }
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        rows = path.read_text(encoding="utf-8").splitlines()
        assert rows[1].startswith("Top;")
        assert rows[2].startswith("Mid;")
        assert rows[3].startswith("Bot;")
        assert rows[4].startswith("Empty;")


# ── Stratification ───────────────────────────────────────────────────


def _label_map_with_score_play(values: dict[str, float | None]) -> dict[str, CardLabels]:
    labels: dict[str, CardLabels] = {}
    for name, v in values.items():
        labels[name] = _label(
            name,
            shrunk_score_play=v,
            shrunk_score_draw=0.0,
            shrunk_cast_lift=0.0,
            shrunk_color_lift={c: 0.0 for c in COLOR_ORDER},
        )
    return labels


class TestSplitCards:
    def test_disjoint_split(self):
        labels = _label_map_with_score_play(
            {f"c{i}": (i / 100.0) for i in range(50)},
        )
        train, val = _split_cards(labels, val_fraction=0.2, seed=42)
        assert set(train).isdisjoint(set(val))
        assert set(train) | set(val) == set(labels.keys())

    def test_fallback_chain_used_when_score_play_empty(self):
        # Card "a" has empty score_play but non-empty score_draw → its
        # stratification key falls back to score_draw.
        labels = {
            "a": _label("a",
                        shrunk_score_play=None, shrunk_score_draw=0.5,
                        shrunk_cast_lift=None,
                        shrunk_color_lift={c: None for c in COLOR_ORDER}),
            "b": _label("b", shrunk_score_play=0.7),
            "c": _label("c", shrunk_score_play=-0.3),
        }
        train, val = _split_cards(labels, val_fraction=0.2, seed=42)
        # All three cards land somewhere; no exception, set complete.
        assert set(train) | set(val) == {"a", "b", "c"}

    def test_catch_all_stratum_for_fully_degenerate(self):
        # Every signed cell empty → catch-all stratum.
        labels = {
            f"x{i}": _label(
                f"x{i}",
                shrunk_score_play=None, shrunk_score_draw=None,
                shrunk_cast_lift=None,
                shrunk_color_lift={c: None for c in COLOR_ORDER},
            )
            for i in range(10)
        }
        train, val = _split_cards(labels, val_fraction=0.2, seed=42)
        assert set(train).isdisjoint(set(val))
        assert len(val) == 2  # 20% of 10
