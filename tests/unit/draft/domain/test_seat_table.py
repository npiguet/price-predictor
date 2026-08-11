"""Unit tests for the seat-table projection.

The seat table is the only channel between Python and the Forge workers, so its
line format is a cross-language contract.
"""

from __future__ import annotations

import io
import json
import random

import pytest

from draft.domain.draft_geometry import Booster, DraftGeometry, DraftRecord, Seat
from draft.domain.seat_table import (
    FORGE_NATIVE_LABEL,
    KIND_DECK,
    KIND_POOL,
    SeatRow,
    format_seat_row,
    parse_seat_row,
    project,
    record_set_code,
    seat_rows,
    write_seat_table,
)


def _record(
    draft_id="d1",
    run_id="r1",
    agents=("gen4", "forge-full"),
    set_code="BLB",
    decks=None,
):
    decks = decks or [[f"{a}-card{i}" for i in range(3)] for a in agents]
    return DraftRecord(
        draft_id=draft_id,
        run_id=run_id,
        timestamp="2026-08-10T00:00:00Z",
        seats=[
            Seat(agent=a, deck=list(d), deck_score=1.0)
            for a, d in zip(agents, decks)
        ],
        boosters=[Booster(set_code=set_code, picks=["x"]) for _ in range(3)],
    )


class TestProjection:
    def test_one_row_per_seat(self):
        record = _record(agents=("gen4", "gen1", "forge-full"))
        assert len(seat_rows(record)) == 3

    def test_row_carries_pod_set_and_label(self):
        row = seat_rows(_record(agents=("gen4",), set_code="MH3"))[0]
        assert (row.draft_id, row.set_code, row.label) == ("d1", "MH3", "gen4")

    def test_every_seat_is_deck_kind(self):
        assert all(r.kind == KIND_DECK for r in seat_rows(_record()))

    def test_cards_are_the_recorded_deck(self):
        record = _record(agents=("gen4",), decks=[["Llanowar Elves", "Forest"]])
        assert seat_rows(record)[0].cards == ("Llanowar Elves", "Forest")

    def test_set_code_comes_from_the_boosters(self):
        assert record_set_code(_record(set_code="OTJ")) == "OTJ"

    def test_project_streams_all_records(self):
        rows = list(project([_record("d1"), _record("d2")]))
        assert [r.draft_id for r in rows] == ["d1", "d1", "d2", "d2"]


class TestLineFormat:
    def test_field_order(self):
        row = SeatRow("d1", "BLB", "gen4", KIND_DECK, ("A", "B"))
        assert format_seat_row(row) == "d1;BLB;gen4;deck;A|B"

    def test_round_trip(self):
        row = SeatRow("d9", "MH3", "forge-full", KIND_DECK, ("A", "B", "C"))
        assert parse_seat_row(format_seat_row(row)) == row

    @pytest.mark.parametrize(
        "name",
        [
            "Ach! Hans, Run!",          # comma and exclamation
            "Yawgmoth's Will",          # apostrophe
            "Borrowing 100,000 Arrows",  # comma inside a number
            "(Wall of Wood)",           # parentheses
        ],
    )
    def test_awkward_card_names_survive_a_round_trip(self, name):
        row = SeatRow("d1", "BLB", "gen4", KIND_DECK, (name, "Forest"))
        assert parse_seat_row(format_seat_row(row)).cards == (name, "Forest")

    def test_no_field_separator_leaks_into_a_line(self):
        row = seat_rows(_record(agents=("gen4",)))[0]
        assert format_seat_row(row).count(";") == 4


class TestWriteSeatTable:
    def test_writes_one_line_per_seat_and_returns_the_count(self):
        buf = io.StringIO()
        written, _, _ = write_seat_table([_record("d1"), _record("d2")], buf)

        lines = buf.getvalue().splitlines()
        assert written == 4
        assert len(lines) == 4

    def test_pod_seats_are_consecutive(self):
        buf = io.StringIO()
        write_seat_table([_record("d1"), _record("d2")], buf)

        ids = [line.split(";")[0] for line in buf.getvalue().splitlines()]
        assert ids == ["d1", "d1", "d2", "d2"]

    def test_every_line_parses_back(self):
        buf = io.StringIO()
        write_seat_table([_record()], buf)

        for line in buf.getvalue().splitlines():
            assert parse_seat_row(line).kind == KIND_DECK


class TestForgeNativeDiversion:
    """A share of forge-full seats play a Forge-built deck from their pool (US4).

    ``forge-full`` names full-strength Forge *decisions* (against ``forge-r30`` /
    ``forge-r100``, which randomise 30% / 100% of picks); its deck is still built
    by the SA/picker builder. A diverted seat is Forge end to end, hence
    ``forge-native``.
    """

    def _pod(self, agents, packs=3, pack_size=2):
        """A record whose booster picks are reconstructible into per-seat pools."""
        pod_size = len(agents)
        boosters = []
        for k in range(packs * pod_size):
            boosters.append(
                Booster(
                    set_code="BLB",
                    picks=[f"p{k}c{j}" for j in range(pack_size)],
                )
            )
        return DraftRecord(
            draft_id="d1",
            run_id="r1",
            timestamp="2026-08-10T00:00:00Z",
            seats=[
                Seat(agent=a, deck=[f"{a}-deck{i}" for i in range(3)], deck_score=1.0)
                for i, a in enumerate(agents)
            ],
            boosters=boosters,
        )

    def test_fraction_zero_diverts_nothing(self):
        rows = seat_rows(self._pod(("forge-full", "gen4")), native_fraction=0.0)

        assert [r.kind for r in rows] == [KIND_DECK, KIND_DECK]
        assert [r.label for r in rows] == ["forge-full", "gen4"]

    def test_fraction_one_diverts_every_reference_seat(self):
        rows = seat_rows(self._pod(("forge-full", "gen4")), native_fraction=1.0)

        forge_row, gen4_row = rows
        assert (forge_row.label, forge_row.kind) == (FORGE_NATIVE_LABEL, KIND_POOL)
        assert (gen4_row.label, gen4_row.kind) == ("gen4", KIND_DECK)

    def test_only_forge_full_seats_are_eligible(self):
        rows = seat_rows(self._pod(("gen4", "gen1")), native_fraction=1.0)

        assert all(r.kind == KIND_DECK for r in rows)
        assert FORGE_NATIVE_LABEL not in {r.label for r in rows}

    def test_a_diverted_seat_carries_its_pool_not_its_deck(self):
        record = self._pod(("forge-full", "gen4"))
        row = seat_rows(record, native_fraction=1.0)[0]

        geometry = DraftGeometry.from_record(record)
        assert row.cards == tuple(geometry.drafted_pool(record, 0))
        assert row.cards != tuple(record.seats[0].deck)

    def test_pool_is_larger_than_the_deck_it_replaces(self):
        record = self._pod(("forge-full", "gen4"))
        row = seat_rows(record, native_fraction=1.0)[0]
        assert len(row.cards) == 3 * 2  # packs x pack_size

    def test_diverted_share_approaches_the_fraction(self):
        rng = random.Random(11)
        record = self._pod(("forge-full",) * 8)

        diverted = 0
        trials = 400
        for _ in range(trials):
            rows = seat_rows(record, native_fraction=0.5, rng=rng)
            diverted += sum(1 for r in rows if r.label == FORGE_NATIVE_LABEL)

        share = diverted / (trials * 8)
        assert 0.45 < share < 0.55

    def test_write_seat_table_reports_reference_and_diverted_counts(self):
        buf = io.StringIO()
        record = self._pod(("forge-full", "forge-full", "gen4"))

        seats, reference, diverted = write_seat_table(
            [record], buf, native_fraction=1.0,
        )

        assert (seats, reference, diverted) == (3, 2, 2)

    def test_counts_are_zero_diverted_at_fraction_zero(self):
        buf = io.StringIO()
        record = self._pod(("forge-full", "gen4"))

        seats, reference, diverted = write_seat_table([record], buf)

        assert (seats, reference, diverted) == (2, 1, 0)

    def test_a_diverted_line_round_trips(self):
        buf = io.StringIO()
        write_seat_table([self._pod(("forge-full",) * 2)], buf, native_fraction=1.0)

        for line in buf.getvalue().splitlines():
            row = parse_seat_row(line)
            assert (row.label, row.kind) == (FORGE_NATIVE_LABEL, KIND_POOL)


class TestTrailingPartialLine:
    """FR-016: a corpus cut off mid-write projects its complete records only."""

    def test_partial_final_record_is_ignored(self, tmp_path):
        from draft.infrastructure.draft_record_io import read_records, record_to_dict

        corpus = tmp_path / "drafts.jsonl"
        complete = json.dumps(record_to_dict(_record("d1")))
        partial = json.dumps(record_to_dict(_record("d2")))[:40]
        corpus.write_text(f"{complete}\n{partial}", encoding="utf-8")

        rows = list(project(read_records(corpus)))

        assert {r.draft_id for r in rows} == {"d1"}
