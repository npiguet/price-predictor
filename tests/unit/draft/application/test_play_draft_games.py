"""Unit tests for the play-draft-games supervisor.

The command's own job is counting: how many rows this run added, when to stop,
and what to report. The matches themselves belong to Forge and are not exercised
here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from draft.application.play_draft_games import (
    PlayDraftGamesConfig,
    PlayDraftGamesUseCase,
    RunSummary,
    format_elapsed,
)


class TestElapsedFormatting:
    def test_minutes_and_seconds(self):
        assert format_elapsed(2292) == "38m 12s"

    def test_pads_seconds(self):
        assert format_elapsed(65) == "1m 05s"

    def test_hours_appear_past_an_hour(self):
        assert format_elapsed(3723) == "1h 02m 03s"

    def test_zero(self):
        assert format_elapsed(0) == "0m 00s"


class TestSummary:
    def test_reports_matches_elapsed_and_output(self, tmp_path):
        out = tmp_path / "draft-games.txt"
        text = RunSummary(matches_played=500, elapsed_seconds=2292, output_path=out).format()

        assert "matches played   500" in text
        assert "elapsed          38m 12s" in text
        assert str(out) in text

    def test_has_no_skipped_line(self):
        """Workers draw their own pairings, so nothing can count failures."""
        text = RunSummary(1, 1.0, Path("out.txt")).format()
        wanted = ("matches", "elapsed", "output")
        labels = [
            line.split()[0]
            for line in text.splitlines()
            if line.startswith(wanted)
        ]
        assert labels == list(wanted)


class TestRunIdScope:
    """The seat table must carry exactly what startup validated (US2)."""

    def _records(self):
        from draft.domain.draft_geometry import Booster, DraftRecord, Seat

        def rec(draft_id, run_id):
            return DraftRecord(
                draft_id=draft_id,
                run_id=run_id,
                timestamp="2026-08-10T00:00:00Z",
                seats=[
                    Seat(agent="gen4", deck=["Forest"], deck_score=1.0),
                    Seat(agent="gen1", deck=["Island"], deck_score=1.0),
                ],
                boosters=[Booster(set_code="BLB", picks=["Forest"])],
            )

        return [rec("d1", "runA"), rec("d2", "runB"), rec("d3", "runA")]

    def _scoped(self, tmp_path, run_ids):
        from draft.application.play_draft_games import selected_records

        config = PlayDraftGamesConfig(
            drafts_path=tmp_path / "drafts.jsonl", run_ids=run_ids,
        )
        with patch(
            "draft.application.play_draft_games.read_records",
            return_value=iter(self._records()),
        ):
            return [r.draft_id for r in selected_records(config)]

    def test_no_run_id_selects_everything(self, tmp_path):
        assert self._scoped(tmp_path, ()) == ["d1", "d2", "d3"]

    def test_one_run_id_selects_its_records(self, tmp_path):
        assert self._scoped(tmp_path, ("runA",)) == ["d1", "d3"]

    def test_several_run_ids_are_unioned(self, tmp_path):
        assert self._scoped(tmp_path, ("runA", "runB")) == ["d1", "d2", "d3"]

    def test_unknown_run_id_selects_nothing(self, tmp_path):
        assert self._scoped(tmp_path, ("nope",)) == []

    def test_seat_table_holds_only_scoped_records(self, tmp_path):
        from draft.application.play_draft_games import write_seat_table_file

        destination = tmp_path / "seats.txt"
        config = PlayDraftGamesConfig(
            drafts_path=tmp_path / "drafts.jsonl", run_ids=("runB",),
        )
        with patch(
            "draft.application.play_draft_games.read_records",
            return_value=iter(self._records()),
        ):
            written, _, _ = write_seat_table_file(config, destination)

        pods = {line.split(";")[0] for line in destination.read_text().splitlines()}
        assert written == 2
        assert pods == {"d2"}


class TestStoppingCondition:
    """--n-pairings counts rows this run added, not rows in the file."""

    def _pool_for(self, config, tmp_path):
        use_case = PlayDraftGamesUseCase(connector=MagicMock())
        return use_case._build_pool(config, tmp_path / "seats.txt", "run-1")

    def test_counts_only_rows_added_by_this_run(self, tmp_path):
        out = tmp_path / "draft-games.txt"
        out.write_text("old\nold\nold\n", encoding="utf-8")
        config = PlayDraftGamesConfig(output_path=out, n_pairings=2)

        pool = self._pool_for(config, tmp_path)

        assert pool._should_stop(3) is False   # baseline, nothing added yet
        assert pool._should_stop(4) is False   # one added
        assert pool._should_stop(5) is True    # two added, target met

    def test_no_pre_existing_rows(self, tmp_path):
        out = tmp_path / "draft-games.txt"
        config = PlayDraftGamesConfig(output_path=out, n_pairings=2)

        pool = self._pool_for(config, tmp_path)

        assert pool._should_stop(1) is False
        assert pool._should_stop(2) is True

    def test_unbounded_run_has_no_stop_condition(self, tmp_path):
        config = PlayDraftGamesConfig(output_path=tmp_path / "o.txt", n_pairings=None)

        pool = self._pool_for(config, tmp_path)

        assert pool._should_stop is None

    def test_worker_count_reaches_the_pool(self, tmp_path):
        config = PlayDraftGamesConfig(output_path=tmp_path / "o.txt", workers=4)

        pool = self._pool_for(config, tmp_path)

        assert pool._worker_count == 4


class TestUnboundedRunAndExitCodes:
    """US3: no target means play until interrupted; 130 is a *bare* interrupt."""

    def _exit_code(self, summary):
        from draft.infrastructure.cli import build_parser, run_play_draft_games

        args = build_parser().parse_args(
            ["play-draft-games", "--drafts-path", "corpus.jsonl"]
        )
        from draft.domain.draft_geometry import Booster, DraftRecord, Seat

        record = DraftRecord(
            draft_id="d1",
            run_id="r1",
            timestamp="2026-08-10T00:00:00Z",
            seats=[
                Seat(agent="gen4", deck=["Forest"], deck_score=1.0),
                Seat(agent="gen1", deck=["Island"], deck_score=1.0),
            ],
            boosters=[Booster(set_code="BLB", picks=["Forest"])],
        )
        with patch("pathlib.Path.exists", return_value=True), patch(
            "draft.infrastructure.draft_record_io.read_records", return_value=[record]
        ), patch(
            "draft.application.play_draft_games.PlayDraftGamesUseCase.execute",
            return_value=summary,
        ):
            return run_play_draft_games(args)

    def test_clean_finish_is_zero(self):
        assert self._exit_code(RunSummary(10, 1.0, Path("o.txt"), interrupted=False)) == 0

    def test_interrupt_after_a_match_is_zero(self):
        assert self._exit_code(RunSummary(3, 1.0, Path("o.txt"), interrupted=True)) == 0

    def test_interrupt_before_any_match_is_130(self):
        assert self._exit_code(RunSummary(0, 1.0, Path("o.txt"), interrupted=True)) == 130

    def test_zero_matches_without_interrupt_is_not_130(self):
        """A bounded run that recorded nothing still finished; only a signal is 130."""
        assert self._exit_code(RunSummary(0, 1.0, Path("o.txt"), interrupted=False)) == 0

    def test_absent_target_leaves_the_run_unbounded(self):
        config = PlayDraftGamesConfig(n_pairings=None)
        assert config.n_pairings is None


class TestSeatTableLifetime:
    """The seat table is scratch: created in the temp dir, gone when the run ends."""

    def _run(self, tmp_path, corpus_rows=1, pool_raises=False):
        from draft.domain.draft_geometry import Booster, DraftRecord, Seat

        record = DraftRecord(
            draft_id="d1",
            run_id="r1",
            timestamp="2026-08-10T00:00:00Z",
            seats=[
                Seat(agent="gen4", deck=["Forest"], deck_score=1.0),
                Seat(agent="gen1", deck=["Island"], deck_score=1.0),
            ],
            boosters=[Booster(set_code="BLB", picks=["Forest"])],
        )
        seen: dict[str, Path] = {}

        class FakePool:
            def __init__(self, **kw):
                seen["seats_file"] = kw.get("_seats_file")

            def output_line_count(self):
                return 0

            def run(self):
                if pool_raises:
                    raise KeyboardInterrupt
                seen["existed_during_run"] = seen["path"].exists()

        config = PlayDraftGamesConfig(
            drafts_path=tmp_path / "drafts.jsonl",
            output_path=tmp_path / "out.txt",
        )
        with patch(
            "draft.application.play_draft_games.read_records", return_value=[record]
        ), patch.object(
            PlayDraftGamesUseCase, "_build_pool"
        ) as build_pool:

            def capture(cfg, seats_file, run_id):
                seen["path"] = seats_file
                pool = MagicMock()
                pool.output_line_count.return_value = 0
                pool.run.side_effect = (
                    KeyboardInterrupt if pool_raises
                    else (lambda: seen.update(existed_during_run=seats_file.exists()))
                )
                return pool

            build_pool.side_effect = capture
            use_case = PlayDraftGamesUseCase(connector=MagicMock())
            if pool_raises:
                with pytest.raises(KeyboardInterrupt):
                    use_case.execute(config)
            else:
                use_case.execute(config)
        return seen

    def test_seat_table_exists_while_workers_run(self, tmp_path):
        seen = self._run(tmp_path)
        assert seen["existed_during_run"] is True

    def test_seat_table_is_deleted_afterwards(self, tmp_path):
        seen = self._run(tmp_path)
        assert not seen["path"].exists()

    def test_seat_table_is_deleted_on_interrupt(self, tmp_path):
        seen = self._run(tmp_path, pool_raises=True)
        assert not seen["path"].exists()
