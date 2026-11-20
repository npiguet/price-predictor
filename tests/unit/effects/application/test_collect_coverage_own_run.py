"""The coverage collector (T102): the run's own decks file, the earlier
runs it counts from, and what the run itself played and wrote."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from effects.application.collect_coverage import CollectCoverageConfig
from tests.unit.effects.application.collect_coverage_support import _two_cards
from tests.unit.effects.builders import (
    make_entity,
    write_plain_shard,
)
from tests.unit.effects.domain.test_coverage import _record


class TestTheDeckFileIsTheRunsOwn:
    """Two runs on one records directory replaced each other's fixed-name
    deck file, which every worker reloads at its start."""

    def test_the_decks_file_names_the_run(self, tmp_path, fake_supervisor):
        from effects.application import collect_coverage

        cards = tmp_path / "cardsfolder"
        cards.mkdir()
        (cards / "aa.txt").write_text(
            "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        with patch.object(collect_coverage, "consult_castability", return_value={}):
            collect_coverage.run(CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                target_records=1, decks_per_round=1, no_progress_rounds=1, workers=1,
            ))

        # The path itself is the supervisor's (test_collector_connector).
        fake_supervisor.decks_path.assert_called_with("coverage")
        decks_file = fake_supervisor.play_round.call_args.args[1]
        assert decks_file is fake_supervisor.decks_path.return_value


class TestCountFrom:
    """V4 #5: coverage was counted only under the directory written to, so a
    run in its own run directory (the artifact layout) started at zero and
    re-collected what earlier runs already hold."""

    def _shard_crediting(self, directory, name, records):
        directory.mkdir(parents=True, exist_ok=True)
        record = _record([make_entity("E1", name)], source="E1")
        write_plain_shard(directory, "earlier", [record] * records)

    def test_an_earlier_run_s_records_count_and_its_directory_is_untouched(
        self, tmp_path, fake_supervisor,
    ):
        from effects.application import collect_coverage

        cards = _two_cards(tmp_path, "aa", "bb")
        earlier = tmp_path / "earlier-run"
        self._shard_crediting(earlier, "aa", 5)
        before = sorted(earlier.rglob("*"))
        written = []
        fake_supervisor.play_round.side_effect = (
            lambda decks, *_a, **_k: written.extend(name for deck in decks for name in deck)
        )

        with patch.object(collect_coverage, "consult_castability", return_value={}):
            collect_coverage.run(CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                count_from=(earlier,), target_records=5, decks_per_round=2,
                no_progress_rounds=1, workers=1,
            ))

        assert written, "no round was played"
        assert "aa" not in written, "a card satisfied by --count-from was decked again"
        assert "bb" in written
        assert sorted(earlier.rglob("*")) == before

    @pytest.mark.parametrize("where", ["same", "inside", "holding"])
    def test_an_overlapping_directory_is_refused(self, tmp_path, where):
        records = tmp_path / "records"
        (records / "sub").mkdir(parents=True)
        other = {"same": records, "inside": records / "sub", "holding": tmp_path}[where]

        with pytest.raises(ValueError, match="overlaps"):
            CollectCoverageConfig(effect_records=records, count_from=(other,))

    def test_a_missing_directory_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no such directory"):
            CollectCoverageConfig(
                effect_records=tmp_path / "records", count_from=(tmp_path / "absent",),
            )


class TestWhatTheRunItselfPlayedAndWrote:
    """The zero-record check counts this run's own shards, and a run that
    could deck nothing has played no round."""

    def _config(self, tmp_path):
        cards = _two_cards(tmp_path, "aa", "bb")
        return CollectCoverageConfig(
            effect_records=tmp_path / "records", cards_folders=(cards,),
            target_records=1, decks_per_round=2, no_progress_rounds=1, workers=1,
        )

    def test_another_run_s_shard_is_no_record_of_this_run(
        self, tmp_path, fake_supervisor, caplog,
    ):
        from effects.application import collect_coverage

        fake_supervisor.run_id = "this-run"
        played = []

        def other_run_writes(*_args, **_kwargs):
            played.append(1)
            record = _record([make_entity("E9", "zz")], source="E9")
            write_plain_shard(
                tmp_path / "records", f"r{len(played)}", [record], run_id="other-run",
            )

        fake_supervisor.play_round.side_effect = other_run_writes
        with patch.object(collect_coverage, "consult_castability", return_value={}):
            assert collect_coverage.run(self._config(tmp_path)) == 1
        assert played
        assert "no new effect record" in caplog.text

    def test_this_run_s_own_shard_counts(self, tmp_path, fake_supervisor):
        from effects.application import collect_coverage

        fake_supervisor.run_id = "this-run"

        def own_run_writes(*_args, **_kwargs):
            record = _record([make_entity("E1", "aa")], source="E1")
            write_plain_shard(tmp_path / "records", "a", [record], run_id="this-run")

        fake_supervisor.play_round.side_effect = own_run_writes
        with patch.object(collect_coverage, "consult_castability", return_value={}):
            assert collect_coverage.run(self._config(tmp_path)) == 0

    def test_a_run_that_could_deck_nothing_played_no_round_and_fails(
        self, tmp_path, fake_supervisor, caplog,
    ):
        from effects.application import collect_coverage, coverage_rounds

        with patch.object(collect_coverage, "consult_castability", return_value={}), \
             patch.object(coverage_rounds, "build_coverage_decks", return_value=[]):
            assert collect_coverage.run(self._config(tmp_path)) == 1
        fake_supervisor.play_round.assert_not_called()
        assert "No round was played" in caplog.text
        assert "round 1" not in caplog.text
