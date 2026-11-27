"""The coverage collector: what a coverage count credits, an older
record stopping it, and the workers it runs on."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from effects.application.collect_coverage import CollectCoverageConfig
from effects.domain.event_schema import Event, EventType
from tests.unit.effects.application.collect_coverage_support import (
    _GEN3_ROOTS,
    _GEN3_SHARD,
    _holding,
    _texts_by_game,
    _turn_action,
)
from tests.unit.effects.builders import (
    make_entity,
    write_plain_shard,
)
from tests.unit.effects.domain.test_coverage import _record

# ── what a coverage count credits (the generation) ──


class TestOnlyWhatBuildCorpusKeepsCounts:
    """``count_shard`` credited every record kind, ``turn_action``
    included, so every card drawn (and every permanent untapped) made
    progress though build-corpus writes no such record to any stratum."""

    def test_a_turn_action_record_credits_nothing(self, tmp_path):
        from effects.application.coverage_counting import count_shard

        drawn, land = make_entity("E1", "drawn card", zone="hand"), make_entity("E2", "forest")
        shard = write_plain_shard(tmp_path, "x", [_turn_action([drawn, land], [
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "library", "to_zone": "hand"}),
            Event(type=EventType.UNTAPPED, subjects=("E2",)),
        ])])

        assert count_shard(shard).credits == {}

    def test_a_record_build_corpus_refuses_credits_nothing(self, tmp_path):
        """A resolution with no acting ability is refused as ``no-ability``."""
        from dataclasses import replace

        from effects.application.coverage_counting import count_shard

        record = replace(
            _record([make_entity("E1", "shock")], source="E1"), ability=None,
        )
        shard = write_plain_shard(tmp_path, "x", [record])

        assert count_shard(shard).credits == {}

    def test_a_kept_record_still_credits(self, tmp_path):
        from effects.application.coverage_counting import count_shard

        shard = write_plain_shard(tmp_path, "x", [
            _record([make_entity("E1", "shock")], source="E1"),
        ])

        assert count_shard(shard).credits == {"shock": 1}


class TestAnOlderRecordStopsTheCount:
    """A record of an older generation met counting a run's shards is a
    refusal, not damage to count past: every reader refuses the set it is in,
    so the collector exits 1 before playing a game (FR-033)."""

    @staticmethod
    def _older_shard(tmp_path: Path) -> Path:
        """The gen-3 fixture's records less ``generation``: gen-2's."""
        import gzip
        import json

        from tests.unit.effects.builders import gen3_fixture_records

        path = tmp_path / "old.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as out:
            for data in gen3_fixture_records():
                del data["generation"]
                out.write(json.dumps(data) + "\n")
        return path

    def test_the_card_count_refuses_it_naming_its_generation(self, tmp_path):
        from effects.application.coverage_counting import count_shard
        from effects.domain.shard_generation import OlderGenerationError

        with pytest.raises(OlderGenerationError, match=r"line 1: the record is gen-2 "):
            count_shard(self._older_shard(tmp_path))

    def test_the_held_out_text_count_refuses_it(self, tmp_path):
        from effects.application.coverage_counting import count_text_shard
        from effects.domain.shard_generation import OlderGenerationError

        _, everything = _texts_by_game(_GEN3_SHARD)
        with pytest.raises(OlderGenerationError, match="the record is gen-2 "):
            count_text_shard(self._older_shard(tmp_path), roots=_GEN3_ROOTS,
                             held_out=_holding(everything))

    def test_the_collectors_exit_1_on_it(self):
        from effects.domain.shard_generation import OlderGenerationError
        from effects.infrastructure import cli

        for run in (cli.run_collect_coverage, cli.run_collect_variants):
            assert cli._exit_code(run.exit_codes, OlderGenerationError("gen-2")) == 1


class TestTheCountersUseTheRunsWorkers:
    """Shard counting passes the run's ``workers``, so it never runs one
    process per CPU, which would exceed the machine's budget."""

    def test_every_count_runs_on_config_workers(self, tmp_path, fake_supervisor):
        from effects.application import collect_coverage

        cards = tmp_path / "cardsfolder"
        cards.mkdir()
        (cards / "aa.txt").write_text(
            "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        seen = []
        real = collect_coverage.CoverageCounter.update

        def _spy(self, coverage, records_dir, *, workers=None):
            seen.append(workers)
            return real(self, coverage, records_dir, workers=workers)

        with patch.object(collect_coverage, "consult_castability", return_value={}), \
                patch.object(collect_coverage.CoverageCounter, "update", _spy):
            collect_coverage.run(CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                target_records=1, decks_per_round=1, no_progress_rounds=1, workers=7,
            ))

        assert len(seen) >= 2, "the initial count and a round's count"
        assert set(seen) == {7}
