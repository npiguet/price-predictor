"""``build-corpus`` refuses every record whose own changes end the game, in
every stratum, under ``game-end``, counted per sampling class (decision D13,
findings 9.7 and 13.3 (a)). The raw corpus keeps them."""

from __future__ import annotations

from collections import Counter

import pytest

from effects.application.build_corpus import build
from effects.domain.event_schema import EventType
from effects.domain.records import RecordKind
from effects.domain.rule_detectors.sba_board import game_ends
from effects.domain.sampling import sampling_class
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import read_shard
from tests.unit.effects.application.build_corpus_support import _written
from tests.unit.effects.application.test_build_corpus_gen3 import _gen3_config, _gen3_records
from tests.unit.effects.builders import GEN3_RECORDS

_GAME_END_EVENTS = frozenset({EventType.PLAYER_LOST, EventType.PLAYER_WON})


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> CorpusStore:
    root = tmp_path_factory.mktemp("game-end-build")
    assert build(_gen3_config(_gen3_records(root), root / "out")) == 0
    return CorpusStore(root / "out")


def _ending_by_class() -> Counter[str]:
    """The fixture's raw records that end the game, by sampling class."""
    return Counter(
        str(sampling_class(r)) for r in read_shard(GEN3_RECORDS)
        if r.kind is not RecordKind.TURN_ACTION and game_ends(r)
    )


def test_the_fixture_holds_game_ending_records_of_two_classes():
    """Two combats and their probes, a lethal burn spell and two game-end
    records: what the build below must refuse."""
    assert _ending_by_class() == {"combat": 4, "resolution-effect": 3}


def test_no_stratum_holds_a_record_holding_the_games_end(built):
    """Training, both validation strata and the gate-one subset, natural and
    added copies alike (a copy's id starts with its record's)."""
    ending = {r.record_id for r in read_shard(GEN3_RECORDS)
              if r.kind is not RecordKind.TURN_ACTION and game_ends(r)}
    written = _written(built)
    assert written
    assert not [r for r in written if any(
        e.type in _GAME_END_EVENTS for e in r.payload.all_events())]
    assert not [r for r in written if r.record_id.split("~")[0] in ending]


def test_the_manifest_counts_the_refusals_by_reason_and_by_class(built):
    manifest = built.load()
    expected = _ending_by_class()
    assert manifest.quality_dropped["game-end"] == sum(expected.values())
    assert manifest.quality_dropped_by_class["game-end"] == dict(expected)
    assert sum(manifest.quality_dropped_by_class[reason][klass]
               for reason in manifest.quality_dropped_by_class
               for klass in manifest.quality_dropped_by_class[reason]
               ) == sum(manifest.quality_dropped.values())
