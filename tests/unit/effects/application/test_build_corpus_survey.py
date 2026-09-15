"""The survey pass: what the whole corpus holds, keyed by provenance key.

The survey never resolves a key to ability text — that fold happens once, in
the main process, against the one ``SidecarCache`` it builds for the holdout
(see ``test_ability_text.py``). So these tests build records with explicit
``ability`` key tuples and check what the survey counts against those keys,
never against text.
"""

from __future__ import annotations

import zlib
from collections import Counter

import pytest

from effects.application import build_corpus
from effects.application.build_corpus import (
    ShardSurvey,
    SurveyConfig,
    ability_key,
    game_hash,
    init_survey_worker,
    merge_surveys,
    parse_ability_key,
    progress_line,
    run_survey,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
)
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot
from effects.infrastructure.record_io import write_shard


def _entity(name: str, **overrides) -> EntityState:
    defaults = {
        "id": f"E-{name}", "name": name, "zone": "battlefield",
        "controller": "P0",
    }
    defaults.update(overrides)
    return EntityState(**defaults)


def _record(game_id: str, entities=(), ability=None, **overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": f"{game_id}.1", "run_id": "run", "timestamp": "t",
        "game_id": game_id, "kind": RecordKind.RESOLUTION,
        "moment": Moment.RESOLUTION, "actor_player": "P0",
        "ability": ability,
        "state": StateSnapshot(
            global_=GlobalState(
                turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
            ),
            players=(),
            entities=tuple(entities),
        ),
        "payload": ResolutionPayload(),
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


#: (moment, payload) for each ``kind`` string a survey test builds a record
#: with. Only the kinds the tests below actually use.
_PAYLOAD_BY_KIND: dict[RecordKind, tuple[Moment | None, object]] = {
    RecordKind.RESOLUTION: (Moment.RESOLUTION, ResolutionPayload()),
    RecordKind.REWRITE: (
        None, RewritePayload(incoming=Event(type=EventType.ZONE_CHANGE)),
    ),
    RecordKind.COMBAT: (None, CombatPayload()),
}


def a_record(
    *,
    game_id: str = "g1",
    record_id: str | None = None,
    kind: str = "resolution",
    ability: tuple[ProvenanceKey, ...] | None = None,
    entity_names: tuple[str, ...] = (),
    **overrides,
) -> EffectRecord:
    """Build a real ``EffectRecord`` for the survey tests.

    A thin wrapper over the copied ``_record``/``_entity`` factories: ``kind``
    (a plain string, e.g. ``"combat"``) picks the matching discriminator and
    payload, and ``entity_names`` builds board entities by name. ``ability``
    normalizes ``()`` to ``None`` so a no-acting-line kind (``combat``) can be
    asked for with either spelling.
    """
    record_kind = RecordKind(kind)
    moment, payload = _PAYLOAD_BY_KIND[record_kind]
    entities = tuple(_entity(name) for name in entity_names)
    extra = {"kind": record_kind, "moment": moment, "payload": payload, **overrides}
    if record_id is not None:
        extra["record_id"] = record_id
    return _record(game_id, entities=entities, ability=(ability or None), **extra)


def a_shard_survey(
    *,
    key: str,
    games: set[int],
    records: int,
    hashes: tuple[int, ...],
    name: str = "shard",
    size: int = 0,
) -> ShardSurvey:
    """One shard's contribution for exactly one ability key, for merge tests."""
    return ShardSurvey(
        name=name,
        size=size,
        records=records,
        key_games={key: set(games)},
        key_records=Counter({key: records}),
        key_hashes={key: tuple(hashes)},
    )


def test_ability_key_round_trips_through_its_rendering():
    keys = (
        ProvenanceKey("cardsfolder/s/swamp.txt", 0, "spell", 1),
        ProvenanceKey("tokenscripts/role_wicked.txt", 1, "static", 0),
    )
    record = a_record(ability=keys)
    assert parse_ability_key(ability_key(record)) == keys


def test_ability_key_is_none_for_a_record_with_no_acting_line():
    assert ability_key(a_record(ability=())) is None


def test_game_hash_is_stable_across_calls():
    """Pinned to the algorithm, not merely to idempotency (FR-088a).

    ``game_hash("x") == game_hash("x")`` is also true of the built-in
    ``hash()``, whose salt is per-process rather than per-call — so that
    assertion alone cannot fail for the exact defect the function's docstring
    warns against, and that defect would only corrupt ``key_games``
    cardinality under the multi-worker pool path, where it is hardest to
    notice.
    """
    assert game_hash("run.0-a.3") == zlib.crc32(b"run.0-a.3")
    assert game_hash("run.0-a.3") == game_hash("run.0-a.3")


def test_survey_counts_records_classes_and_games(tmp_path):
    shard = tmp_path / "run.0-a.jsonl.gz"
    write_shard(shard, [
        a_record(record_id="run.0-a.0", game_id="g1", kind="rewrite"),
        a_record(record_id="run.0-a.1", game_id="g1", kind="rewrite"),
        a_record(record_id="run.0-a.2", game_id="g2", kind="combat", ability=()),
    ])

    survey = run_survey(
        tmp_path,
        config=SurveyConfig(
            records_dir=str(tmp_path),
            held_out_names=frozenset(),
            held_out_script_files=frozenset(),
            text_cap=200,
            seed=42,
        ),
        workers=1,
    )

    assert survey.records == 3
    assert survey.class_records["rewrite"] == 2
    assert survey.class_records["combat"] == 1
    assert survey.games == frozenset({"g1", "g2"})
    assert survey.held_out_games == frozenset()


def test_survey_marks_every_game_naming_a_held_out_card(tmp_path):
    shard = tmp_path / "run.0-a.jsonl.gz"
    write_shard(shard, [
        a_record(record_id="run.0-a.0", game_id="g1", entity_names=("Soul Echo",)),
        a_record(record_id="run.0-a.1", game_id="g2", entity_names=("Grizzly Bears",)),
    ])

    survey = run_survey(
        tmp_path,
        config=SurveyConfig(
            records_dir=str(tmp_path),
            held_out_names=frozenset({"soul echo"}),
            held_out_script_files=frozenset(),
            text_cap=200,
            seed=42,
        ),
        workers=1,
    )

    assert survey.held_out_games == frozenset({"g1"})


def test_survey_counts_games_not_records_for_rarity(tmp_path):
    # One ability, three records, two games: rarity must read 2.
    keys = (ProvenanceKey("cardsfolder/b/bear.txt", 0, "spell", 0),)
    write_shard(tmp_path / "run.0-a.jsonl.gz", [
        a_record(record_id="run.0-a.0", game_id="g1", ability=keys),
        a_record(record_id="run.0-a.1", game_id="g1", ability=keys),
        a_record(record_id="run.0-a.2", game_id="g2", ability=keys),
    ])

    survey = run_survey(
        tmp_path,
        config=SurveyConfig(
            records_dir=str(tmp_path), held_out_names=frozenset(),
            held_out_script_files=frozenset(), text_cap=200, seed=42,
        ),
        workers=1,
    )

    rendered = ability_key(a_record(ability=keys))
    assert len(survey.key_games[rendered]) == 2
    assert survey.key_records[rendered] == 3


def test_survey_records_each_shards_name_relative_to_the_corpus_root(tmp_path):
    (tmp_path / "depleted").mkdir()
    write_shard(tmp_path / "depleted" / "run.0-a.jsonl.gz", [a_record()])

    survey = run_survey(
        tmp_path,
        config=SurveyConfig(
            records_dir=str(tmp_path), held_out_names=frozenset(),
            held_out_script_files=frozenset(), text_cap=200, seed=42,
        ),
        workers=1,
    )

    assert [s.name for s in survey.shards] == ["depleted/run.0-a.jsonl.gz"]


def test_merging_shard_surveys_unions_games_and_sums_records():
    config = SurveyConfig(
        records_dir=".", held_out_names=frozenset(),
        held_out_script_files=frozenset(), text_cap=2, seed=42,
    )
    init_survey_worker(config)
    # Two shards, same ability, disjoint games.
    merged = merge_surveys([
        a_shard_survey(key="k", games={1}, records=1, hashes=(10,)),
        a_shard_survey(key="k", games={2}, records=1, hashes=(20,)),
    ])
    assert merged.key_records["k"] == 2
    assert len(merged.key_games["k"]) == 2
    assert sorted(merged.key_heaps["k"].values()) == [10, 20]


def test_merge_surveys_fails_loudly_when_the_worker_was_never_initialized(
    monkeypatch,
):
    """No silent cap-0 degradation (FR-088a).

    ``config.text_cap if config is not None else 0`` used to be the fallback:
    with no config, ``cap`` became 0, every ``CapHeap(0).offer`` became a
    no-op and every ``threshold()`` read back None, meaning "admit every
    record" — so a missing ``init_survey_worker`` call would report success
    having applied no cap at all. ``survey_shard`` already fails loudly on
    this precondition; ``merge_surveys`` must fail the same way.
    """
    monkeypatch.setattr(build_corpus, "_CONFIG", None)

    with pytest.raises(AssertionError, match="init_survey_worker"):
        merge_surveys([])


class TestProgressSaysWhetherToWait:
    """A count alone does not answer the question an operator is asking.

    "Surveyed 100 of 3205" says the run is alive. It does not say whether the
    remaining 3,105 are ten minutes away or two hours, which is what decides
    whether to sit and watch it.
    """

    def test_the_line_carries_the_rate_and_what_is_left(self):
        line = progress_line(done=100, total=3205, elapsed=120.0)
        assert "100 of 3205" in line
        assert "50/min" in line          # 100 shards in 2 minutes
        assert "62 min left" in line     # 3105 remaining at 50/min

    def test_a_pass_that_has_just_started_reports_no_rate_rather_than_dividing_by_zero(self):
        assert "0 of 10" in progress_line(done=0, total=10, elapsed=0.0)

    def test_the_last_line_says_nothing_is_left(self):
        assert "0 min left" in progress_line(done=10, total=10, elapsed=5.0)
