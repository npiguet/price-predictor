"""The survey pass: what the whole corpus holds, keyed by acting keys.

The survey never folds a key to its rendered ability text — that fold happens
once, in the main process (see ``test_build_corpus_decide.py``). So these tests
build records with explicit ``ability`` key tuples and check what the survey
counts against those keys.

Two populations are counted apart: rarity, the games and the held-out texts'
games over every kept record, and the training supply over the records the
write pass will route to training alone.
"""

from __future__ import annotations

from collections import Counter

import pytest

from effects.application.corpus_survey import (
    ReadConfig,
    ShardSurvey,
    SurveyConfig,
    init_survey_worker,
    merge_surveys,
    run_survey,
    survey_shard,
)
from effects.application.parallel_map import progress_line
from effects.domain.corpus_curation import game_disjoint_draw
from effects.domain.corpus_manifest import SourceShard
from effects.domain.corpus_plan import Cell, GameRouting, TrainingSupply
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.split import HeldOutCards
from effects.infrastructure.shard_io import list_shards, write_shard
from effects.infrastructure.sidecar_io import sidecar_path_for
from tests.unit.effects.builders import make_entity, make_record, make_snapshot, write_sidecar

_PAYLOAD_BY_KIND: dict[RecordKind, tuple[Moment | None, object]] = {
    RecordKind.RESOLUTION: (Moment.RESOLUTION, ResolutionPayload()),
    RecordKind.REWRITE: (
        None, RewritePayload(incoming=Event(type=EventType.ZONE_CHANGE)),
    ),
    RecordKind.COMBAT: (None, CombatPayload()),
    RecordKind.TRIGGER: (
        None, TriggerPayload(event=Event(type=EventType.ZONE_CHANGE), fired=True),
    ),
}

_BOLT = (ProvenanceKey("cardsfolder/c/common_bolt.txt", 0, "spell", 0),)
_SECRET = (ProvenanceKey("cardsfolder/s/secret.txt", 0, "spell", 0),)
_GOBLIN = (ProvenanceKey("tokenscripts/goblin_secret.txt", 0, "spell", 1),)


def a_record(
    *,
    game_id: str = "g1",
    record_id: str | None = None,
    kind: str = "resolution",
    ability: tuple[ProvenanceKey, ...] | None = None,
    entity_names: tuple[str, ...] = (),
    **overrides,
) -> EffectRecord:
    """A real ``EffectRecord``: ``kind`` picks the discriminator and payload."""
    record_kind = RecordKind(kind)
    moment, payload = _PAYLOAD_BY_KIND[record_kind]
    entities = tuple(make_entity(f"E-{name}", name) for name in entity_names)
    extra = {"moment": moment, "payload": payload, **overrides}
    if record_id is not None:
        extra["record_id"] = record_id
    return make_record(
        record_kind, game_id=game_id, ability=(ability or None),
        state=make_snapshot(entities=entities), **extra,
    )


def converted_tree(tmp_path, *keys: ProvenanceKey, text=None) -> dict[str, str]:
    """A converted tree whose sidecars render one line for each of ``keys``.

    ``sidecar_roots`` for a survey config. The survey refuses a resolution
    record whose every acting key maps to no rendered line (FR-148), so a test
    about what the survey *keys* on has to say that its keys carry text —
    otherwise it measures the refusal instead of the keying. ``text`` names
    each line's script text by its key; the default is the script's stem.
    """
    root = tmp_path / "converted"
    by_script: dict[str, list[ProvenanceKey]] = {}
    for key in keys:
        by_script.setdefault(key.script_file, []).append(key)
    for script_file, script_keys in by_script.items():
        tree, relative = script_file.split("/", 1)
        path = root / tree / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_sidecar(
            ProvenanceSidecar(
                card=path.stem,
                script_file=script_file,
                lines=tuple(
                    SidecarLine(
                        line_index=index, line_kind="spell", provenance=(key,),
                        script_text=(text or {}).get(key, f"{path.stem} line {index}"),
                    )
                    for index, key in enumerate(script_keys)
                ),
            ),
            sidecar_path_for(path),
        )
    return {
        tree: str(root / tree)
        for tree in {key.script_file.split("/", 1)[0] for key in keys}
    }


#: The text ``_SECRET`` and ``_GOBLIN`` both print, and the holdout that names it.
_SECRET_TEXT = "SP$ DealDamage | NumDmg$ 9"
_HOLDOUT = HeldOutCards(
    names=frozenset({"secret card"}),
    script_files=frozenset({_SECRET[0].script_file, _GOBLIN[0].script_file}),
    texts=frozenset({_SECRET_TEXT}), keys=frozenset({_SECRET_TEXT}), unit="text",
)


def a_config(
    records_dir, *, held_out: HeldOutCards | None = None, sidecar_roots=None,
    heap_cap: int = 200, share: float = 0.0,
) -> SurveyConfig:
    return SurveyConfig(
        read=ReadConfig(
            records_dir=str(records_dir), sidecar_roots=dict(sidecar_roots or {}),
            max_events=64,
            held_out=held_out or HeldOutCards(names=frozenset(), script_files=frozenset()),
            seed=42,
        ),
        heap_cap=heap_cap,
        routing=GameRouting(game_disjoint_share=share, keyword_share=share),
    )


def _survey_one(tmp_path, records, **config):
    write_shard(tmp_path / "records" / "run.0-a.jsonl.gz", records)
    init_survey_worker(a_config(tmp_path / "records", **config))
    return survey_shard(list_shards(tmp_path / "records")[0])


def test_survey_counts_records_classes_and_games(tmp_path):
    write_shard(tmp_path / "run.0-a.jsonl.gz", [
        a_record(record_id="run.0-a.0", game_id="g1", kind="rewrite"),
        a_record(record_id="run.0-a.1", game_id="g1", kind="rewrite"),
        a_record(record_id="run.0-a.2", game_id="g2", kind="combat", ability=()),
    ])

    survey = run_survey(tmp_path, config=a_config(tmp_path), workers=1)

    assert survey.totals.records == 3
    assert survey.totals.games.games == {"g1", "g2"}
    assert survey.totals.games.held_out == set()


def test_survey_marks_every_game_naming_a_held_out_card(tmp_path):
    # Both records are refused: no sidecar root, so the acting key reads as an
    # unconverted script and FR-148's no-acting-text rule takes it. g1 is
    # still marked held out -- a game naming a held-out card is marked before
    # any of its records is judged, so no quality rule can unmark its game.
    keys = (ProvenanceKey("cardsfolder/b/bear.txt", 0, "spell", 0),)
    write_shard(tmp_path / "run.0-a.jsonl.gz", [
        a_record(record_id="run.0-a.0", game_id="g1", ability=keys,
                 entity_names=("Soul Echo",)),
        a_record(record_id="run.0-a.1", game_id="g2", ability=keys,
                 entity_names=("Grizzly Bears",)),
    ])

    survey = run_survey(
        tmp_path,
        config=a_config(tmp_path, held_out=HeldOutCards(
            names=frozenset({"soul echo"}), script_files=frozenset(),
        )),
        workers=1,
    )

    assert survey.totals.games.held_out == {"g1"}


def test_survey_counts_games_not_records_for_rarity(tmp_path):
    # One ability, three records, two games: rarity must read 2.
    keys = (ProvenanceKey("cardsfolder/b/bear.txt", 0, "spell", 0),)
    write_shard(tmp_path / "run.0-a.jsonl.gz", [
        a_record(record_id="run.0-a.0", game_id="g1", ability=keys),
        a_record(record_id="run.0-a.1", game_id="g1", ability=keys),
        a_record(record_id="run.0-a.2", game_id="g2", ability=keys),
    ])

    survey = run_survey(
        tmp_path, config=a_config(tmp_path, sidecar_roots=converted_tree(tmp_path, *keys)),
        workers=1,
    )

    assert len(survey.totals.rarity.key_games[keys]) == 2
    assert survey.totals.rarity.key_records[keys] == 3


def test_survey_records_each_shards_name_relative_to_the_corpus_root(tmp_path):
    (tmp_path / "depleted").mkdir()
    write_shard(tmp_path / "depleted" / "run.0-a.jsonl.gz", [a_record()])

    survey = run_survey(tmp_path, config=a_config(tmp_path), workers=1)

    assert [s.name for s in survey.shards] == ["depleted/run.0-a.jsonl.gz"]


def _part(name: str, games: set[str], hashes: tuple[int, ...], *, heap_cap: int = 2):
    supply = TrainingSupply(heap_cap)
    for value in hashes:
        supply.add(_CELL, _BOLT, value)
    part = ShardSurvey(
        shard=SourceShard(name=name, size=0), supply=supply, records=len(hashes),
    )
    part.games.games = set(games)
    for game in games:
        part.rarity.note(_BOLT, game)
    return part


#: The one cell the merge tests put their records in.
_CELL = Cell("rewrite", family="Moved")


def test_merging_shard_surveys_unions_games_and_sums_records():
    merged = merge_surveys(
        [_part("a", {"g1"}, (10,)), _part("b", {"g2"}, (20,))], heap_cap=2,
    )
    assert merged.totals.rarity.key_records[_BOLT] == 2
    assert len(merged.totals.rarity.key_games[_BOLT]) == 2
    assert sorted(merged.totals.supply.heaps[_CELL, _BOLT].values()) == [10, 20]
    assert merged.totals.supply.keyed[_CELL, _BOLT] == 2
    assert merged.split_games == 0


def test_a_game_found_in_two_shards_is_counted(tmp_path):
    """A raw shard is one JVM lifetime; a game in two means the routing guessed."""
    merged = merge_surveys(
        [_part("a", {"g1"}, (10,)), _part("b", {"g1", "g2"}, (20,))], heap_cap=2,
    )
    assert merged.split_games == 1


def test_merging_supplies_kept_at_different_heap_sizes_fails_loudly():
    """No silent truncation: a heap merged into a smaller one loses hashes."""
    with pytest.raises(ValueError, match="cannot merge"):
        merge_surveys([_part("a", {"g1"}, (10,), heap_cap=5)], heap_cap=2)


def test_a_defective_record_is_counted_and_otherwise_ignored(tmp_path):
    """A record with no acting ability is not a key, a class or a game (FR-148)."""
    from effects.domain.record_quality import QualityDefect

    out = _survey_one(tmp_path, [
        a_record(record_id="clean", game_id="g1", ability=_BOLT),
        a_record(record_id="junk", game_id="g2", ability=()),
    ], sidecar_roots=converted_tree(tmp_path, *_BOLT))

    assert out.refusals.by_reason == {QualityDefect.NO_ABILITY: 1}
    assert out.records == 2
    assert out.games.games == {"g1"}


class TestProgressSaysWhetherToWait:
    """A count alone does not answer the question an operator is asking."""

    def test_the_line_carries_the_rate_and_what_is_left(self):
        line = progress_line(done=100, total=3205, elapsed=120.0)
        assert "100 of 3205" in line
        assert "50/min" in line          # 100 shards in 2 minutes
        assert "62 min left" in line     # 3105 remaining at 50/min

    def test_a_pass_that_has_just_started_reports_no_rate_rather_than_dividing_by_zero(self):
        assert "0 of 10" in progress_line(done=0, total=10, elapsed=0.0)

    def test_the_last_line_says_nothing_is_left(self):
        assert "0 min left" in progress_line(done=10, total=10, elapsed=5.0)


def test_a_refused_record_still_routes_its_game_to_the_card_disjoint_stratum(tmp_path):
    """Held-out routing is decided before the quality check.

    A game whose only mention of a held-out card sits on a record the quality
    rules refuse is still a game that played that card.
    """
    out = _survey_one(tmp_path, [
        a_record(record_id="junk", game_id="g1", ability=(), entity_names=("Soul Echo",)),
    ], held_out=HeldOutCards(names=frozenset({"soul echo"}), script_files=frozenset()))

    assert out.games.held_out == {"g1"}
    assert out.games.games == set()


# ── the training supply: training-bound records only (G4) ──────────────


def test_the_supply_counts_only_records_the_write_pass_trains_on(tmp_path):
    """Every budget and copy threshold is sized on this supply.

    Counted over held-out and game-disjoint games too, it planned training
    slots for records the write pass then routed to validation or dropped, and
    gen-2 delivered 82% of what it planned. Rarity still counts every game.
    """
    games = [f"g{i}" for i in range(400)]
    disjoint = next(g for g in games if game_disjoint_draw(g) < 0.05)
    training = next(g for g in games if game_disjoint_draw(g) >= 0.05)
    held = next(g for g in games if g not in (disjoint, training))
    out = _survey_one(tmp_path, [
        a_record(record_id="t", game_id=training, kind="rewrite", ability=_BOLT),
        a_record(record_id="d", game_id=disjoint, kind="rewrite", ability=_BOLT),
        a_record(record_id="h", game_id=held, kind="rewrite", ability=_BOLT,
                 entity_names=("Soul Echo",)),
    ], held_out=HeldOutCards(names=frozenset({"soul echo"}), script_files=frozenset()),
        sidecar_roots=converted_tree(tmp_path, *_BOLT), share=0.05)

    assert sum(out.supply.keyed.values()) == 1
    assert out.rarity.key_records[_BOLT] == 3
    assert out.games.game_disjoint == {disjoint}
    assert out.games.held_out == {held}


def test_a_keyword_combat_places_its_game_at_the_keyword_threshold(tmp_path):
    from effects.domain.corpus_plan import GameRoute

    routing = GameRouting(game_disjoint_share=0.01, keyword_share=0.15)
    games = [f"g{i}" for i in range(400)]
    between = next(g for g in games if 0.01 <= game_disjoint_draw(g) < 0.15)
    assert routing.route(between, held_out=False, keyword_combat=False) is GameRoute.TRAINING
    assert routing.route(between, held_out=False, keyword_combat=True) is GameRoute.GAME_DISJOINT
    assert routing.route(between, held_out=True, keyword_combat=True) is GameRoute.HELD_OUT
    assert routing.under_keyword_threshold(between)


# ── held-out texts (G13, G14) ──────────────────────────────────────────


def test_only_a_resolution_counts_a_game_toward_its_held_out_text(tmp_path):
    """The card-disjoint cap admits games gate 1 can score, which are resolutions.

    A decision candidate, an unfired trigger, a rewrite or a cost half acting
    through the text holds nothing gate 1 measures; counted, 616 of 4,814
    admitted gen-2 games held no gate-one record at all.
    """
    roots = converted_tree(tmp_path, *_SECRET, text={_SECRET[0]: _SECRET_TEXT})
    out = _survey_one(tmp_path, [
        a_record(record_id="r", game_id="g-resolved", ability=_SECRET),
        a_record(record_id="t", game_id="g-triggered", kind="trigger", ability=_SECRET),
    ], held_out=_HOLDOUT, sidecar_roots=roots)

    assert out.games.held_out_text_games == {_SECRET_TEXT: {"g-resolved"}}
    assert out.games.held_out == {"g-resolved", "g-triggered"}


def test_a_training_record_acting_through_a_held_out_text_is_not_supplied(tmp_path):
    """A variant or token printing a held-out text, in a game naming no held-out card."""
    variant = (ProvenanceKey("variant-scripts/secret_variant_1.txt", 0, "spell", 0),)
    roots = converted_tree(
        tmp_path, *variant, *_BOLT, text={variant[0]: _SECRET_TEXT},
    )
    out = _survey_one(tmp_path, [
        a_record(record_id="v", game_id="g1", kind="rewrite", ability=variant),
        a_record(record_id="b", game_id="g1", kind="rewrite", ability=_BOLT),
    ], held_out=_HOLDOUT, sidecar_roots=roots)

    assert out.games.held_out == set()
    assert [keys for _cell, keys in out.supply.keyed] == [_BOLT]
    assert out.held_out_text_dropped == Counter({"rewrite": 1})
