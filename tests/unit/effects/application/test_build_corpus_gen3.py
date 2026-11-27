"""A real gen-3 corpus built end to end."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from effects.application.build_corpus import build
from effects.application.build_corpus_config import BuildCorpusConfig
from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.split import HoldoutRule
from effects.domain.stratum_placement import (
    MAX_COMPONENT_ERROR,
    MAX_ITEM_ERROR,
    MIN_GAMES_PER_HALF,
)
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import (
    read_records,
    read_shard,
)
from tests.unit.effects.application.build_corpus_support import (
    _NO_RULES,
    _written,
)
from tests.unit.effects.file_builders import write_vocabulary


def _gen3_records(root: Path) -> Path:
    """A records directory holding the fixture's shard, each of its games
    ended by its marker."""
    import shutil

    from tests.unit.effects.builders import GEN3_RECORDS

    records = root / "records"
    records.mkdir(parents=True)
    shutil.copyfile(GEN3_RECORDS, records / GEN3_RECORDS.name)
    return records


def _gen3_config(records: Path, output: Path, **overrides) -> BuildCorpusConfig:
    """The fixture under the real ``rules/``: a holdout leaving games in both
    validation strata and in training, two games a half."""
    from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS

    values = dict(
        records_dir=records, output=output, cards_folders=GEN3_SIDECAR_FOLDERS,
        keyword_definitions=None, vocab_path=str(write_vocabulary(records.parent / "vocab.txt")),
        workers=1,
        holdout=HoldoutRule(permille=30),
        game_disjoint_share=0.05, validation_sample=0, min_games_per_half=2,
        max_item_error=math.inf, max_component_error=math.inf,
    )
    return BuildCorpusConfig(**{**values, **overrides})


@pytest.fixture(scope="module")
def gen3_build(tmp_path_factory) -> CorpusStore:
    """The fixture built once with the precision thresholds lowered (research E5)."""
    root = tmp_path_factory.mktemp("gen3-build")
    assert build(_gen3_config(_gen3_records(root), root / "out")) == 0
    return CorpusStore(root / "out")


def test_no_turn_action_record_is_written_and_each_is_counted(gen3_build):
    """Story 6 acceptance 8 (FR-059)."""
    assert not [r for r in _written(gen3_build) if r.kind is RecordKind.TURN_ACTION]
    from tests.unit.effects.builders import GEN3_FIXTURE_GAMES

    assert gen3_build.load().turn_action_left_out == {
        "cleanup_discard": 2, "day_night": 2, "draw": 7, "fork_edit": 2,
        "game_end": GEN3_FIXTURE_GAMES, "lore_counter": 2, "phasing": 2, "untap": 8}


def _permanent_spell_resolutions() -> list[EffectRecord]:
    """The fixture's resolution records acting through nothing but a permanent's
    own cast spell."""
    from effects.domain.record_quality import is_permanent_spell
    from effects.infrastructure.converted_trees import process_sidecars
    from tests.unit.effects.builders import GEN3_RECORDS, GEN3_SIDECAR_FOLDERS

    sidecars = process_sidecars({"cardsfolder": GEN3_SIDECAR_FOLDERS[0],
                                 "tokenscripts": GEN3_SIDECAR_FOLDERS[1]})
    return [r for r in read_shard(GEN3_RECORDS) if r.kind is RecordKind.RESOLUTION
            and r.ability and all(is_permanent_spell(k, sidecars.resolution_of(k))
                                  for k in r.ability)]


def test_a_permanent_spells_resolution_records_reach_their_strata(gen3_build):
    """Story 6 acceptance 4: the fixture's creature spells resolve through
    their implicit cast spell, which no line carries; none is refused, and
    each one of a placed game is written, keyed to its card."""
    from effects.domain.strata import OUTPUT_STRATA

    spells = _permanent_spell_resolutions()
    assert spells
    manifest = gen3_build.load()
    assert "no-acting-text" not in manifest.quality_dropped
    placed = {*manifest.card_disjoint_games, *manifest.game_disjoint_games}
    written = {r.record_id: r for r in _written(gen3_build)}
    validated = [r for r in spells if r.game_id in placed]
    assert validated, "the fixture needs a permanent spell in a validation game"
    for record in validated:
        assert written[record.record_id].ability == record.ability
    assert {stratum for stratum in manifest.per_stratum if manifest.per_stratum[stratum]} & set(
        OUTPUT_STRATA)


def test_every_written_copy_carries_its_curation(gen3_build):
    """Research T1: a natural copy draws no rule, and a record's copies agree."""
    records = _written(gen3_build)
    assert records and all(r.curation is not None for r in records)
    for record in records:
        curation = record.curation
        assert 1 <= curation.copies <= 4
        if curation.drawn_rule is None:
            assert curation.rule_entities == ()
        else:
            assert curation.drawn_rule in curation.drawn_rules


def test_the_manifest_holds_every_gen3_field(gen3_build):
    """SC-007, FR-060, FR-060a. ``one_sided_rules`` is shown by
    :func:`test_the_manifest_reports_the_rule_row_and_its_one_side`: a corpus
    this small gives each tagged rule a share of nothing in its class."""
    import json

    from effects.infrastructure.rules_catalogue_io import heads_digest, load_heads

    data = json.loads(gen3_build.manifest_path.read_text(encoding="utf-8"))
    for name in ("rule_counts", "rule_map", "rule_map_digest", "rules",
                 "one_sided_conditions", "unbudgeted_rules", "left_out_of_half",
                 "turn_action_left_out", "subtypes", "keywords", "halves"):
        assert data.get(name), name
    manifest = gen3_build.load()
    assert manifest.rule_map_digest == heads_digest()
    heads = load_heads()
    # The map holds every tagged rule and every rule a copy is drawn for: a
    # protection rule (702.16j's protection from everything) may be drawn on a
    # protection copy with no natural record tagging it.
    tagged = set(manifest.rule_counts["rules"])
    assert tagged <= set(manifest.rule_map)
    assert {rule[:6] for rule in set(manifest.rule_map) - tagged} <= {"702.16"}
    for rule, entry in manifest.rule_map.items():
        assert entry.verdict == (heads[rule].verdict if rule in heads else "missing")
    # FR-054: every tagged rule but a trained one is named as unbudgeted, a
    # rule only a validation game or a turn action shows included.
    assert set(manifest.unbudgeted_rules) == {
        rule for rule, entry in manifest.rule_map.items() if entry.verdict != "trained"}
    from tests.unit.effects.builders import GEN3_FIXTURE_RECORDS

    # Every record the fixture holds, its markers included.
    assert manifest.rule_counts["records"] == GEN3_FIXTURE_RECORDS
    assert set(manifest.halves) == {"card-disjoint", "game-disjoint"}


def test_a_real_gen3_manifest_keeps_its_digest_when_read_back(gen3_build):
    """FR-147: a checkpoint pins its corpus by the manifest's digest. A real
    gen-3 manifest, every gen-3 field filled, reads back to the digest of the
    JSON as written, by the rule feature 023 fixed, and writes back byte for
    byte, so loading the manifest a checkpoint names never moves its digest."""
    import copy
    import hashlib
    import json

    from effects.domain.corpus_manifest import CorpusManifest

    def as_written(data: dict) -> str:
        return json.dumps(data, indent=2, sort_keys=True)

    raw = json.loads(gen3_build.manifest_path.read_text(encoding="utf-8"))
    loaded = CorpusManifest.from_dict(raw)
    payload = copy.deepcopy({
        k: v for k, v in raw.items() if k not in CorpusManifest._OUTSIDE_THE_DIGEST})
    payload["sources"] = sorted(payload["sources"], key=lambda s: s["name"])
    # A rule row's ``drawn`` reports and decides nothing: outside the digest.
    for row in (row for rules in (payload["rules"], *payload["validation_rules"].values())
                for by_rule in rules.values() for by_polarity in by_rule.values()
                for row in by_polarity.values()):
        del row["drawn"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert loaded.digest() == hashlib.blake2b(
        canonical.encode("utf-8"), digest_size=16).hexdigest()
    assert raw["generation"] == 3
    assert as_written(loaded.as_dict()) == as_written(raw)
    assert gen3_build.manifest_path.read_text(encoding="utf-8") == as_written(raw)


def test_the_game_disjoint_stratum_holds_forge_ai_games_only(gen3_build):
    """FR-059: a random-seat game never enters the game-disjoint stratum."""
    assert not [r for r in read_records(gen3_build.game_disjoint_dir) if r.random_seat]


def test_a_gen2_record_stops_the_build_before_any_record_is_read(tmp_path):
    """Story 6 acceptance 6 (FR-018): the CLI maps the refusal to exit 1."""
    import gzip
    import json

    from tests.unit.effects.builders import gen3_fixture_records

    records = _gen3_records(tmp_path)
    with gzip.open(records / "old.jsonl.gz", "wt", encoding="utf-8") as handle:
        for data in gen3_fixture_records():
            del data["generation"]
            handle.write(json.dumps(data) + "\n")
    with pytest.raises(BuildCorpusError, match="the record is gen-2 "):
        build(_gen3_config(records, tmp_path / "out"))
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_at_the_default_thresholds_the_exit_1_comes_after_the_manifest(tmp_path, caplog):
    """Research E5: a corpus this small misses the rule-test target, and the
    build still writes the corpus and its manifest first."""
    import logging

    output = tmp_path / "out"
    config = _gen3_config(
        _gen3_records(tmp_path), output, min_games_per_half=MIN_GAMES_PER_HALF,
        max_item_error=MAX_ITEM_ERROR, max_component_error=MAX_COMPONENT_ERROR,
    )
    with caplog.at_level(logging.ERROR, logger="effects.application.build_corpus"):
        assert build(config) == 1
    store = CorpusStore(output)
    assert store.load().rule_map
    assert list(store.training_dir.glob("*.jsonl.gz"))
    missed = [r.getMessage() for r in caplog.records if "precision target" in r.getMessage()]
    assert any("card-disjoint selection: rule tests" in message for message in missed)
    assert any("game-disjoint reporting: rule tests" in message for message in missed)


def test_protection_copies_are_written_as_their_rules_draw_them(tmp_path):
    """FR-058, research R5: a record a protection copy can be made of enters
    the budgets as a record of its own, and each copy the budgets draw is
    written as a synthetic record carrying its rule."""
    import json

    from effects.domain.protection_model import PROTECTION_RULES

    rules = tmp_path / "rules"
    (rules / "scenarios").mkdir(parents=True)
    (rules / "heads").mkdir()
    # A scenario no record can tag: a build refuses a catalogue with none.
    (rules / "scenarios" / "103.json").write_bytes(
        (_NO_RULES / "scenarios" / "103.json").read_bytes())
    (rules / "heads" / "702.jsonl").write_text("".join(
        json.dumps({"id": rule, "output": "trained", "outputs": []}) + "\n"
        for rule in sorted(PROTECTION_RULES)), encoding="utf-8")
    assert build(_gen3_config(_gen3_records(tmp_path), tmp_path / "out", rules_dir=rules)) == 0
    store = CorpusStore(tmp_path / "out")

    copies = [r for r in _written(store) if r.synthetic]
    assert copies
    for record in copies:
        assert "~pc" in record.record_id
        assert record.curation.natural_share == 0
        assert record.curation.drawn_rule in PROTECTION_RULES
        assert record.curation.drawn_rule in record.curation.drawn_rules
    # The map snapshot names every drawn rule, which no detector tags here:
    # the trainer reads a drawn rule's outputs off the manifest alone.
    manifest = store.load()
    assert {r.curation.drawn_rule for r in copies} <= manifest.budgeted_rules
    assert manifest.budgeted_rules <= PROTECTION_RULES
