from __future__ import annotations

import json

import pytest

from effects.domain.corpus_manifest import (
    ClassCounts,
    CorpusManifest,
    CuratedShard,
    FamilyRow,
    ManifestError,
    SourceShard,
    rule_curation_fields,
)


def manifest(**overrides) -> CorpusManifest:
    base = dict(
        seed=42,
        surface="script",
        vocab_path="models/effects/vocab-script.txt",
        variant_scripts="output/effects/variant-scripts",
        holdout_permille=20,
        holdout_max_carriers=8,
        text_cap=200,
        card_disjoint_text_cap=50,
        training_records=0,
        class_mix={"rewrite": 1.0},
        delivered_mix={"rewrite": 1.0},
        held_out_cards=("soul echo",),
        held_out_texts=("deals 3 damage",),
        card_disjoint_games=("run.0-a.1",),
        game_disjoint_games=("run.0-a.2",),
        rarity={"deals 3 damage": 7},
        sources=(SourceShard(name="depleted/run.0-a.jsonl.gz", size=1234),),
        per_class={"rewrite": ClassCounts(read=10, kept=4, dropped_by_cap=6, unique_texts=2)},
        per_stratum={
            "training": 4, "card-disjoint": 3, "game-disjoint": 2,
            "dropped-held-out": 1,
        },
        unique_texts={"training": 2, "card-disjoint": 1, "game-disjoint": 1},
        shortfall={},
    )
    base.update(overrides)
    return CorpusManifest(**base)


@pytest.fixture
def manifest_dict() -> dict:
    return manifest().as_dict()


def test_round_trips_through_a_dict():
    original = manifest()
    assert CorpusManifest.from_dict(original.as_dict()) == original


def test_the_digest_rule_is_pinned():
    """The hash rule is fixed: a checkpoint records this exact string (FR-147)."""
    assert manifest().digest() == "c1570dee3a1e4210c68b9b234e6dd7f3"


def test_every_field_is_written_and_the_manifest_names_its_generation(manifest_dict):
    from dataclasses import fields

    assert set(manifest_dict) == {spec.name for spec in fields(CorpusManifest)}
    assert manifest_dict["generation"] == 3


def test_a_manifest_of_an_older_generation_is_refused_naming_it(manifest_dict):
    """A gen-2 build wrote no ``generation`` member but wrote the gen-2 tables;
    a gen-1 manifest has neither."""
    gen2 = {k: v for k, v in manifest_dict.items() if k != "generation"}
    with pytest.raises(ManifestError, match="gen-2 manifest"):
        CorpusManifest.from_dict(gen2)
    gen1 = {k: v for k, v in gen2.items()
            if k not in ("families", "signatures", "policy_counts")}
    with pytest.raises(ManifestError, match="gen-1 manifest"):
        CorpusManifest.from_dict(gen1)
    with pytest.raises(ManifestError, match="gen-4 manifest"):
        CorpusManifest.from_dict({**manifest_dict, "generation": 4})


def test_a_manifest_missing_a_member_is_refused(manifest_dict):
    del manifest_dict["rule_map_digest"]
    with pytest.raises(ManifestError, match="rule_map_digest"):
        CorpusManifest.from_dict(manifest_dict)


def test_digest_changes_when_a_recorded_decision_changes():
    assert manifest().digest() != manifest(text_cap=100).digest()


def test_digest_ignores_the_order_a_source_list_was_built_in():
    a = manifest(sources=(
        SourceShard(name="b.jsonl.gz", size=2), SourceShard(name="a.jsonl.gz", size=1),
    ))
    b = manifest(sources=(
        SourceShard(name="a.jsonl.gz", size=1), SourceShard(name="b.jsonl.gz", size=2),
    ))
    assert a.digest() == b.digest()


def test_drift_names_added_and_removed_shards():
    current = (
        SourceShard(name="a.jsonl.gz", size=1),
        SourceShard(name="c.jsonl.gz", size=3),
    )
    added, removed, resized = manifest().drift(current)
    assert added == ("a.jsonl.gz", "c.jsonl.gz")
    assert removed == ("depleted/run.0-a.jsonl.gz",)
    assert resized == ()


def test_drift_reports_a_shard_that_grew():
    current = (SourceShard(name="depleted/run.0-a.jsonl.gz", size=9999),)
    added, removed, resized = manifest().drift(current)
    assert (added, removed) == ((), ())
    assert resized == ("depleted/run.0-a.jsonl.gz",)


def test_digest_changes_when_the_delivered_mixture_changes():
    """Two datasets holding different class proportions are different
    datasets, so a checkpoint pinned to one must refuse the other."""
    assert manifest().digest() != manifest(
        delivered_mix={"rewrite": 0.5, "combat": 0.5},
    ).digest()


def test_the_tallies_round_trip(manifest_dict):
    manifest_dict.update({
        "quality_dropped": {"no-ability": 3},
        "games_by_source": {"depleted": 10, "full-strength": 4},
        "held_out_games_by_source": {"depleted": 1, "full-strength": 4},
        "shard_records": 2000,
        "validation_sample": 2048,
        "max_events_per_record": 64,
        "unattributed_records": 512,
    })
    loaded = CorpusManifest.from_dict(manifest_dict)
    assert loaded.as_dict()["quality_dropped"] == {"no-ability": 3}
    assert loaded.unattributed_records == 512
    assert CorpusManifest.from_dict(loaded.as_dict()) == loaded


def test_the_no_acting_text_tally_stays_out_of_the_digest(manifest_dict):
    """The digest is what FR-147 pins a checkpoint to, and it stands for the
    split and the contents. ``no_acting_text_scripts`` says which scripts the
    build refused records through, which an operator reads and no sampling
    decision depends on."""
    before = CorpusManifest.from_dict(manifest_dict)
    after = CorpusManifest.from_dict({
        **manifest_dict,
        "no_acting_text_scripts": {"cardsfolder/g/grizzly_bears.txt": 12},
    })

    assert before.no_acting_text_scripts == {}
    assert after.no_acting_text_scripts == {"cardsfolder/g/grizzly_bears.txt": 12}
    assert before.digest() == after.digest()


def test_a_different_held_out_text_still_moves_the_digest():
    """The exclusion is one named field, not a hole in the hash."""
    assert manifest().digest() != manifest(
        held_out_texts=("gains 4 life",),
    ).digest()


# ── curation fields (FR-053, FR-042; T082, T084) ───────────────────────────


def test_the_curation_fields_round_trip_and_move_the_digest():
    curated = manifest(
        families={"rewrite": {"Moved": FamilyRow(
            available=10, share=5, planned=6, written=8, repeats=3, shortfall=0,
        )}},
        signatures={"rewrite": {"Moved": {"died:changed": 8}}},
        policy_counts={"rewrite": {"on_policy": 6, "off_policy": 2}},
        legality_counts={"blockers": {"real": 4, "what_if": 9}},
        keyword_threshold_games=("run.0-a.9",),
        held_out_texts_without_resolution=("deals 3 damage",),
        held_out_texts_under_five_games={"deals 3 damage": 2},
        game_disjoint_share=0.01,
        game_disjoint_keyword_share=0.15,
        game_disjoint_keywords=("trample",),
        reuse_cap=4,
    )
    restored = CorpusManifest.from_dict(json.loads(json.dumps(curated.as_dict())))

    assert restored == curated
    assert isinstance(restored.families["rewrite"]["Moved"], FamilyRow)
    assert curated.digest() != manifest().digest()


def test_class_targets_round_trip_beside_what_was_delivered():
    """G4: a class written short of its target shows in the manifest."""
    original = manifest(class_targets={"rewrite": 5})
    data = original.as_dict()
    assert data["class_targets"] == {"rewrite": 5}
    assert CorpusManifest.from_dict(data) == original
    assert original.digest() != manifest().digest()


# ── gen-3 fields (FR-060, FR-060a; T138) ────────────────────────────────


def _gen3_fields(**extra_map) -> dict:
    """``extra_map``: further ``rule -> RuleMapEntry`` snapshot entries."""
    from effects.domain.corpus_plan import GameStrata
    from effects.domain.rule_budgets import BudgetedRecord, BudgetInputs, plan_rule_budgets
    from effects.domain.rule_catalogue import MapVerdict, Polarity, RuleMapEntry
    from effects.domain.rule_detectors import RuleSource
    from effects.domain.strata import Half, Stratum

    records = [
        BudgetedRecord(game="g1", record_hash=1, klass="effects", text="t1", natural=1,
                       source=RuleSource.FORGE_AI, tags=(("702.16e", Polarity.FIRES),)),
        BudgetedRecord(game="g6", record_hash=6, klass="effects", text="t6", natural=1,
                       source=RuleSource.FORGE_AI, tags=(("702.16e", Polarity.FIRES),)),
        BudgetedRecord(game="g2", record_hash=2, klass="effects", text="t2", natural=1,
                       source=RuleSource.RANDOM_SEAT,
                       tags=(("702.16e", Polarity.DOES_NOT_FIRE), ("510.1", Polarity.FIRES))),
    ]
    budgets = plan_rule_budgets(records, BudgetInputs(
        class_targets={"effects": 8}, budgeted={"702.16e"}, reuse_cap=2, text_cap=0))
    strata = GameStrata(
        card_disjoint=frozenset({"g3"}), game_disjoint=frozenset({"g4", "g5"}),
        held_out=frozenset({"g3"}), surveyed=frozenset({"g1", "g2", "g3", "g4", "g5"}),
        halves={Stratum.GAME_DISJOINT: {Half.SELECTION: frozenset({"g5", "g4"}),
                                        Half.REPORTING: frozenset()}},
        left_out_of_half={Stratum.GAME_DISJOINT: {Half.REPORTING: ("702.16e",)}},
    )
    return rule_curation_fields(
        budgets=budgets, strata=strata,
        rule_map={
            "702.16e": RuleMapEntry("702.16e", MapVerdict.TRAINED, ("damage_taken", "gate")),
            "510.1": RuleMapEntry("510.1", MapVerdict.RECORDED),
            **extra_map,
        },
        rule_map_digest="abc123",
        rule_counts={"records": 2, "rules": {"702.16e": {"fires": {"forge_ai": 1}}},
                     "not_countable": [], "none": {}, "review": []},
        carried={("702.16e", "damage_taken"): 3},
        one_sided_conditions={"decision": ["Berserk"]},
        turn_action_left_out={"untap": 3, "draw": 2},
        subtypes={"zombie", "plains", "not-a-subtype"},
        keywords={"toxic", "flying"},
    )


def test_the_gen3_fields_round_trip_through_json():
    gen3 = manifest(**_gen3_fields())
    restored = CorpusManifest.from_dict(json.loads(json.dumps(gen3.as_dict())))
    assert restored == gen3
    assert gen3.digest() != manifest().digest()


def test_the_gen3_fields_hold_what_the_build_decided():
    from effects.domain.closed_lists import ordered_keywords
    from effects.domain.forge_vocabularies import SUBTYPES

    gen3 = manifest(**_gen3_fields())
    data = json.loads(json.dumps(gen3.as_dict()))
    row = data["rules"]["effects"]["702.16e"]["fires"]
    assert row == {"available": {"forge_ai": 2}, "share": 4, "written": {"forge_ai": 4},
                   "added": {"forge_ai": 2}, "shortfall": 0}
    assert data["rules"]["effects"]["702.16e"]["does_not_fire"]["shortfall"] == 2
    assert data["rules"]["effects"]["510.1"]["fires"]["share"] is None
    assert data["unbudgeted_rules"] == {"510.1": {"fires": {"random_seat": 1}}}
    assert data["one_sided_rules"] == {"effects": ["702.16e"]}
    assert data["rule_map"]["702.16e"] == {"verdict": "trained",
                                           "outputs": ["damage_taken", "gate"]}
    assert data["halves"] == {"game-disjoint": {"selection": ["g4", "g5"], "reporting": []}}
    assert data["left_out_of_half"] == {"game-disjoint": {"reporting": ["702.16e"]}}
    assert gen3.subtypes == tuple(sorted(("zombie", "plains"), key=SUBTYPES.index))
    assert gen3.keywords == ordered_keywords({"toxic", "flying"})
    assert gen3.budgeted_rules == {"702.16e"}
    assert gen3.rule_outputs("702.16e") == ("damage_taken", "gate")
    assert gen3.rule_outputs("999.9") == ()
    # Only the outputs a tagged record carried are effective, in the map's order.
    assert data["effective_outputs"] == {"510.1": [], "702.16e": ["damage_taken"]}
    assert gen3.effective_rule_outputs == {"510.1": (), "702.16e": ("damage_taken",)}


def test_an_unbudgeted_rule_no_training_record_shows_is_still_named():
    """FR-054: every tagged rule the map does not mark ``trained`` is named as
    unbudgeted, a rule tagged only on records that train nowhere (a validation
    game's, a ``turn_action`` record's) with no counts."""
    from effects.domain.rule_catalogue import MapVerdict, RuleMapEntry

    fields = _gen3_fields(**{"502.3": RuleMapEntry("502.3", MapVerdict.MISSING)})
    data = json.loads(json.dumps(manifest(**fields).as_dict()))
    assert data["unbudgeted_rules"] == {"502.3": {}, "510.1": {"fires": {"random_seat": 1}}}


def test_the_collection_fields_round_trip():
    """P3a #17: a validation stratum's rule rows, the collection flags, the
    thresholds the build ran under and what reading met all read back."""
    from effects.domain.rule_budgets import RuleRow

    row = RuleRow(available={"forge_ai": 2}, share=4, written={"forge_ai": 3},
                  added={"forge_ai": 1}, shortfall=1)
    built = manifest(
        validation_rules={"card-disjoint": {"resolution-effect": {"702.2b": {"fires": row}}}},
        record_flags={"combat": {"fork": 1}},
        placement_thresholds={"min_games_per_half": 300, "max_item_error": 0.1,
                              "max_component_error": None},
        keyword_definitions_digest="ab", decode_off_list={"global.phase 'x'": 2},
        damaged_tails=("run.0-a.jsonl.gz",), join_gaps={"unconfigured": {"variant-scripts": 3}},
    )
    back = CorpusManifest.from_dict(json.loads(json.dumps(built.as_dict())))
    assert back == built


def test_the_rules_tagged_nowhere_written_round_trip_and_leave_the_digest_alone():
    """They report what the counts name and no stratum writes; they decide
    nothing, so the digest a checkpoint pins ignores them."""
    plain = manifest(**_gen3_fields())
    reported = manifest(**_gen3_fields(), tagged_nowhere_written=("104.2a", "703.4c"))
    restored = CorpusManifest.from_dict(json.loads(json.dumps(reported.as_dict())))
    assert restored.tagged_nowhere_written == ("104.2a", "703.4c")
    assert reported.digest() == plain.digest()


def test_the_curated_shards_round_trip_and_leave_the_digest_alone():
    """``shards`` lists each stratum's files and record counts, which the
    trainer compares with what it finds on disk (finding 11.4). Two builds
    deciding the same thing and naming their shards differently (another
    repack, another filesystem) have one digest."""
    listed = manifest(shards={
        "training": (CuratedShard(name="shard-00001.jsonl.gz", records=40),
                     CuratedShard(name="shard-00002.jsonl.gz", records=38)),
        "card-disjoint": (),
    })
    renamed = manifest(shards={
        "training": (CuratedShard(name="part-a.jsonl.gz", records=78),),
        "card-disjoint": (),
    })
    restored = CorpusManifest.from_dict(json.loads(json.dumps(listed.as_dict())))
    assert restored == listed
    assert restored.shards["training"][1] == CuratedShard(
        name="shard-00002.jsonl.gz", records=38)
    assert listed.digest() == renamed.digest() == manifest().digest()
