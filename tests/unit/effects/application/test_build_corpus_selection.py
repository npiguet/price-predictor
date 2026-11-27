"""Selection: seats, families, the reuse cap and per-class targets."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.records import (
    RecordKind,
)
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import (
    iter_shards,
    read_records,
    read_shard,
    write_shard,
)
from tests.unit.effects.application.build_corpus_support import (
    _BOLT_KEY,
    _BUDGET_FREE_RULES,
    _HELD_OUT_CARD,
    _SHOCK_KEY,
    _TIMESTAMP,
    _resolution,
    _training_ids,
    make_record,
)
from tests.unit.effects.builders import (
    write_ended_shard,
)


@pytest.fixture(scope="module")
def shared_build(tmp_path_factory, shared_corpus) -> CorpusStore:
    """``shared_corpus`` built once under :data:`_BUDGET_FREE_RULES`."""
    output = tmp_path_factory.mktemp("shared-build") / "curated"
    return shared_corpus.build(output, rules=_BUDGET_FREE_RULES)


def test_random_seat_changes_no_selection(tmp_path, shared_corpus, shared_build):
    """FR-050: the off-policy flag is reported, never selected on.

    Flipped on the training games only: a random seat keeps a game out of
    both validation strata (FR-059), which is placement, not selection.
    """
    manifest = shared_build.load()
    validation = {*manifest.card_disjoint_games, *manifest.game_disjoint_games}
    flipped_root = tmp_path / "flipped"
    for shard in iter_shards(shared_corpus.records):
        records = [r if r.game_id in validation else replace(r, random_seat=True)
                   for r in read_shard(shard)]
        write_shard(flipped_root / shard.relative_to(shared_corpus.records), records)
    flipped = shared_corpus.build(
        tmp_path / "flipped-out", rules=_BUDGET_FREE_RULES, records_dir=flipped_root,
    )

    assert _training_ids(shared_build) == _training_ids(flipped)
    counts = flipped.load().policy_counts
    assert counts and all(c.get("on_policy", 0) == 0 for c in counts.values())


def test_the_manifest_records_families_policy_and_the_flags(tmp_path, shared_corpus):
    """FR-053."""
    manifest = shared_corpus.build(tmp_path / "out", rules=_BUDGET_FREE_RULES, reuse_cap=3).load()

    assert manifest.reuse_cap == 3
    assert manifest.game_disjoint_share == 0.16
    effect = manifest.families["resolution-effect"]
    assert all(
        set(row.as_dict()) == {
            "available", "share", "planned", "written", "repeats", "shortfall",
        }
        for row in effect.values()
    )
    written = sum(row.written for row in effect.values())
    assert written == manifest.per_class["resolution-effect"].kept
    assert manifest.signatures["resolution-effect"]
    assert manifest.policy_counts["resolution-effect"]["on_policy"] == written


def test_a_short_family_is_repeated_up_to_the_reuse_cap(tmp_path, a_corpus):
    """FR-049 step 1 through the whole build: a family below its share repeats."""
    from effects.domain.records import PlayabilityBlockersPayload, PlayabilitySubkind

    legality = [
        make_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
            record_id=f"g-clean-1.leg{i}", game_id="g-clean-1", timestamp=_TIMESTAMP,
            what_if=i >= 2,
            payload=PlayabilityBlockersPayload(anchor_attacker=f"E{i}", must_block=()),
        )
        for i in range(22)
    ]
    a_corpus.with_extra_records(legality)
    manifest = a_corpus.build(
        tmp_path / "out", rules=_BUDGET_FREE_RULES,
        class_mix={"playability-legality": 1.0}, reuse_cap=4, validation_sample=0,
    ).load()

    # 22 distinct records set the class budget. Half of it, 11, is the real
    # half's share; its two records can supply at most 8 with four copies
    # each, and the what-if half is planned the 14 left over.
    families = manifest.families["playability-legality"]
    assert families["real:none"].written == 8
    assert families["real:none"].repeats == 6
    assert families["real:none"].shortfall == 3
    # Records with no acting text are sampled by hash rate: 14 in expectation.
    what_if = families["what-if:none"].written
    assert 8 <= what_if <= 20
    assert manifest.legality_counts["blockers"] == {"real": 8, "what-if": what_if}


# ── what the plan counts, and what it may train on ──────


def test_a_held_back_game_takes_no_training_slot(tmp_path, a_corpus):
    """A text's capped slots go to its training games' records.

    Counted over every kept record, a text's smallest hashes could all sit in
    a held-out game: the plan's threshold admitted those, the write pass then
    routed the game away, and the text's slots went unwritten. Gen-2 wrote
    82% of the records it planned.
    """
    write_ended_shard(a_corpus.records / "full-strength" / "run.0-held.jsonl.gz", [
        _resolution(
            "g-held-2.0", "g-held-2", ability=(_BOLT_KEY,),
            entity_names=(_HELD_OUT_CARD,),
        ),
        *(
            _resolution(f"g-held-2.shock{n}", "g-held-2", ability=(_SHOCK_KEY,))
            for n in range(60)
        ),
    ])
    store = a_corpus.build(tmp_path / "curated", text_cap=2)

    shock = [r for r in read_records(store.training_dir) if r.ability == (_SHOCK_KEY,)]
    assert len(shock) == 2


def test_every_class_delivers_its_target(tmp_path, shared_corpus):
    """The manifest shows targeted beside delivered, so a shortfall is visible."""
    manifest = shared_corpus.build(tmp_path / "curated", text_cap=2).load()

    assert manifest.class_targets
    for klass, target in manifest.class_targets.items():
        assert manifest.per_class[klass].kept == target
