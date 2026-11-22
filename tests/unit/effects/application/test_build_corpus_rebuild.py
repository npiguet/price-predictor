"""Rebuilding over a previous dataset, and verify leaving it alone."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    CombatPayload,
)
from effects.domain.strata import Stratum
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import (
    read_records,
    write_shard,
)
from tests.unit.effects.application.build_corpus_support import (
    _BOLT_KEY,
    _BUDGET_FREE_RULES,
    CorpusFixture,
    _combat,
    _resolution,
)


def test_a_rebuild_removes_a_shard_the_new_build_does_not_write(tmp_path, a_corpus):
    """FR-144: a grown corpus is rebuilt whole, never extended in place.

    Same-named source shards are truncate-overwritten, so the common case
    survived; a source shard renamed, removed, or a rebuild pointed at a
    different --records-dir left files behind that the trainer reads as part
    of the dataset while the manifest and its digest describe only the second
    build.
    """
    out = tmp_path / "curated"
    store = a_corpus.build(out)

    orphans = [
        store.training_dir / "orphan.jsonl.gz",
        store.card_disjoint_dir / "orphan.jsonl.gz",
        store.game_disjoint_dir / "orphan.jsonl.gz",
    ]
    for orphan in orphans:
        write_shard(orphan, [_resolution("orphan.1", "g-orphan", ability=(_BOLT_KEY,))])

    a_corpus.build(out)

    assert not any(orphan.exists() for orphan in orphans)
    assert store.manifest_path.exists()
    assert "g-orphan" not in {r.game_id for r in read_records(store.training_dir)}


def test_verify_leaves_the_written_dataset_alone(tmp_path, a_corpus):
    out = tmp_path / "curated"
    store = a_corpus.build(out)
    before = sorted(p.name for p in store.training_dir.glob("*"))

    a_corpus.build(out, verify=True)

    assert sorted(p.name for p in store.training_dir.glob("*")) == before


def test_a_refused_rebuild_leaves_the_previous_dataset_in_place(tmp_path, a_corpus):
    """The survey sees every refusal the write pass would, so a corpus the
    rules reject is refused before the old strata are cleared."""
    out = tmp_path / "out"
    a_corpus.build(out)
    before = CorpusStore(out).load().digest()
    flooded = a_corpus.with_extra_records([
        replace(
            _combat(f"g-clean-1.flood-{n}", "g-clean-1"),
            payload=CombatPayload(
                attackers=("E0",),
                events=tuple(
                    Event(type=EventType.DAMAGE_DEALT, subjects=("P0",),
                          attributed_to="root")
                    for _ in range(100)
                ),
            ),
        )
        for n in range(5)
    ])

    with pytest.raises(BuildCorpusError, match="combat"):
        flooded.build(out)

    store = CorpusStore(out)
    assert store.load().digest() == before
    assert any(store.output_dir(Stratum.TRAINING).iterdir())


# ── I3: a stratum with an empty sample is a build that failed quietly ──


def test_a_rebuild_over_a_grown_corpus_keeps_every_game_disjoint_game(tmp_path, a_corpus):
    """SC-007."""
    first = set(
        a_corpus.build(tmp_path / "first", rules=_BUDGET_FREE_RULES).load().game_disjoint_games
    )
    assert first

    a_corpus.with_extra_records([
        CorpusFixture.resolution(f"g-more-{i}.1", game=f"g-more-{i}") for i in range(40)
    ])
    grown = set(
        a_corpus.build(tmp_path / "grown", rules=_BUDGET_FREE_RULES).load().game_disjoint_games
    )

    assert first <= grown
