"""Cut games, the digest and the manifest's counts, and builds that must agree."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from effects.application.build_corpus import build
from effects.domain.corpus_manifest import SourceShard
from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    RecordKind,
    TurnActionSubkind,
)
from effects.domain.strata import Stratum
from effects.infrastructure.shard_io import (
    iter_shards,
    read_shard,
    write_shard,
)
from tests.unit.effects.application.build_corpus_support import (
    _BUDGET_RULE,
    _HELD_OUT_CARD,
    _HELD_OUT_KEY,
    _SHOCK_KEY,
    _TIMESTAMP,
    CorpusFixture,
    _budget_rules,
    _line,
    _resolution,
    _write_corpus,
    _written,
    make_record,
)


def test_one_raw_corpus_builds_to_one_digest_wherever_it_sits_and_whatever_ran_before(
    tmp_path, a_corpus,
):
    """The selection set is found by the corpus digest: a copy of the raw
    corpus elsewhere, or a second build in the same process, must reach the
    same one. What reading met (a damaged last member, a key read as no line)
    is recorded outside the digest, the damaged shard by its path
    under --records-dir."""
    import gzip
    import shutil

    good = _line(CorpusFixture.resolution("g-clean-1.cut", game="g-clean-1"))
    with open(a_corpus.records / "full-strength" / "run.0-cut.jsonl.gz", "wb") as handle:
        handle.write(gzip.compress(good))
        handle.write(gzip.compress(good)[:12] + bytes(40))

    first = a_corpus.build(tmp_path / "first").load()
    again = a_corpus.build(tmp_path / "again").load()
    moved = tmp_path / "elsewhere" / "records"
    shutil.copytree(a_corpus.records, moved)
    elsewhere = a_corpus.build(tmp_path / "third", records_dir=moved).load()

    assert first.damaged_tails == elsewhere.damaged_tails
    assert first.damaged_tails[0].startswith("full-strength/run.0-cut.jsonl.gz")
    assert first.join_gaps
    assert first.digest() == again.digest() == elsewhere.digest()


def test_a_build_on_two_workers_is_the_build_on_one(tmp_path, a_corpus):
    """Each pool worker hands back its shards' tallies, and the parent sums
    them: two workers write the dataset one does, under the same digest and
    the same reading tallies."""
    import gzip

    good = _line(CorpusFixture.resolution("g-clean-1.cut", game="g-clean-1"))
    with open(a_corpus.records / "full-strength" / "run.0-cut.jsonl.gz", "wb") as handle:
        handle.write(gzip.compress(good))
        handle.write(gzip.compress(good)[:12] + bytes(40))
    one = a_corpus.build(tmp_path / "one").load()
    two = a_corpus.build(tmp_path / "two", workers=2).load()
    assert one.digest() == two.digest()
    assert one.damaged_tails == two.damaged_tails and two.damaged_tails


def test_the_manifest_records_the_code_and_the_sidecars_it_was_built_from(
    tmp_path, a_corpus,
):
    """The commit the build ran on and a hash of the sidecar trees it joined
    against, both outside the digest; and per class the kept resolution
    records acting only through runtime-only keys."""
    first = a_corpus.build(tmp_path / "first").load()
    assert len(first.code_version) >= 40 or first.code_version == ""
    assert first.record_flags["resolution-effect"]["runtime_only"] == 1  # g-clean-3.level
    sidecar = next(a_corpus.cards[0].rglob("*.provenance.json"))
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8")
    second = a_corpus.build(tmp_path / "second").load()
    assert first.sidecar_digest and first.sidecar_digest != second.sidecar_digest
    assert first.digest() == second.digest()


# ── fix round 1: _output_name must be injective (review Finding 1) ─────


def test_output_name_is_injective_on_the_reviewers_counterexample():
    """``"/" -> "__"`` collided: ``"a_/b"`` and ``"a/_b"`` both became
    ``"a___b"``. Two source shards writing the same output filename is a
    silent overwrite (``write_shard`` truncates), so this has to hold.
    """
    assert _part_name("a_/b") != _part_name("a/_b")


def test_output_name_flattens_a_subdirectory_path():
    assert _part_name("depleted/run.0-a.jsonl.gz") == "depleted_srun.0-a.jsonl.gz"


def _part_name(name: str) -> str:
    return SourceShard(name=name, size=0).part_name


# ── fix round 1: per-stratum counts and unique texts (review Finding 2) ─


def test_a_cut_game_is_written_nowhere_and_counted(tmp_path, a_corpus, caplog):
    """A gen-3 game whose shard holds no game-end marker was cut (a
    killed worker); its records are incomplete, so no stratum holds them,
    even when it names a held-out card, and the manifest counts it."""
    cut = [
        _resolution("g-cut.1", "g-cut", ability=(_HELD_OUT_KEY,), entity_names=(_HELD_OUT_CARD,)),
        _resolution("g-cut.2", "g-cut", ability=(_SHOCK_KEY,)),
        make_record(RecordKind.TURN_ACTION, record_id="g-cut.3", game_id="g-cut",
                    timestamp=_TIMESTAMP, ability=None, generation=3),
    ]
    write_shard(a_corpus.records / "full-strength" / "run.0-cut.jsonl.gz", cut)
    with caplog.at_level(logging.INFO):
        store = a_corpus.build(tmp_path / "curated")
    manifest = store.load()
    assert manifest.per_stratum[Stratum.DROPPED_CUT] == 3
    assert manifest.per_stratum[Stratum.DROPPED_UNSURVEYED] == 0
    assert manifest.cut_games == 1
    assert "g-cut" not in {r.game_id for r in _written(store)}
    assert "g-cut" not in {*manifest.card_disjoint_games, *manifest.game_disjoint_games}
    assert "game_end" in manifest.turn_action_left_out
    assert "draw" not in manifest.turn_action_left_out, "a cut game's turn action is its"
    assert "1 of 5 game(s)" in caplog.text and "game-end marker" in caplog.text


def test_the_cut_count_moves_the_digest_and_nothing_else_does(tmp_path, a_corpus):
    """The games dropped as cut decide which records the corpus holds."""
    first = a_corpus.build(tmp_path / "first").load()
    write_shard(a_corpus.records / "full-strength" / "run.0-cut.jsonl.gz",
                [_resolution("g-cut.1", "g-cut", ability=(_SHOCK_KEY,))])
    second = a_corpus.build(tmp_path / "second").load()
    assert (first.cut_games, second.cut_games) == (0, 1)
    assert first.as_dict()["cut_games"] == 0
    assert first.digest() != second.digest()


def test_a_corpus_whose_every_game_was_cut_is_refused(tmp_path):
    """No marker at all: the collection wrote none, and every game would drop."""
    fixture = _write_corpus(tmp_path / "source")
    for shard in iter_shards(fixture.records):
        records = [r for r in read_shard(shard) if r.subkind is not TurnActionSubkind.GAME_END]
        write_shard(shard, records)
    with pytest.raises(BuildCorpusError, match="game-end marker"):
        build(fixture.config(tmp_path / "curated"))


def test_the_manifest_s_rule_counts_say_what_the_catalogue_says_of_untagged_rules(
    tmp_path, a_corpus,
):
    """The rule counts in ``rule-counts.json``'s full shape: the declared rules
    no record tagged, the rules declared without a detector, the untaggable
    declarations and the duplicates (``CatalogueStanding``)."""
    counts = a_corpus.build(tmp_path / "curated").load().rule_counts
    assert {"declared_zero", "none", "untaggable", "same_as"} <= set(counts)
    assert "103.6a" in counts["none"]
    # A rule declared with a detector no record of this corpus satisfies.
    counts = a_corpus.build(tmp_path / "budgeted",
                            rules_dir=_budget_rules(tmp_path / "rules")).load().rule_counts
    assert counts["declared_zero"] == [_BUDGET_RULE]


def test_the_events_no_output_reads_are_counted_outside_the_digest(tmp_path, a_corpus):
    """``RecordTargets.unsupervised``: a name change, a mana cost change and a
    hidden keyword change have no output column; the build counts them by
    kind over the kept records, in the manifest and its report, outside the
    digest."""
    renamed = CorpusFixture.resolution("g-clean-1.renamed", game="g-clean-1", events=(
        Event(type=EventType.NAME_CHANGE, subjects=("E0",), params={"name": "x"}),
        Event(type=EventType.NAME_CHANGE, subjects=("E0",), params={"name": "y"}),
    ))
    a_corpus.with_extra_records([renamed])
    manifest = a_corpus.build(tmp_path / "curated").load()
    assert manifest.unsupervised_events == {"name_change": 2}
    assert manifest.digest() == replace(manifest, unsupervised_events={}).digest()
