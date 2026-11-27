"""Damaged shards and records that move between the survey and the write pass."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from effects.application.build_corpus import build
from effects.domain.corpus_manifest import SourceShard
from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.provenance import ProvenanceKey
from effects.domain.strata import Stratum
from effects.domain.stratum_placement import (
    MAX_ITEM_ERROR,
)
from effects.infrastructure.shard_io import (
    write_shard,
)
from tests.unit.effects.application.build_corpus_support import (
    _HELD_OUT_CARD,
    _SHOCK_KEY,
    CorpusFixture,
    _combat,
    _dataset,
    _line,
    _resolution,
    _written,
)
from tests.unit.effects.record_builders import lines_of, make_entity


def _raw_shard(path: Path, *members: bytes) -> None:
    """A shard of the given gzip members, as a worker appends them."""
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        for member in members:
            handle.write(gzip.compress(member))


@pytest.mark.parametrize("damage", ["undecodable line", "corrupt member", "unknown generation"])
def test_a_damaged_shard_stops_the_build_naming_it_before_anything_is_cleared(
    tmp_path, a_corpus, damage,
):
    """A shard the survey cannot read whole makes the dataset unusable: the build
    exits 1 naming the shard, and the earlier dataset under --output survives."""
    import gzip
    import json

    from effects.infrastructure.record_codec import record_to_dict

    out = tmp_path / "curated"
    a_corpus.build(out)
    before = _dataset(out)
    good = _line(CorpusFixture.resolution("g-damaged.1", game="g-damaged"))
    shard = a_corpus.records / "full-strength" / "run.0-damaged.jsonl.gz"
    if damage == "undecodable line":
        _raw_shard(shard, good + b"{not json\n")
    elif damage == "corrupt member":
        with open(shard, "wb") as handle:
            handle.write(gzip.compress(good))
            handle.write(gzip.compress(good)[:12] + bytes(40))
            handle.write(gzip.compress(good))
    else:
        data = record_to_dict(CorpusFixture.resolution("g-damaged.2", game="g-damaged"))
        data["generation"] = 4
        _raw_shard(shard, good + (json.dumps(data) + "\n").encode("utf-8"))

    with pytest.raises(BuildCorpusError, match="run.0-damaged.jsonl.gz"):
        build(a_corpus.config(out))
    assert _dataset(out) == before


@pytest.mark.parametrize("game", ["g-late", "g-clean-1"])
def test_records_a_shard_gains_between_the_passes_are_written_nowhere(
    tmp_path, a_corpus, caplog, game,
):
    """A collection still appending under --records-dir. A new game was never
    placed and may name a held-out card; a game the survey placed has records
    it never planned. Either way the write pass reads past what the survey
    read, writes those records nowhere and counts them."""
    import gzip

    from effects.application import build_corpus as module

    shard = a_corpus.records / "full-strength" / "run.0-a.jsonl.gz"
    late = [_resolution(f"{game}.late{n}", game, ability=(_SHOCK_KEY,),
                        entity_names=(_HELD_OUT_CARD,) if game == "g-late" else ())
            for n in range(3)]
    decide = module._decide_or_refuse

    def decide_then_grow(*args, **kwargs):
        decided = decide(*args, **kwargs)
        with open(shard, "ab") as handle:
            handle.write(gzip.compress(b"".join(_line(record) for record in late)))
        return decided

    with patch.object(module, "_decide_or_refuse", decide_then_grow):
        store = a_corpus.build(tmp_path / "curated")
    manifest = store.load()
    assert manifest.per_stratum[Stratum.DROPPED_UNSURVEYED] == 3
    assert not {r.record_id for r in late} & {r.record_id for r in _written(store)}
    assert "run.0-a.jsonl.gz" in caplog.text and "changed size since the survey" in caplog.text


def test_records_appended_after_the_survey_to_a_shard_holding_a_cut_game_are_written_nowhere(
    tmp_path, a_corpus, monkeypatch,
):
    """A shard holding an ended game and a cut one gains records between the
    passes, the cut game's last ones and its marker among them: the survey
    read the cut game as cut, so the write pass drops what the survey read of
    it as cut and the rest, never surveyed, as unsurveyed; none is written,
    and the ended game is."""
    import gzip

    from effects.application import build_corpus as module
    from tests.unit.effects.record_builders import game_end_marker

    shard = a_corpus.records / "full-strength" / "run.0-mixed.jsonl.gz"
    ended = [_resolution(f"g-ended.{n}", "g-ended", ability=(_SHOCK_KEY,)) for n in range(2)]
    cut = [_resolution(f"g-cut.{n}", "g-cut", ability=(_SHOCK_KEY,)) for n in range(2)]
    write_shard(shard, [*ended, game_end_marker(ended[-1]), *cut])
    late = [_resolution("g-cut.late", "g-cut", ability=(_SHOCK_KEY,))]
    late.append(game_end_marker(late[-1]))
    decide = module._decide_or_refuse

    def decide_then_grow(*args, **kwargs):
        decided = decide(*args, **kwargs)
        with open(shard, "ab") as handle:
            handle.write(gzip.compress(b"".join(_line(record) for record in late)))
        return decided

    monkeypatch.setattr(module, "_decide_or_refuse", decide_then_grow)
    store = a_corpus.build(tmp_path / "curated")
    manifest = store.load()
    written = {r.record_id for r in _written(store)}
    assert manifest.cut_games == 1
    assert manifest.per_stratum[Stratum.DROPPED_CUT] == 2
    assert manifest.per_stratum[Stratum.DROPPED_UNSURVEYED] == 2
    assert not {r.record_id for r in [*cut, *late]} & written
    assert "g-ended" in {r.game_id for r in _written(store)} | {
        *manifest.card_disjoint_games, *manifest.game_disjoint_games}


def test_a_damaged_last_member_after_a_marker_is_named_once_and_changes_nothing(
    tmp_path, a_corpus, caplog,
):
    """The survey reads a shard's markers, then its records: the damaged
    last member is one shard's one damaged tail, and the write pass, reading
    the same bytes, finds the same tally."""
    import gzip

    from tests.unit.effects.record_builders import game_end_marker

    ended = [_resolution(f"g-d.{n}", "g-d", ability=(_SHOCK_KEY,)) for n in range(2)]
    after = [_resolution(f"g-e.{n}", "g-e", ability=(_SHOCK_KEY,)) for n in range(2)]
    good = gzip.compress(b"".join(_line(r) for r in [*ended, game_end_marker(ended[-1])]))
    bad = bytearray(gzip.compress(b"".join(_line(r) for r in after)))
    bad[len(bad) // 2] ^= 0xFF
    (a_corpus.records / "full-strength" / "run.0-damaged.jsonl.gz").write_bytes(good + bytes(bad))

    with caplog.at_level(logging.WARNING):
        manifest = a_corpus.build(tmp_path / "curated").load()

    assert len(manifest.damaged_tails) == 1
    assert manifest.damaged_tails[0].startswith("full-strength/run.0-damaged.jsonl.gz")
    assert "changed between the passes" not in caplog.text


def test_the_manifest_records_how_its_records_were_collected_and_read(
    tmp_path, a_corpus, caplog,
):
    """Per class, the kept records collected as forks, interventions,
    directed forks, aimed or degraded; the placement thresholds and the
    keyword table the build ran under; and what reading met without stopping:
    values outside the wire's closed lists and keys of a tree with no root."""
    import gzip
    import hashlib
    import json

    from effects.infrastructure.record_codec import record_to_dict

    definitions = tmp_path / "keyword-definitions.json"
    definitions.write_text("{}", encoding="utf-8")
    odd = record_to_dict(CorpusFixture.resolution("g-clean-1.odd", game="g-clean-1"))
    odd["state"]["global"]["combat_substep"] = "third_strike"
    variant = ProvenanceKey("variant-scripts/x_variant.txt", 0, "spell", 0)
    # An attacker printing a line of the variant tree, which this build has no
    # root for: the survey's keyword read finds no line for it.
    from effects.domain.state_snapshot import CombatStatus

    unrooted = _combat("g-clean-1.variant", "g-clean-1")
    unrooted = replace(unrooted, state=replace(unrooted.state, entities=(
        make_entity("E0", "x variant", lines=lines_of(printed=(variant,)),
                    combat=CombatStatus(attacking="P1")),)))
    from tests.unit.effects.record_builders import game_end_marker

    # The game ends in this shard too: a game without its marker is cut.
    with open(a_corpus.records / "full-strength" / "run.0-extra.jsonl.gz", "wb") as handle:
        handle.write(gzip.compress((json.dumps(odd) + "\n").encode("utf-8") + _line(unrooted)
                                   + _line(game_end_marker(unrooted))))

    # A records set naming a tree the run has no root for is refused, naming it.
    with pytest.raises(BuildCorpusError, match="variant-scripts"):
        a_corpus.build(tmp_path / "unrooted", keyword_definitions=definitions)
    assert not (tmp_path / "unrooted").exists(), "nothing under --output was touched"
    with open(a_corpus.records / "full-strength" / "run.0-extra.jsonl.gz", "wb") as handle:
        handle.write(gzip.compress((json.dumps(odd) + "\n").encode("utf-8")
                                   + _line(game_end_marker(unrooted))))

    manifest = a_corpus.build(tmp_path / "curated", keyword_definitions=definitions).load()

    assert manifest.record_flags["combat"]["fork"] == 1
    assert manifest.placement_thresholds == {
        "min_games_per_half": 0, "max_item_error": MAX_ITEM_ERROR, "max_component_error": None}
    assert manifest.keyword_definitions_digest == hashlib.sha256(b"{}").hexdigest()
    assert manifest.decode_off_list == {"global.combat_substep 'third_strike'": 1}
    bears = "cardsfolder/g/grizzly_bears.txt"
    assert manifest.join_gaps == {"runtime_only": {bears: 4}, "dropped": {bears: 4}}
    # The two passes ask the sidecars different numbers of times: a join gap
    # alone is no sign the corpus changed between them.
    assert "changed between the passes" not in caplog.text
    data = json.loads((tmp_path / "curated" / "manifest.json").read_text(encoding="utf-8"))
    assert data["damaged_tails"] == []


def test_a_shard_whose_last_member_is_damaged_builds_and_is_named_in_the_manifest(
    tmp_path, a_corpus,
):
    """A shard whose last gzip member is damaged: the read ends there quietly
    (a worker killed mid-write leaves such a tail), and the manifest names it."""
    import gzip

    good = _line(CorpusFixture.resolution("g-clean-1.cut", game="g-clean-1"))
    shard = a_corpus.records / "full-strength" / "run.0-cut.jsonl.gz"
    with open(shard, "wb") as handle:
        handle.write(gzip.compress(good))
        handle.write(gzip.compress(good)[:12] + bytes(40))
    manifest = a_corpus.build(tmp_path / "curated").load()
    assert any("run.0-cut.jsonl.gz" in where for where in manifest.damaged_tails)


def test_only_a_decode_tally_differing_between_the_passes_is_a_changed_corpus(caplog):
    """The two passes ask the sidecar cache different numbers of times, so
    their join gaps differ on every build with one; their decode tallies read
    the same bytes and differ only when a shard changed between them."""
    from effects.application.corpus_report import report_write_pass
    from effects.application.corpus_survey import ReadGaps, ShardSurvey, Survey
    from effects.application.corpus_write import WriteResult
    from effects.domain.corpus_plan import TrainingSupply

    survey = Survey(totals=ShardSurvey(shard=SourceShard("", 0), supply=TrainingSupply(1),
                                       reading=ReadGaps(unconfigured=Counter({"t": 3}))),
                    shards=())
    report_write_pass(survey, WriteResult(reading=ReadGaps(unconfigured=Counter({"t": 1}))))
    assert "changed between the passes" not in caplog.text
    report_write_pass(survey, WriteResult(reading=ReadGaps(
        off_list=Counter({("global.phase", "'x'"): 1}), unconfigured=Counter({"t": 3}))))
    assert "changed between the passes" in caplog.text


def test_a_shard_damaged_between_the_passes_stops_the_write_pass_and_says_so(
    tmp_path, a_corpus, monkeypatch,
):
    """The survey read the shard whole; damage appended before the write pass
    stops the build there, after --output was cleared, and the message says
    the dataset was cleared."""
    import gzip

    from effects.application import build_corpus as module

    shard = a_corpus.records / "full-strength" / "run.0-a.jsonl.gz"
    good = _line(CorpusFixture.resolution("g-clean-1.later", game="g-clean-1"))
    decide = module._decide_or_refuse

    def decide_then_damage(*args, **kwargs):
        decided = decide(*args, **kwargs)
        with open(shard, "ab") as handle:
            handle.write(gzip.compress(good)[:12] + bytes(40))
            handle.write(gzip.compress(good))
        return decided

    monkeypatch.setattr(module, "_decide_or_refuse", decide_then_damage)
    with pytest.raises(BuildCorpusError, match="already cleared") as raised:
        build(a_corpus.config(tmp_path / "curated"))
    assert "run.0-a.jsonl.gz" in str(raised.value)
