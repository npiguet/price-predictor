"""Defective and unjoinable records: dropped, counted or refused."""

from __future__ import annotations

from collections import Counter

import pytest

from effects.application.corpus_report import check_refusals as _check_refusals
from effects.domain.corpus_manifest import SourceShard
from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.infrastructure.shard_io import (
    read_records,
)
from tests.unit.effects.application.build_corpus_support import (
    _BEAR_CARD,
    _BEAR_CAST_KEY,
    _BEAR_FILE,
    _BEAR_TEXT,
    _Card,
    _tallies,
    _write_card,
)


def test_source_of_is_the_first_path_component():
    assert SourceShard(name="depleted/run.0-a.jsonl.gz", size=0).source == "depleted"
    assert SourceShard(name="run.0-a.jsonl.gz", size=0).source == "."


def test_defective_records_reach_no_output_and_are_counted(tmp_path, a_corpus):
    """A junk resolution record in a training game and one in a held-out game both vanish.

    The second is an event flood -- a whole game that landed on one record --
    rather than the unattributed record it used to be: an unattributed event
    is a watch statistic now, not a refusal (FR-148).
    """
    from effects.domain.record_quality import QualityDefect
    corpus = a_corpus.with_extra_records([
        a_corpus.resolution("junk-train", game="g-clean-1", ability=()),
        a_corpus.resolution(
            "junk-held", game="g-tainted",
            events=tuple(
                Event(
                    type=EventType.DAMAGE_DEALT, subjects=("P0",),
                    attributed_to="root",
                )
                for _ in range(65)
            ),
        ),
    ])
    store = corpus.build(tmp_path / "out")
    ids = {r.record_id for d in (store.training_dir, store.card_disjoint_dir,
                                  store.game_disjoint_dir, store.gate_one_dir)
           for r in read_records(d)}
    assert "junk-train" not in ids and "junk-held" not in ids
    assert store.load().quality_dropped == {
        QualityDefect.NO_ABILITY: 1, QualityDefect.EVENT_FLOOD: 1,
    }


def test_a_permanent_spell_s_resolution_survives_keyed_to_its_card(default_build):
    """Story 6 acceptance 4 (FR-059), end to end through both passes.

    ``g-clean-3.cast`` acts through the implicit permanent spell alone — a
    ``dropped_keys`` entry no line carries — and is kept, its acting key the
    permanent's card. ``g-clean-3.level`` acts through a key past everything
    the face declared, which is a real ability whose text the join cannot
    reach yet, and is kept too.
    """
    manifest = default_build.load()
    assert "no-acting-text" not in manifest.quality_dropped
    assert manifest.no_acting_text_scripts == {}
    written = {r.record_id: r for r in read_records(default_build.training_dir)}
    written |= {r.record_id: r for r in read_records(default_build.card_disjoint_dir)}
    written |= {r.record_id: r for r in read_records(default_build.game_disjoint_dir)}
    assert written["g-clean-3.cast"].ability == (_BEAR_CAST_KEY,)
    assert _BEAR_CAST_KEY.script_file == _BEAR_FILE
    assert "g-clean-3.level" in written


def test_a_dropped_key_that_is_no_permanent_spell_is_still_refused(tmp_path, a_corpus):
    """FR-148's third rule, end to end: a resolution acting through nothing
    but a dropped trait other than the implicit cast spell has no text."""
    from effects.domain.record_quality import QualityDefect

    corpus = a_corpus.with_extra_records([
        a_corpus.resolution(
            "g-clean-1.dropped", game="g-clean-1",
            ability=(ProvenanceKey(_BEAR_FILE, 0, "trigger", 0),)),
    ])
    _write_card(corpus.cards[0].parent, _Card(
        _BEAR_FILE, _BEAR_CARD, _BEAR_TEXT, index_within_kind=1,
        dropped=(_BEAR_CAST_KEY, ProvenanceKey(_BEAR_FILE, 0, "trigger", 0)),
    ))
    store = corpus.build(tmp_path / "out")
    assert store.load().quality_dropped == {QualityDefect.NO_ACTING_TEXT: 1}


def test_a_record_acting_through_an_unjoinable_key_is_refused_not_fatal(tmp_path, a_corpus):
    """FR-148's fourth rule, end to end through both passes.

    A charm cast through Flashback keys its mode as a trait plus an ``option``
    no converted line carries, inside a range the sidecar declares. The join
    raises on that key; the build refuses the one record and writes the rest.
    """
    from effects.domain.record_quality import QualityDefect

    mode_key = ProvenanceKey(_BEAR_FILE, 0, "spell", 1, option=0)
    corpus = a_corpus.with_extra_records([
        a_corpus.resolution("flashback-mode", game="g-clean-1", ability=(mode_key,)),
    ])
    store = corpus.build(tmp_path / "curated")
    assert store.load().quality_dropped.get(QualityDefect.UNJOINABLE_KEY) == 1
    written = {
        r.record_id
        for d in (store.training_dir, store.card_disjoint_dir, store.game_disjoint_dir)
        for r in read_records(d)
    }
    assert "flashback-mode" not in written
    assert "g-clean-3.level" in written


def test_unjoinable_keys_past_the_share_stop_the_build():
    """Many records failing the join is a reconversion, not a collector gap."""
    tallies = _tallies(
        read=Counter({"resolution-effect": 990}),
        refused_by_class=Counter({"resolution-effect": 30}),
        quality_dropped=Counter({"unjoinable-key": 30}),
        unjoinable_scripts=Counter({_BEAR_FILE: 30}),
    )

    with pytest.raises(BuildCorpusError) as raised:
        _check_refusals(*tallies)

    message = str(raised.value)
    assert "reconverted" in message
    assert _BEAR_FILE in message


def test_unjoinable_keys_under_the_share_raise_nothing():
    _check_refusals(*_tallies(
        read=Counter({"resolution-effect": 99_970}),
        refused_by_class=Counter({"resolution-effect": 30}),
        quality_dropped=Counter({"unjoinable-key": 30}),
    ))


def test_a_few_unjoinable_keys_in_a_small_corpus_raise_nothing():
    """A pilot of a few hundred records holding one Flashback cast builds."""
    _check_refusals(*_tallies(
        read=Counter({"resolution-effect": 14}),
        refused_by_class=Counter({"resolution-effect": 1}),
        quality_dropped=Counter({"unjoinable-key": 1}),
    ))
