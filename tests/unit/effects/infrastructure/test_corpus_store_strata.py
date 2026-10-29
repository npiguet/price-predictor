"""Gate 1's population: card-disjoint resolutions whose text training never acted with.

Both halves come from the curated corpus the checkpoint trained on. The raw
shards are the wrong source for either: every game outside the recorded
validation sets is not a training game — curation dropped millions of held-out
records and the corpus keeps growing after a build — and reading those as
training put held-out texts in the "seen" set, draining the stratum.
"""

from __future__ import annotations

from effects.application.gate_one import unique_text_stratum
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import EffectRecord, RecordKind
from effects.infrastructure.corpus_store import (
    CorpusStore,
    card_disjoint_resolutions,
    training_acting_keys,
)
from effects.infrastructure.record_io import write_shard
from tests.unit.effects.builders import make_record


def _key(name: str) -> ProvenanceKey:
    return ProvenanceKey(f"cardsfolder/x/{name}.txt", 0, "spell", 0)


def _record(
    game: str, number: int, ability, kind: RecordKind = RecordKind.RESOLUTION,
) -> EffectRecord:
    return make_record(
        kind, record_id=f"{game}.{number}", run_id="r", game_id=game,
        ability=ability,
    )


class _Texts:
    """A key's text is the card it names."""

    def text_of(self, key):
        return key.script_file.rsplit("/", 1)[-1].removesuffix(".txt")


def _corpus(tmp_path, *, training, card_disjoint):
    store = CorpusStore(tmp_path / "corpus")
    write_shard(store.training_dir / "shard-00001.jsonl.gz", training)
    write_shard(store.card_disjoint_dir / "shard-00001.jsonl.gz", card_disjoint)
    return store


def test_training_acting_keys_reads_the_curated_training_stratum(tmp_path):
    shock = _record("train", 1, (_key("shock"),))
    store = _corpus(
        tmp_path,
        # A curated record written twice, as --reuse-cap copies are.
        training=[shock, shock, _record("train", 2, None, RecordKind.COMBAT)],
        card_disjoint=[_record("held", 1, (_key("bolt"),))],
    )
    assert training_acting_keys(store.training_dir) == {(_key("shock"),)}


def test_card_disjoint_resolutions_streams_the_stratum_s_resolutions_only(tmp_path):
    store = _corpus(
        tmp_path, training=[],
        card_disjoint=[
            _record("held", 1, (_key("bolt"),)),
            _record("held", 2, None, RecordKind.COMBAT),
            _record("other", 1, (_key("growth"),)),
        ],
    )
    found = card_disjoint_resolutions(store.directory, frozenset({"held"}))
    assert [r.record_id for r in found] == ["held.1"]


def test_a_text_seen_only_outside_the_training_stratum_stays_unique(tmp_path):
    """Raw shards outside the build (dropped held-out games, later runs) are
    not training: a held-out text they act with is still unseen."""
    store = _corpus(
        tmp_path,
        training=[_record("train", 1, (_key("shock"),))],
        card_disjoint=[
            _record("held", 1, (_key("bolt"),)),
            _record("held", 2, (_key("shock"),)),
        ],
    )
    raw = tmp_path / "records"
    write_shard(raw / "dropped.jsonl.gz", [_record("dropped", 1, (_key("bolt"),))])

    stratum = unique_text_stratum(store.directory, frozenset({"held"}), _Texts().text_of)

    assert [r.record_id for r in stratum] == ["held.1"]
