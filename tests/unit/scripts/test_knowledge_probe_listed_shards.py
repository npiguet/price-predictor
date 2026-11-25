"""The probes' record walk reads the shards the manifest lists (finding 11.4)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from effect_knowledge_probes import common, method_c_probe, probing, record_walk
from effect_knowledge_probes.line_labelling import ObservedOutcomes

from effects.domain.corpus_manifest import CuratedShard
from effects.domain.strata import Stratum
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.curated_shards import CorpusShardsError, count_records
from effects.infrastructure.shard_io import write_shard
from tests.unit.effects.application.test_train_startup import _manifest

SHARD = "shard-00001.jsonl.gz"


@pytest.fixture
def context(tmp_path, records):
    """A context over a corpus listing one shard in each of its three strata."""
    store = CorpusStore(tmp_path / "corpus")
    listing = {}
    for stratum in ("training", *common.STRATA):
        shard = store.output_dir(Stratum(stratum)) / SHARD
        write_shard(shard, records[:6])
        listing[stratum] = (CuratedShard(SHARD, count_records(shard)),)
    manifest = _manifest(shards=listing)
    store.save(manifest)
    return common.ProbeContext(
        checkpoint_path=tmp_path / "latest.pt", checkpoint=None, corpus=store.directory,
        manifest=manifest, corpus_digest="", cards_folders=(), variant_scripts=None)


def _add_stale_shard(context, stratum: str, records) -> None:
    directory = CorpusStore(context.corpus).output_dir(Stratum(stratum))
    write_shard(directory / "stale.jsonl.gz", records[:2])


def test_the_walk_reads_the_listed_shards(context, records):
    walked = list(record_walk.records(context.stratum_shards("training")))
    assert [r.record_id for r in walked] == [r.record_id for r in records[:6]]


def test_the_validation_shards_are_each_strata_first_listed_shards(context):
    both = [context.stratum_shards(s)[0] for s in common.STRATA]
    assert list(context.validation_shards(1)) == both


@pytest.mark.parametrize("stratum", ["training", *common.STRATA])
def test_a_stale_shard_is_refused_by_name_by_every_walk(context, records, stratum):
    _add_stale_shard(context, stratum, records)
    walks = [
        lambda: context.validation_shards(),
        lambda: context.stratum_shards("training"),
        lambda: probing._Probing._observe_training(SimpleNamespace(  # noqa: SLF001
            args=SimpleNamespace(profile_shards=None), context=context,
            outcomes=ObservedOutcomes(lambda _: None))),
        lambda: next(method_c_probe._stream(context.stratum_shards("training"))),  # noqa: SLF001
    ]
    for walk in walks:
        with pytest.raises(CorpusShardsError, match=f"{stratum}: stale.jsonl.gz is not listed"):
            walk()
