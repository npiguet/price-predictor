"""The build's outputs are what its manifest says (findings 11.4, 14.3).

The manifest lists every shard each stratum holds, with the records the
write pass put in it, so a reader can tell a stale or missing shard from the
dataset the manifest describes.
"""

from __future__ import annotations

import pytest

from effects.domain.corpus_manifest import CuratedShard
from effects.domain.strata import OUTPUT_STRATA
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import iter_shards, read_shard
from tests.unit.effects.application.test_build_corpus import CorpusFixture, _write_corpus


@pytest.fixture(scope="module")
def a_corpus(tmp_path_factory) -> CorpusFixture:
    return _write_corpus(tmp_path_factory.mktemp("outputs") / "source")


@pytest.fixture(scope="module")
def built(tmp_path_factory, a_corpus) -> CorpusStore:
    """Repacked to small shards, so a stratum holds several."""
    return a_corpus.build(tmp_path_factory.mktemp("outputs-build") / "curated", shard_records=2)


def _on_disk(store: CorpusStore, stratum: str) -> tuple[CuratedShard, ...]:
    directory = store.output_dir(stratum)
    return tuple(
        CuratedShard(name=path.relative_to(directory).as_posix(),
                     records=sum(1 for _ in read_shard(path)))
        for path in sorted(iter_shards(directory))
    )


def test_the_manifest_lists_every_shard_of_every_stratum_with_its_records(built):
    listed = built.load().shards

    assert set(listed) == set(OUTPUT_STRATA)
    assert len(listed["training"]) > 1, "the fixture must repack into several shards"
    for stratum in OUTPUT_STRATA:
        assert listed[stratum] == _on_disk(built, stratum), stratum


def test_the_listing_is_sorted_by_name_and_sums_to_the_stratum_counts(built):
    manifest = built.load()
    for stratum, shards in manifest.shards.items():
        assert [shard.name for shard in shards] == sorted(shard.name for shard in shards)
        assert sum(shard.records for shard in shards) == manifest.per_stratum[stratum]
