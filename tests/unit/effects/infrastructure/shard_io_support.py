"""Builders shared by the shard-IO tests."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tests.unit.effects.builders import gen3_fixture_records

#: What each older generation's records lack of a gen-3 one's: gen-2 wrote
#: no ``generation``, gen-1 no ``random_seat`` either.
_EARLIER = {
    "gen-1": ("generation", "random_seat"),
    "gen-2": ("generation",),
}


def _older_shard(path: Path, generation: str) -> Path:
    """The gen-3 fixture's records as ``generation`` wrote them: the real
    records less the members it lacked."""
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for data in gen3_fixture_records():
            for member in _EARLIER[generation]:
                del data[member]
            out.write(json.dumps(data) + "\n")
    return path


def _curated(root: Path, shard: Path) -> Path:
    """A curated dataset every one of whose shards is a copy of ``shard``:
    its training shard and both validation samples, beside a manifest."""
    import shutil

    from effects.infrastructure.corpus_store import CorpusStore
    from tests.unit.effects.application.test_train_startup import _manifest

    store = CorpusStore(root / "corpus")
    store.training_dir.mkdir(parents=True)
    store.samples_dir.mkdir(parents=True)
    shutil.copyfile(shard, store.training_dir / "t.jsonl.gz")
    for stratum in ("card-disjoint", "game-disjoint"):
        shutil.copyfile(shard, store.sample_path(stratum))
    store.save(_manifest())
    return store.directory
