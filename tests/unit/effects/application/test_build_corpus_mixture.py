"""The class mixture a build delivers against the one the manifest records."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from effects.domain.records import (
    EffectRecord,
)
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import (
    read_records,
)
from tests.unit.effects.application.build_corpus_support import (
    _BOLT_CARD,
    _BOLT_FILE,
    _BOLT_KEY,
    _BOLT_TEXT,
    _HELD_OUT_CARD,
    _HELD_OUT_FILE,
    _HELD_OUT_KEY,
    _HELD_OUT_TEXT,
    _MIX_PER_SHARD,
    _MIX_SHARDS,
    _MIXTURE,
    CorpusFixture,
    _combat,
    _resolution,
    _rewrite,
    _write_card,
)
from tests.unit.effects.builders import (
    write_ended_shard,
)


@pytest.fixture(scope="module")
def a_mixture_corpus(tmp_path_factory) -> CorpusFixture:
    """Four shards, one text repeated far past the cap, and an uncapped class.

    ``rewrite`` carries one ability text and is what the cap bites; ``combat``
    has no acting line at all (FR-087), so nothing caps it and its
    availability is exact either way. Asked for half and half, the two come
    out half and half only if availability is estimated corpus-wide.

    One further game holds the held-out ability resolving, which is what the
    card-disjoint stratum is made of; its records are ``resolution-effect``,
    a class outside this corpus's ``--class-mix``, so nothing it holds reaches
    the training mixture under test.
    """
    root = tmp_path_factory.mktemp("mixture-source")
    _write_card(root, _HELD_OUT_FILE, _HELD_OUT_CARD, _HELD_OUT_TEXT)
    _write_card(root, _BOLT_FILE, _BOLT_CARD, _BOLT_TEXT)

    records_dir = root / "records"
    for shard in range(_MIX_SHARDS):
        rows: list[EffectRecord] = [
            _rewrite(f"rw.{shard}.{n}", f"g-rw-{shard}", ability=(_BOLT_KEY,))
            for n in range(_MIX_PER_SHARD)
        ]
        rows += [
            _combat(f"cb.{shard}.{n}", f"g-cb-{shard}")
            for n in range(_MIX_PER_SHARD)
        ]
        if shard == 0:
            rows.append(
                _resolution("g-tainted.1", "g-tainted", ability=(_HELD_OUT_KEY,)),
            )
        write_ended_shard(records_dir / f"run.0-{shard}.jsonl.gz", rows)

    return CorpusFixture(
        records=records_dir, cards=(root / "cardsfolder",),
    )


@pytest.fixture(scope="module")
def mixture_build(tmp_path_factory, a_mixture_corpus) -> CorpusStore:
    return a_mixture_corpus.build(tmp_path_factory.mktemp("mixture") / "curated", **_MIXTURE)


def _delivered_shares(training_dir: Path) -> dict[str, float]:
    from effects.domain.sampling import sampling_class

    counts: Counter[str] = Counter()
    for record in read_records(training_dir):
        counts[sampling_class(record)] += 1
    total = sum(counts.values())
    return {name: count / total for name, count in counts.items()}


def test_the_delivered_class_mixture_is_the_one_the_manifest_records(mixture_build):
    """FR-139: a class is written at its share or its shortfall is recorded.

    ``class_capped_records`` was summed from each shard's own ``CapHeap``
    thresholds while the real cap is corpus-wide, so availability was
    over-estimated by a factor of the shard count for exactly the classes the
    cap touches. ``build()`` divides each class's target by that availability
    to get its admission rate, so every capped class under-delivered against
    the mixture the manifest went on to record.
    """
    delivered = _delivered_shares(mixture_build.training_dir)
    requested = mixture_build.load().class_mix
    for name, share in requested.items():
        assert abs(delivered.get(name, 0.0) - share) < 0.05, (
            f"{name}: delivered {delivered.get(name, 0.0):.3f} against a "
            f"recorded share of {share:.3f}"
        )


def test_the_manifest_records_what_was_delivered_beside_what_was_asked_for(mixture_build):
    """The requested mixture is a claim; the delivered one is a fact, and the
    dataset's whole value is that its manifest is true."""
    manifest = mixture_build.load()
    delivered = _delivered_shares(mixture_build.training_dir)
    assert set(manifest.delivered_mix) == set(delivered)
    for name, share in delivered.items():
        assert manifest.delivered_mix[name] == pytest.approx(share, abs=1e-6)


# ── a rebuild replaces the dataset rather than layering over it ────────
