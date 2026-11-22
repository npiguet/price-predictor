"""The build's outputs are what its manifest says (findings 11.4, 14.3).

The manifest lists every shard each stratum holds, with the records the
write pass put in it, so a reader can tell a stale or missing shard from the
dataset the manifest describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.corpus_report import recorded_path
from effects.application.corpus_survey import SurveyedRecord
from effects.domain.corpus_manifest import CuratedShard
from effects.domain.corpus_plan import BuildCorpusError, Cell
from effects.domain.rule_detectors import RuleSource
from effects.domain.sampling import SamplingClass
from effects.domain.strata import OUTPUT_STRATA
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import iter_shards, read_shard
from price_predictor.infrastructure.forge_classpath import project_root
from tests.unit.effects.application.build_corpus_support import (
    CorpusFixture,
    _write_corpus,
)
from tests.unit.effects.application.test_build_corpus_decide import (
    REPRINT_A,
    REWRITE,
    SurveySpec,
    _decide,
    a_survey,
    fake_sidecars,  # noqa: F401 (a fixture)
)
from tests.unit.effects.application.test_build_corpus_variant_tree import (
    a_corpus_with_variants,  # noqa: F401 (a fixture)
)

#: No game-disjoint draw: the stratum comes out empty.
_NO_GAME_DISJOINT = dict(game_disjoint_share=0.0, game_disjoint_keyword_share=0.0)


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


# ── F09b: the empty-stratum refusal, the merged supply, the digest's paths ──


def test_a_second_decide_over_one_survey_gives_the_same_plan(fake_sidecars):  # noqa: F811
    """``decide`` folds the training candidates into a copy of the survey's
    supply. Merged by reference, the survey's own heaps took the candidates'
    hashes, and a second ``decide`` counted them twice (finding 14.3)."""
    cell = Cell(REWRITE, family="Draw")
    survey = a_survey(SurveySpec(
        cells={(cell, REPRINT_A): (20, range(100, 120))}, heap_cap=10,
        games=frozenset({"g1"}),
    ))
    survey.totals.candidates.extend(
        SurveyedRecord(game="g1", record_hash=value, klass=SamplingClass(REWRITE), cell=cell,
                       keys=REPRINT_A, tags=(), source=RuleSource.FORGE_AI)
        for value in range(5)
    )
    surveyed = survey.totals.supply.heaps[cell, REPRINT_A].values()

    def plan():
        return _decide(survey, fake_sidecars, text_cap=10, class_mix={REWRITE: 1.0},
                       game_disjoint_share=0.0, game_disjoint_keyword_share=0.0)

    first, second = plan(), plan()

    assert second.selection == first.selection
    assert survey.totals.supply.heaps[cell, REPRINT_A].values() == surveyed


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_an_empty_game_disjoint_stratum_with_a_sample_asked_for_is_refused_before_the_clear(
    tmp_path, a_corpus,
):
    """The trainer validates on the stratum's sample, which an empty stratum
    leaves empty; told only after the write pass, the old dataset was already
    cleared (finding 14.3). Refused at the decision, it stays byte for byte."""
    out = tmp_path / "curated"
    a_corpus.build(out, validation_sample=8)
    before = _files(out)

    with pytest.raises(BuildCorpusError, match="game-disjoint"):
        a_corpus.build(out, validation_sample=8, **_NO_GAME_DISJOINT)

    assert _files(out) == before


def test_an_empty_game_disjoint_stratum_builds_when_no_sample_is_asked_for(tmp_path, a_corpus):
    manifest = a_corpus.build(tmp_path / "curated", validation_sample=0, **_NO_GAME_DISJOINT).load()
    assert manifest.game_disjoint_games == ()


def test_one_build_typed_with_either_slash_has_one_digest(
    tmp_path, a_corpus_with_variants,  # noqa: F811
):
    """The digest a checkpoint pins holds ``vocab_path`` and ``variant_scripts``:
    as typed, one file spelled two ways pinned two digests (finding 14.3)."""
    fixture, variant_root = a_corpus_with_variants
    vocab = Path(fixture.config(tmp_path).vocab_path)
    native = fixture.build(
        tmp_path / "native", vocab_path=str(vocab), variant_scripts=str(Path(variant_root)),
    ).load()
    posix = fixture.build(
        tmp_path / "posix", vocab_path=vocab.as_posix(),
        variant_scripts=Path(variant_root).as_posix(),
    ).load()

    assert native.vocab_path == posix.vocab_path == vocab.resolve().as_posix()
    assert native.variant_scripts == Path(variant_root).resolve().as_posix()
    assert native.digest() == posix.digest()


def test_a_path_under_the_project_root_is_recorded_relative_to_it(monkeypatch):
    """Typed relative to the working directory or absolute, and with either
    slash, a path inside the checkout is recorded one way."""
    root = project_root()
    monkeypatch.chdir(root / "src")
    expected = "models/effects/gen3/vocab.txt"

    assert recorded_path("../models/effects/gen3/vocab.txt") == expected
    assert recorded_path(str(root / "models" / "effects" / "gen3" / "vocab.txt")) == expected
    assert recorded_path((root / expected).as_posix()) == expected


def test_a_path_outside_the_project_root_is_recorded_absolute(tmp_path):
    outside = tmp_path / "vocab.txt"
    assert recorded_path(str(outside)) == outside.resolve().as_posix()
