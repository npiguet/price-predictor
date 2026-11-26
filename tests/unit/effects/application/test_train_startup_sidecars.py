"""``train-effect-model`` refuses converted trees reconverted since its corpus
was built (finding 11.3): the records' provenance keys were joined against the
sidecars the manifest's ``sidecar_digest`` names, and another conversion's
lines would be read under the same keys without anything raising."""

from __future__ import annotations

import pytest

from effects.application.train_effect_model import run
from effects.infrastructure.converted_trees import SidecarsChangedError, SourceTrees
from tests.unit.effects.application import test_train_startup as _startup

# The startup tests' corpus, card tree, config and stopped loop.
make_record = _startup.make_record
corpus = _startup.corpus
reports = _startup.reports
config = _startup.config
started = _startup.started


def _reconvert(tree) -> None:
    """Rewrite one sidecar at another size, as a reconversion would."""
    sidecar = next(tree.rglob("*.provenance.json"))
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8")


def test_a_reconverted_tree_is_refused_before_any_step_naming_both_digests(
    corpus, config, started,
):
    recorded = corpus.load().sidecar_digest
    _reconvert(config.cards_folders[0])
    found = SourceTrees.of(config.cards_folders).converted_sidecar_digest()
    with pytest.raises(SidecarsChangedError) as excinfo:
        run(config)
    assert recorded in str(excinfo.value)
    assert found in str(excinfo.value)
    assert started == []


def test_the_trees_the_corpus_was_built_against_are_accepted(config, started):
    run(config)
    assert len(started) == 1
