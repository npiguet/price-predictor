"""Classes and strata the build refuses, and unresolved events it counts."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from effects.application.corpus_report import check_refusals as _check_refusals
from effects.application.corpus_report import check_samples as _check_samples
from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    CombatPayload,
)
from effects.infrastructure.shard_io import (
    read_records,
)
from tests.unit.effects.application.build_corpus_support import (
    _MIXTURE,
    _combat,
    _tallies,
)
from tests.unit.effects.application.test_build_corpus_mixture import (
    a_mixture_corpus,  # noqa: F401 (a fixture)
)


def _unresolved_event() -> Event:
    return Event(
        type=EventType.DAMAGE_DEALT, subjects=("P0",), attributed_to="unresolved",
    )


def test_a_combat_record_with_an_unresolved_event_reaches_training(tmp_path, a_corpus):
    """Nothing resolves in a damage step, so the collector stamps every combat
    event ``unresolved`` by design and the cause lives in ``cause``. Judged by
    the withdrawn unattributed rule the class was 98.9% refused on the real
    corpus — a statement about the rule, not about the records.
    """
    corpus = a_corpus.with_extra_records([
        replace(
            _combat("g-clean-1.combat", "g-clean-1"),
            payload=CombatPayload(attackers=("E0",), events=(_unresolved_event(),)),
        ),
    ])
    store = corpus.build(tmp_path / "out")

    trained = {r.record_id for r in read_records(store.training_dir)}
    assert "g-clean-1.combat" in trained


# ── C1 guard: a whole class refused stops the build ───────────────────


def test_a_class_refused_past_the_share_stops_the_build(tmp_path, a_corpus):
    """A class the quality rules refuse wholesale is a rule bug or an
    unpatched checkout, not a corpus to train on. Written silently it is a
    dataset missing a sampling class, which the trainer reports as a class the
    corpus happens not to hold.
    """
    corpus = a_corpus.with_extra_records([
        replace(
            _combat(f"g-clean-1.flood-{n}", "g-clean-1"),
            payload=CombatPayload(
                attackers=("E0",),
                events=tuple(
                    Event(type=EventType.DAMAGE_DEALT, subjects=("P0",),
                          attributed_to="root")
                    for _ in range(100)
                ),
            ),
        )
        for n in range(5)
    ])

    with pytest.raises(BuildCorpusError, match="combat") as raised:
        corpus.build(tmp_path / "out")

    message = str(raised.value)
    assert "unresolved" in message and "degraded" in message


def test_an_empty_stratum_sample_is_refused(tmp_path):
    with pytest.raises(BuildCorpusError, match="card-disjoint"):
        _check_samples({"card-disjoint": 0, "game-disjoint": 12}, size=2048)


def test_check_samples_passes_when_every_stratum_has_records():
    _check_samples({"card-disjoint": 4, "game-disjoint": 12}, size=2048)


def test_check_samples_says_nothing_when_no_sample_was_asked_for():
    _check_samples({"card-disjoint": 0, "game-disjoint": 0}, size=0)


def test_a_stratum_whose_sample_comes_out_empty_stops_the_build(
    tmp_path, a_mixture_corpus,  # noqa: F811
):
    """The mixture names neither resolution class and the card-disjoint
    stratum holds nothing else, so its sample is empty. Shipped, the trainer
    validates ``nan`` on the stratum that selects the best checkpoint.

    A game-disjoint share places games in that stratum, which a build asking
    for a sample refuses to leave empty before it writes anything.
    """
    with pytest.raises(BuildCorpusError, match="card-disjoint validation sample is empty"):
        a_mixture_corpus.build(tmp_path / "curated", **{
            **_MIXTURE, "validation_sample": 8, "game_disjoint_share": 0.5,
        })


def test_a_kept_record_with_an_unresolved_event_is_counted_not_refused(
    tmp_path, a_corpus,
):
    """FR-148: unattributed events are a watch statistic in the manifest.

    The refusal that used to be here threw away an eighth of the resolution
    outcomes on the real corpus, the "died" zone outcome among them. The
    count is what replaces it: a rising share is a statement about the
    collector's attribution, not about the corpus.
    """
    # In ``g-tainted`` (card-disjoint), which is admitted whole: the training
    # output is subsampled to the eight-class mixture and this fixture holds
    # two of the eight, so a training record is not a reliable place to look
    # for one particular id.
    corpus = a_corpus.with_extra_records([
        a_corpus.resolution(
            "unattributed-1", game="g-tainted",
            events=(_unresolved_event(),),
        ),
    ])
    store = corpus.build(tmp_path / "out")

    manifest = store.load()
    assert manifest.unattributed_records >= 1
    assert "unattributed-events" not in manifest.quality_dropped
    kept = {r.record_id for r in read_records(store.card_disjoint_dir)}
    assert "unattributed-1" in kept


def test_a_class_lost_to_no_acting_text_is_told_to_reconvert():
    """The third cause the refusal guard names, beside a degraded checkout and
    a rule that does not fit: sidecars written before the converter claimed
    the traits whose text its lines carry. An operator reading only "unpatched
    checkout" would go looking at Forge, and the corpus is the thing to fix.
    """
    tallies = _tallies(
        read=Counter({"resolution-effect": 10}),
        refused_by_class=Counter({"resolution-effect": 90}),
        quality_dropped=Counter({"no-acting-text": 88, "no-ability": 2}),
    )

    with pytest.raises(BuildCorpusError) as raised:
        _check_refusals(*tallies)

    message = str(raised.value)
    assert "no-acting-text" in message
    assert "reconvert" in message


def test_a_class_that_keeps_most_of_its_records_raises_nothing():
    """The guard is for a class lost whole, not for the few percent the rules
    are there to take."""
    _check_refusals(*_tallies(
        read=Counter({"resolution-effect": 95}),
        refused_by_class=Counter({"resolution-effect": 5}),
        quality_dropped=Counter({"no-acting-text": 5}),
    ))


# ── selection, placement and refusals (T036, T069, T080, T083) ─────
