"""``decide``: fold the survey into thresholds, targets and the two strata.

The survey (``run_survey``) keys everything by provenance key and never
resolves a key to ability text (see ``test_build_corpus_survey.py``); these
tests build ``Survey`` objects directly and exercise the fold that happens
once, here, against a fake ``SidecarCache``.

A survey key is a *rendered* provenance key — ``ability_key()``'s output,
``script_file|face|trait_kind|index`` (``;``-joined for several keys) — and
``decide`` parses it back with ``parse_ability_key`` before resolving it
against the sidecars. A bare mnemonic like ``"reprint-a"`` does not parse
(there is no ``|`` in it), so every label these tests need resolved is put
through ``_rendered_key`` first, which renders a real single-key string
carrying the label as its script file; ``decisions.thresholds`` is then
indexed by that same rendered string, never by the bare label. A label that
is only ever used as a fallback text name (``held_out_text_games`` in the
card-disjoint test) needs no such wrapping, because ``decide`` never parses
those keys — it looks them up in the key->text table built from
``key_records`` and falls back to the key itself unparsed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from effects.application.build_corpus import (
    BuildCorpusConfig, Survey, ability_key, decide,
)
from effects.domain.corpus_curation import CapHeap
from effects.domain.effect_model import CLASS_REWRITE
from effects.domain.provenance import ProvenanceKey, SidecarLine
from tests.unit.effects.application.test_ability_text import FakeSidecarCache

_UNSET = object()


def _provenance_key(label: str) -> ProvenanceKey:
    """A single, deterministic ``ProvenanceKey`` naming ``label``."""
    return ProvenanceKey(f"cardsfolder/x/{label.replace('-', '_')}.txt", 0, "spell", 0)


def _rendered_key(label: str) -> str:
    """``label``, rendered the way a real survey key would be.

    ``decide`` parses every survey key with ``parse_ability_key``, which
    expects ``ability_key()``'s ``script_file|face|trait_kind|index`` shape.
    This keeps the label readable (it becomes the script file) while
    producing something that round-trips through that parser.
    """
    return ability_key(SimpleNamespace(ability=(_provenance_key(label),)))


REPRINT_A = _rendered_key("reprint-a")
REPRINT_B = _rendered_key("reprint-b")
UNKNOWN_KEY = _rendered_key("unknown-key")


@pytest.fixture
def fake_sidecars() -> FakeSidecarCache:
    """``reprint-a``/``reprint-b`` resolve to one text; ``held-out-text`` to
    its own; ``unknown-key`` is left unregistered so ``line_for`` raises
    ``KeyError`` for it, the same way a real corpus/sidecar mismatch would.
    """

    def _line(text: str, key: ProvenanceKey) -> SidecarLine:
        return SidecarLine(
            line_index=0, line_kind="spell", provenance=(key,), script_text=text,
        )

    key_a = _provenance_key("reprint-a")
    key_b = _provenance_key("reprint-b")
    key_held_out = _provenance_key("held-out-text")
    return FakeSidecarCache(lines={
        key_a: _line("deals 3 damage", key_a),
        key_b: _line("deals 3 damage", key_b),
        key_held_out: _line("held-out-text", key_held_out),
    })


def a_survey(*, heap_cap: int = 200, **overrides) -> Survey:
    """A real ``Survey`` with sensible defaults.

    ``key_hashes`` (``{key: iterable of ints}``) is not a ``Survey`` field: it
    builds ``key_heaps`` by offering each hash into a fresh
    ``CapHeap(heap_cap)`` per key. Every key named by ``key_hashes`` or
    ``key_games`` also gets an entry in the returned ``key_records`` (zero,
    absent a ``key_records`` entry of its own) and ``key_heaps`` (empty,
    absent matching ``key_hashes``), because ``decide`` discovers which keys
    to fold by walking ``key_records`` alone.

    ``class_records`` defaults to one token record of one class rather than to
    nothing: ``decide`` always computes class targets from
    ``class_capped_records``, and an empty availability map makes
    ``class_targets`` raise — a scenario no test here is about.
    ``class_capped_records`` defaults to a copy of whatever ``class_records``
    ends up being, absent its own override.
    """
    key_records = Counter(overrides.pop("key_records", {}))
    key_hashes: dict[str, object] = overrides.pop("key_hashes", {})
    key_games = {
        key: set(games) for key, games in overrides.pop("key_games", {}).items()
    }
    for key in set(key_hashes) | set(key_games):
        key_records.setdefault(key, 0)

    key_heaps: dict[str, CapHeap] = {}
    for key in key_records:
        heap = CapHeap(heap_cap)
        for value in key_hashes.get(key, ()):
            heap.offer(value)
        key_heaps[key] = heap

    given_class_records = overrides.pop("class_records", _UNSET)
    class_records = Counter(
        {CLASS_REWRITE: 1} if given_class_records is _UNSET else given_class_records
    )
    class_capped_records = Counter(
        overrides.pop("class_capped_records", dict(class_records))
    )

    base = dict(
        shards=(),
        records=sum(key_records.values()),
        key_games=key_games,
        key_records=key_records,
        key_heaps=key_heaps,
        class_records=class_records,
        class_capped_records=class_capped_records,
        held_out_text_games={},
        held_out_games=frozenset(),
        games=frozenset(),
    )
    base.update(overrides)
    return Survey(**base)


def a_config(**overrides) -> BuildCorpusConfig:
    return BuildCorpusConfig(records_dir=Path("."), **overrides)


def test_two_keys_folding_to_one_text_share_a_threshold(fake_sidecars):
    survey = a_survey(key_records={REPRINT_A: 300, REPRINT_B: 300},
                      key_hashes={REPRINT_A: range(300), REPRINT_B: range(300)})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(text_cap=100))

    assert decisions.thresholds[REPRINT_A] == decisions.thresholds[REPRINT_B]


def test_rarity_counts_games_over_the_whole_corpus(fake_sidecars):
    survey = a_survey(key_games={REPRINT_A: {1, 2}, REPRINT_B: {2, 3}})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config())

    # Both fold to one text, and game 2 is one game: three distinct games.
    assert decisions.rarity["deals 3 damage"] == 3


def test_a_text_under_the_cap_gets_no_threshold(fake_sidecars):
    survey = a_survey(key_records={REPRINT_A: 5})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(text_cap=100))

    assert decisions.thresholds[REPRINT_A] is None


def test_a_key_no_sidecar_can_read_gets_no_threshold_and_no_rarity(fake_sidecars):
    survey = a_survey(key_records={UNKNOWN_KEY: 500})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(text_cap=10))

    assert UNKNOWN_KEY not in decisions.thresholds


def test_games_split_into_the_two_strata_and_training(fake_sidecars):
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(10)}),
        held_out_games=frozenset({"g0", "g1"}),
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(game_disjoint_target=3))

    assert decisions.card_disjoint <= frozenset({"g0", "g1"})
    assert len(decisions.game_disjoint) == 3
    assert not (decisions.card_disjoint & decisions.game_disjoint)
    assert not (decisions.game_disjoint & frozenset({"g0", "g1"}))


def test_the_game_disjoint_draw_is_seeded(fake_sidecars):
    survey = a_survey(games=frozenset({f"g{i}" for i in range(50)}))
    first = decide(survey, sidecars=fake_sidecars, surface="script",
                   config=a_config(game_disjoint_target=5, seed=7))
    again = decide(survey, sidecars=fake_sidecars, surface="script",
                   config=a_config(game_disjoint_target=5, seed=7))

    assert first.game_disjoint == again.game_disjoint


def test_a_game_disjoint_target_larger_than_the_corpus_takes_what_there_is(fake_sidecars):
    survey = a_survey(games=frozenset({"g0", "g1"}))
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(game_disjoint_target=100))

    assert decisions.game_disjoint == frozenset({"g0", "g1"})


def test_the_card_disjoint_cap_stops_admitting_games_once_texts_are_covered(fake_sidecars):
    # Twenty games all carrying the one held-out text; a cap of 2 admits few.
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(20)}),
        held_out_games=frozenset({f"g{i}" for i in range(20)}),
        held_out_text_games={"held-out-text": {f"g{i}" for i in range(20)}},
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(card_disjoint_text_cap=2))

    assert 0 < len(decisions.card_disjoint) < 20


def test_capped_class_records_never_exceed_what_the_corpus_holds(fake_sidecars):
    survey = a_survey(class_records={"rewrite": 100, "combat": 50})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(text_cap=10))

    for name, held in survey.class_records.items():
        assert decisions.capped_class_records[name] <= held
