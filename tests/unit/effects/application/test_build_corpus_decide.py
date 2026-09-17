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
indexed by that same rendered string, never by the bare label.
``held_out_text_games`` is keyed the same way: its keys are resolved to a
script text and matched against the holdout, so a bare label there would not
parse either.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from effects.application.build_corpus import (
    BuildCorpusConfig,
    Survey,
    ability_key,
    decide,
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
#: Same script file as ``reprint-a``, a trait kind that sidecar never names:
#: the mismatch answer, as opposed to the two no-text ones.
MISMATCH_KEY = ability_key(SimpleNamespace(ability=(
    ProvenanceKey("cardsfolder/x/reprint_a.txt", 0, "trigger", 0),
)))
#: The one key ``fake_sidecars`` resolves to a text the holdout names.
HELD_OUT_KEY = _rendered_key("held-out-text")
HELD_OUT_TEXTS = frozenset({"held-out-text"})


@pytest.fixture
def fake_sidecars() -> FakeSidecarCache:
    """``reprint-a``/``reprint-b`` resolve to one text; ``held-out-text`` to
    its own; ``unknown-key`` names a script no other key here registers, so
    ``line_for`` reads it as no text, the same way a script the converted
    corpus never held would.
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

    ``decide`` now checks every ``key_heaps`` entry's cap against
    ``config.text_cap`` before it merges anything (a survey built at one cap
    cannot be decided against another). Any test whose survey carries a
    ``key_records``/``key_hashes``/``key_games`` entry — so its ``key_heaps``
    is non-empty — MUST pass ``heap_cap`` equal to whatever ``text_cap`` the
    ``a_config`` it calls ``decide`` with uses, or ``decide`` raises.

    ``key_hashes`` (``{key: iterable of ints}``) is not a ``Survey`` field: it
    builds ``key_heaps`` by offering each hash into a fresh
    ``CapHeap(heap_cap)`` per key. Every key named by ``key_hashes`` or
    ``key_games`` also gets an entry in the returned ``key_records`` (zero,
    absent a ``key_records`` entry of its own) and ``key_heaps`` (empty,
    absent matching ``key_hashes``), because ``decide`` discovers which keys
    to fold by walking ``key_records`` alone.

    ``class_records`` defaults to one token record of one class rather than to
    nothing: ``decide`` always computes class targets from what the cap leaves
    available, and an empty availability map makes ``class_targets`` raise — a
    scenario no test here is about. ``class_key_records`` (sampling class ->
    rendered key -> records) defaults to empty, which means every record of
    every class has no acting line and so is never capped.
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
    class_key_records = {
        name: Counter(per_key)
        for name, per_key in overrides.pop("class_key_records", {}).items()
    }

    base = dict(
        shards=(),
        records=sum(key_records.values()),
        key_games=key_games,
        key_records=key_records,
        key_heaps=key_heaps,
        class_records=class_records,
        class_key_records=class_key_records,
        held_out_text_games={},
        held_out_games=frozenset(),
        games=frozenset(),
    )
    base.update(overrides)
    return Survey(**base)


def a_config(**overrides) -> BuildCorpusConfig:
    return BuildCorpusConfig(records_dir=Path("."), **overrides)


def test_two_keys_folding_to_one_text_share_a_threshold(fake_sidecars):
    # Disjoint, non-overlapping hash ranges: a threshold computed PER KEY
    # (rather than per text, the regression this test is named against) would
    # give REPRINT_A 99 and REPRINT_B 1099 — different. Only a genuine merge
    # into one per-text heap gives both keys the same threshold, and that
    # threshold has to be the smallest 100 of the whole 600-value union, i.e.
    # 99, not either key's own unmerged value.
    survey = a_survey(
        key_records={REPRINT_A: 300, REPRINT_B: 300},
        key_hashes={REPRINT_A: range(0, 300), REPRINT_B: range(1000, 1300)},
        heap_cap=100,
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(text_cap=100))

    assert decisions.thresholds[REPRINT_A] == decisions.thresholds[REPRINT_B] == 99


def test_rarity_counts_games_over_the_whole_corpus(fake_sidecars):
    survey = a_survey(key_games={REPRINT_A: {1, 2}, REPRINT_B: {2, 3}})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config())

    # Both fold to one text, and game 2 is one game: three distinct games.
    assert decisions.rarity["deals 3 damage"] == 3


def test_a_text_under_the_cap_gets_no_threshold(fake_sidecars):
    survey = a_survey(key_records={REPRINT_A: 5}, heap_cap=100)
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(text_cap=100))

    assert decisions.thresholds[REPRINT_A] is None


def test_a_key_no_sidecar_can_read_gets_no_threshold_and_no_rarity(fake_sidecars):
    survey = a_survey(key_records={UNKNOWN_KEY: 500}, heap_cap=10)
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(text_cap=10))

    assert UNKNOWN_KEY not in decisions.thresholds


def test_a_sidecar_mismatch_propagates_out_of_the_fold(fake_sidecars):
    """The contract's fail-loudly case, pinned at ``decide``'s own fold.

    ``_text_of_rendered_key`` has no ``except``, on purpose: a key the sidecar
    exists for and does not describe means the cards were reconverted between
    collection and training, and every downstream number computed from a
    silent "no text" would be wrong in a way nothing reports. A ``try`` added
    there for tidiness would restore exactly the bug the branch removed, so
    the raise is asserted rather than left to inspection.
    """
    survey = a_survey(key_records={MISMATCH_KEY: 5}, heap_cap=100)

    with pytest.raises(KeyError, match="neither the lines nor the dropped_keys"):
        decide(survey, sidecars=fake_sidecars, surface="script",
               held_out_texts=HELD_OUT_TEXTS,
               config=a_config(text_cap=100))


def test_a_survey_built_at_a_different_text_cap_raises(fake_sidecars):
    # a_survey's heaps are built at cap 50 (via heap_cap); a_config asks for
    # 100. decide() must fail loudly rather than let the thresholds it
    # computes silently disagree with the text_cap the manifest will record.
    survey = a_survey(key_records={REPRINT_A: 5}, heap_cap=50)

    with pytest.raises(ValueError, match="text_cap"):
        decide(survey, sidecars=fake_sidecars, surface="script",
               held_out_texts=HELD_OUT_TEXTS,
               config=a_config(text_cap=100))


def test_games_split_into_the_two_strata_and_training(fake_sidecars):
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(10)}),
        held_out_games=frozenset({"g0", "g1"}),
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(game_disjoint_target=3))

    assert decisions.card_disjoint <= frozenset({"g0", "g1"})
    assert len(decisions.game_disjoint) == 3
    assert not (decisions.card_disjoint & decisions.game_disjoint)
    assert not (decisions.game_disjoint & frozenset({"g0", "g1"}))


def test_the_game_disjoint_draw_is_seeded(fake_sidecars):
    survey = a_survey(games=frozenset({f"g{i}" for i in range(50)}))
    first = decide(survey, sidecars=fake_sidecars, surface="script",
                   held_out_texts=HELD_OUT_TEXTS,
                   config=a_config(game_disjoint_target=5, seed=7))
    again = decide(survey, sidecars=fake_sidecars, surface="script",
                   held_out_texts=HELD_OUT_TEXTS,
                   config=a_config(game_disjoint_target=5, seed=7))

    assert first.game_disjoint == again.game_disjoint


def test_a_game_disjoint_target_larger_than_the_corpus_takes_what_there_is(fake_sidecars):
    survey = a_survey(games=frozenset({"g0", "g1"}))
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(game_disjoint_target=100))

    assert decisions.game_disjoint == frozenset({"g0", "g1"})


def _many_text_games(count: int, *, carry_held_out: int | None = None) -> dict[str, set[str]]:
    """``held_out_text_games`` for ``count`` games that each carry many texts.

    Every game carries a text of its own that no sidecar can read — so
    ``decide`` never folds it to a held-out text — plus, for the first
    ``carry_held_out`` games (all of them by default), the one text
    ``fake_sidecars`` resolves into the holdout.

    The private per-game texts are the whole point: a real game carries tens
    of distinct ability texts, so a rule that admitted a game while *any*
    text it carries was under cap never declined one, because a game's own
    private texts sit at zero forever.
    """
    games = [f"g{i}" for i in range(count)]
    carriers = games if carry_held_out is None else games[:carry_held_out]
    out: dict[str, set[str]] = {HELD_OUT_KEY: set(carriers)}
    for index, game in enumerate(games):
        out[_rendered_key(f"filler-{index}")] = {game}
    return out


def test_the_card_disjoint_cap_admits_exactly_its_cap_of_games(fake_sidecars):
    # Twenty games, each carrying the held-out text plus a private text of its
    # own. Under `any(under cap)` every one of the twenty was admitted, because
    # each game's private text is always at zero; under `all(under cap)` only
    # the cap's worth of games gets in, which is what FR-142 says.
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(20)}),
        held_out_games=frozenset({f"g{i}" for i in range(20)}),
        held_out_text_games=_many_text_games(20),
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(card_disjoint_text_cap=2))

    assert len(decisions.card_disjoint) == 2


def test_a_text_the_holdout_does_not_name_is_not_tallied(fake_sidecars):
    # Twenty games each carrying a private text; only the first five also carry
    # the held-out one. With a cap of 5 the five carriers are admitted and the
    # other fifteen are not: the cap counts held-out texts and nothing else, so
    # a game holding nothing gate 1 measures earns no place in the stratum.
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(20)}),
        held_out_games=frozenset({f"g{i}" for i in range(20)}),
        held_out_text_games=_many_text_games(20, carry_held_out=5),
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(card_disjoint_text_cap=5))

    assert decisions.card_disjoint == frozenset({f"g{i}" for i in range(5)})


def test_the_card_disjoint_draw_is_seeded(fake_sidecars):
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(20)}),
        held_out_games=frozenset({f"g{i}" for i in range(20)}),
        held_out_text_games=_many_text_games(20),
    )
    first = decide(survey, sidecars=fake_sidecars, surface="script",
                   held_out_texts=HELD_OUT_TEXTS,
                   config=a_config(card_disjoint_text_cap=2, seed=7))
    again = decide(survey, sidecars=fake_sidecars, surface="script",
                   held_out_texts=HELD_OUT_TEXTS,
                   config=a_config(card_disjoint_text_cap=2, seed=7))

    assert first.card_disjoint == again.card_disjoint


def test_capped_class_records_never_exceed_what_the_corpus_holds(fake_sidecars):
    survey = a_survey(
        key_records={REPRINT_A: 100},
        class_records={"rewrite": 100, "combat": 50},
        class_key_records={"rewrite": {REPRINT_A: 100}},
        heap_cap=10,
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(text_cap=10))

    for name, held in survey.class_records.items():
        assert decisions.capped_class_records[name] <= held


def test_class_targets_are_computed_from_the_capped_counts_not_the_raw_ones(fake_sidecars):
    # class_records is what the corpus holds before the per-text cap; a cap of
    # 10 over one text carrying all 100 of them leaves 10. class_targets has to
    # be sized off that: fed the raw count instead, it would target more
    # training records than the cap is about to leave available, and the
    # manifest would advertise a mixture the corpus cannot supply.
    survey = a_survey(
        key_records={REPRINT_A: 100},
        class_records={"rewrite": 100},
        class_key_records={"rewrite": {REPRINT_A: 100}},
        heap_cap=10,
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(text_cap=10, class_mix={"rewrite": 1.0}))

    assert decisions.class_targets["rewrite"] == 10


def test_the_capped_estimate_counts_a_text_once_over_the_whole_corpus(fake_sidecars):
    # Two keys folding to one text, 300 records each, at a cap of 100. The cap
    # is per text and corpus-wide, so 100 records of that text survive in
    # total. Estimating it per key — or, as the survey used to, per shard —
    # gives 200 here and `shard count x cap` on a real corpus, and every class
    # the cap touches is then admitted at a rate sized against records that do
    # not exist.
    survey = a_survey(
        key_records={REPRINT_A: 300, REPRINT_B: 300},
        class_records={"rewrite": 600},
        class_key_records={"rewrite": {REPRINT_A: 300, REPRINT_B: 300}},
        heap_cap=100,
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(text_cap=100))

    assert decisions.capped_class_records["rewrite"] == 100


def test_a_capped_texts_survivors_are_split_over_the_classes_carrying_it(fake_sidecars):
    # One text, 400 records: 300 rewrite and 100 trigger. The cap admits the
    # 100 smallest hashes of the text, and a record's hash is computed from its
    # id alone, so the survivors fall in the classes' own proportions: 75 and
    # 25, not 100 each.
    survey = a_survey(
        key_records={REPRINT_A: 400},
        class_records={"rewrite": 300, "trigger": 100},
        class_key_records={"rewrite": {REPRINT_A: 300}, "trigger": {REPRINT_A: 100}},
        heap_cap=100,
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(text_cap=100))

    assert decisions.capped_class_records == {"rewrite": 75, "trigger": 25}


def test_a_class_with_no_acting_ability_is_never_capped(fake_sidecars):
    # `combat` and `playability-legality` carry no acting line (FR-087), so no
    # text caps them and their availability is exact however small the cap is.
    survey = a_survey(class_records={"combat": 500})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       held_out_texts=HELD_OUT_TEXTS,
                       config=a_config(text_cap=1))

    assert decisions.capped_class_records["combat"] == 500
