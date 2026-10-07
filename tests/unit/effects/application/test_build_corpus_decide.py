"""``decide``: fold the survey into copy plans, class budgets and the two strata.

The survey (``run_survey``) keys everything by provenance key and never
resolves a key to ability text (see ``test_build_corpus_survey.py``); these
tests build ``Survey`` objects directly and exercise the fold that happens
once, here, against a fake ``SidecarCache``.

A survey key is a *rendered* provenance key — ``ability_key()``'s output,
``script_file|face|trait_kind|index`` — and ``decide`` parses it back with
``parse_ability_key`` before resolving it against the sidecars. A bare
mnemonic does not parse, so every label these tests need resolved is put
through ``_rendered_key`` first. ``held_out_text_games`` is keyed the same way.

Gen-2 selection (FR-049) is a cell-by-cell plan: a class budget split equally
across rule families, a family's across outcome signatures, the legality class
across its real and what-if halves, and a cell's across its ability texts, each
level capped by what its members can supply with ``--reuse-cap`` repeats and
``--text-cap`` per text per cell.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from effects.application.build_corpus import (
    BuildCorpusConfig,
    Cell,
    Survey,
    ability_key,
    decide,
)
from effects.domain.corpus_curation import CapHeap, game_disjoint_draw
from effects.domain.provenance import ProvenanceKey, SidecarLine
from tests.unit.effects.application.test_ability_text import FakeSidecarCache

#: The survey's heap size under ``--text-cap 0``: every hash is kept.
NO_CAP = sys.maxsize
REWRITE = "rewrite"
EFFECT = "resolution-effect"
LEGALITY = "playability-legality"


def _provenance_key(label: str) -> ProvenanceKey:
    """A single, deterministic ``ProvenanceKey`` naming ``label``."""
    return ProvenanceKey(f"cardsfolder/x/{label.replace('-', '_')}.txt", 0, "spell", 0)


def _rendered_key(label: str) -> str:
    """``label``, rendered the way a real survey key would be."""
    return ability_key(SimpleNamespace(ability=(_provenance_key(label),)))


REPRINT_A = _rendered_key("reprint-a")
REPRINT_B = _rendered_key("reprint-b")
UNKNOWN_KEY = _rendered_key("unknown-key")
MISMATCH_KEY = ability_key(SimpleNamespace(ability=(
    ProvenanceKey("cardsfolder/x/reprint_a.txt", 0, "trigger", 0),
)))
HELD_OUT_KEY = _rendered_key("held-out-text")
HELD_OUT_TEXTS = frozenset({"held-out-text"})
#: Texts for the family tests, one key each.
TEXT_KEYS = {f"t{i}": _rendered_key(f"t{i}") for i in range(1, 31)}


@pytest.fixture
def fake_sidecars() -> FakeSidecarCache:
    """``reprint-a``/``reprint-b`` resolve to one text; every other label to
    its own; ``unknown-key`` names a script no sidecar describes."""

    def _line(text: str, key: ProvenanceKey) -> SidecarLine:
        return SidecarLine(
            line_index=0, line_kind="spell", provenance=(key,), script_text=text,
        )

    lines = {}
    for label, text in (
        ("reprint-a", "deals 3 damage"), ("reprint-b", "deals 3 damage"),
        ("held-out-text", "held-out-text"),
        *((label, f"text {label}") for label in TEXT_KEYS),
    ):
        key = _provenance_key(label)
        lines[key] = _line(text, key)
    return FakeSidecarCache(lines=lines)


def a_survey(
    *, heap_cap: int = 200, cells: dict | None = None,
    keyless: dict | None = None, **overrides,
) -> Survey:
    """A real ``Survey`` with sensible defaults.

    ``cells`` maps ``(Cell, rendered key)`` to a record count, or to
    ``(count, hashes)``; without explicit hashes the key's records get the
    hashes ``0 .. count-1`` offset by the key's position, so two keys never
    collide. ``keyless`` maps a ``Cell`` to its records with no acting key.
    Absent both, the survey holds one keyless rewrite record, so the class
    mixture has a class to size against — an empty availability map makes
    ``class_targets`` raise, a scenario no test here is about.
    """
    cells = dict(cells or {})
    keyless = Counter(keyless or ({} if cells else {Cell(REWRITE): 1}))
    cell_key_records: Counter = Counter()
    heaps: dict = {}
    for position, (slot, value) in enumerate(sorted(cells.items())):
        count, hashes = value if isinstance(value, tuple) else (
            value, range(position * 10_000, position * 10_000 + value),
        )
        cell_key_records[slot] = count
        heap = CapHeap(heap_cap)
        for item in hashes:
            heap.offer(item)
        heaps[slot] = heap
    key_records = Counter(overrides.pop("key_records", {}))
    for (_, key), count in cell_key_records.items():
        key_records[key] += count
    key_games = {
        key: set(games) for key, games in overrides.pop("key_games", {}).items()
    }
    for key in key_games:
        key_records.setdefault(key, 0)
    class_records = Counter()
    for (cell, _), count in cell_key_records.items():
        class_records[cell.klass] += count
    for cell, count in keyless.items():
        class_records[cell.klass] += count

    base = dict(
        shards=(),
        records=sum(class_records.values()),
        key_games=key_games,
        key_records=key_records,
        class_records=class_records,
        cell_key_records=cell_key_records,
        cell_key_heaps=heaps,
        cell_keyless=keyless,
        keyword_combat_games=frozenset(),
        held_out_text_games={},
        held_out_games=frozenset(),
        games=frozenset(),
    )
    base.update(overrides)
    return Survey(**base)


def a_config(**overrides) -> BuildCorpusConfig:
    overrides.setdefault("keyword_definitions", None)
    return BuildCorpusConfig(records_dir=Path("."), **overrides)


def _decide(survey, fake_sidecars, **config):
    return decide(survey, sidecars=fake_sidecars, surface="script",
                  held_out_texts=HELD_OUT_TEXTS, config=a_config(**config))


def _written(decisions, survey, cell: Cell, key: str) -> int:
    """Copies a plan writes for every hash the survey kept for ``(cell, key)``."""
    plan = decisions.plans[cell, key]
    return sum(plan.copies(value) for value in survey.cell_key_heaps[cell, key].values())


# ── the fold ────────────────────────────────────────────────────────────


def test_two_keys_folding_to_one_text_share_one_cap(fake_sidecars):
    """The cap is per text: two keys of one text share its ``--text-cap``."""
    cell = Cell(REWRITE, family="Draw")
    survey = a_survey(
        cells={(cell, REPRINT_A): 300, (cell, REPRINT_B): 300}, heap_cap=100,
    )
    decisions = _decide(survey, fake_sidecars, text_cap=100, class_mix={REWRITE: 1.0})

    assert decisions.plans[cell, REPRINT_A] == decisions.plans[cell, REPRINT_B]
    total = _written(decisions, survey, cell, REPRINT_A) + _written(
        decisions, survey, cell, REPRINT_B,
    )
    assert total == 100


def test_rarity_counts_games_over_the_whole_corpus(fake_sidecars):
    survey = a_survey(key_games={REPRINT_A: {1, 2}, REPRINT_B: {2, 3}})
    decisions = _decide(survey, fake_sidecars)

    assert decisions.rarity["deals 3 damage"] == 3


def test_a_key_no_sidecar_can_read_is_sampled_with_the_cells_keyless_records(
    fake_sidecars,
):
    cell = Cell(REWRITE, family="Draw")
    survey = a_survey(cells={(cell, UNKNOWN_KEY): 500}, heap_cap=10)
    decisions = _decide(survey, fake_sidecars, text_cap=10, class_mix={REWRITE: 1.0})

    assert (cell, UNKNOWN_KEY) not in decisions.plans
    assert decisions.keyless_plans[cell].base == 1
    assert UNKNOWN_KEY not in decisions.key_text


def test_a_sidecar_mismatch_propagates_out_of_the_fold(fake_sidecars):
    """A key the sidecar exists for and does not describe fails loudly."""
    survey = a_survey(cells={(Cell(REWRITE), MISMATCH_KEY): 5}, heap_cap=100)

    with pytest.raises(KeyError, match="neither the lines nor the dropped_keys"):
        _decide(survey, fake_sidecars, text_cap=100)


def test_a_survey_built_at_a_different_text_cap_raises(fake_sidecars):
    survey = a_survey(cells={(Cell(REWRITE), REPRINT_A): 5}, heap_cap=50)

    with pytest.raises(ValueError, match="text_cap"):
        _decide(survey, fake_sidecars, text_cap=100)


# ── the game-disjoint stratum (FR-044–046) ──────────────────────────────


def test_a_game_enters_the_stratum_by_its_own_hash(fake_sidecars):
    games = frozenset(f"g{i}" for i in range(200))
    survey = a_survey(games=games, held_out_games=frozenset({"g0", "g1"}))
    decisions = _decide(survey, fake_sidecars, game_disjoint_share=0.3,
                        game_disjoint_keyword_share=0.3)

    expected = {
        game for game in games - {"g0", "g1"} if game_disjoint_draw(game) < 0.3
    }
    assert decisions.game_disjoint == expected
    assert not decisions.game_disjoint & {"g0", "g1"}


def test_a_held_out_game_never_enters_whatever_its_hash(fake_sidecars):
    games = frozenset(f"g{i}" for i in range(50))
    survey = a_survey(games=games, held_out_games=games)
    decisions = _decide(survey, fake_sidecars, game_disjoint_share=1.0)

    assert decisions.game_disjoint == frozenset()


def test_a_keyword_combat_game_takes_the_keyword_threshold(fake_sidecars):
    """Spec Story 4 scenario 5: the higher threshold for a listed keyword."""
    games = frozenset(f"g{i}" for i in range(400))
    between = {g for g in games if 0.01 <= game_disjoint_draw(g) < 0.15}
    assert between, "the fixture needs a game between the two thresholds"
    survey = a_survey(games=games, keyword_combat_games=frozenset(between))
    decisions = _decide(survey, fake_sidecars)

    assert between <= decisions.game_disjoint
    assert decisions.keyword_threshold_games == frozenset(between)


def test_a_rebuild_over_more_games_keeps_every_placed_game(fake_sidecars):
    """SC-007: placement depends on the game alone."""
    first = _decide(a_survey(games=frozenset(f"g{i}" for i in range(300))),
                    fake_sidecars, game_disjoint_share=0.1)
    grown = _decide(a_survey(games=frozenset(f"g{i}" for i in range(900))),
                    fake_sidecars, game_disjoint_share=0.1)

    assert first.game_disjoint
    assert first.game_disjoint <= grown.game_disjoint


# ── the card-disjoint stratum (unchanged) ──────────────────────────────


def _many_text_games(count: int, *, carry_held_out: int | None = None) -> dict[str, set[str]]:
    """``held_out_text_games`` for ``count`` games that each carry many texts."""
    games = [f"g{i}" for i in range(count)]
    carriers = games if carry_held_out is None else games[:carry_held_out]
    out: dict[str, set[str]] = {HELD_OUT_KEY: set(carriers)}
    for index, game in enumerate(games):
        out[_rendered_key(f"filler-{index}")] = {game}
    return out


def test_the_card_disjoint_cap_admits_exactly_its_cap_of_games(fake_sidecars):
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(20)}),
        held_out_games=frozenset({f"g{i}" for i in range(20)}),
        held_out_text_games=_many_text_games(20),
    )
    decisions = _decide(survey, fake_sidecars, card_disjoint_text_cap=2)

    assert len(decisions.card_disjoint) == 2


def test_a_text_the_holdout_does_not_name_is_not_tallied(fake_sidecars):
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(20)}),
        held_out_games=frozenset({f"g{i}" for i in range(20)}),
        held_out_text_games=_many_text_games(20, carry_held_out=5),
    )
    decisions = _decide(survey, fake_sidecars, card_disjoint_text_cap=5)

    assert decisions.card_disjoint == frozenset({f"g{i}" for i in range(5)})


def test_the_card_disjoint_draw_is_seeded(fake_sidecars):
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(20)}),
        held_out_games=frozenset({f"g{i}" for i in range(20)}),
        held_out_text_games=_many_text_games(20),
    )
    first = _decide(survey, fake_sidecars, card_disjoint_text_cap=2, seed=7)
    again = _decide(survey, fake_sidecars, card_disjoint_text_cap=2, seed=7)

    assert first.card_disjoint == again.card_disjoint


# ── class availability ─────────────────────────────────────────────────


def test_availability_counts_each_text_once_up_to_the_cap_per_cell(fake_sidecars):
    """The class mixture divides distinct records, never repeats."""
    cell = Cell(REWRITE, family="Draw")
    survey = a_survey(
        cells={(cell, REPRINT_A): 300, (cell, REPRINT_B): 300},
        keyless={Cell("combat", family="none"): 50},
        heap_cap=100,
    )
    decisions = _decide(survey, fake_sidecars, text_cap=100)

    assert decisions.capped_class_records == {REWRITE: 100, "combat": 50}


def test_a_class_with_no_acting_ability_is_never_capped(fake_sidecars):
    survey = a_survey(keyless={Cell("combat", family="none"): 500})
    decisions = _decide(survey, fake_sidecars, text_cap=1)

    assert decisions.capped_class_records["combat"] == 500


# ── family and signature balancing (FR-049, spec Story 4) ───────────────


def _families(per_family: dict[str, list[int]], *, klass=REWRITE, signature=""):
    """``(Cell, key) -> count`` for families whose texts hold these counts."""
    cells = {}
    labels = iter(TEXT_KEYS)
    for family, counts in per_family.items():
        for count in counts:
            cells[Cell(klass, family=family, signature=signature), TEXT_KEYS[next(labels)]] = count
    return cells


def test_a_short_family_is_written_at_capacity_and_the_rest_redistributed(
    fake_sidecars,
):
    """Spec Story 4 scenario 1, through ``decide``.

    Four families, each with one text; the class budget is 4,000. ``short``
    holds 75 distinct records, so with ``--reuse-cap 4`` its capacity is 300:
    it is written at capacity (repeated) and its unused 700 of the 1,000 fair
    share goes equally to the other three.
    """
    cells = _families({"short": [75], "a": [3000], "b": [3000], "c": [3000]})
    survey = a_survey(cells=cells, heap_cap=NO_CAP)
    decisions = _decide(
        survey, fake_sidecars, text_cap=0, reuse_cap=4,
        class_mix={REWRITE: 1.0}, training_records=4000,
    )
    shares = decisions.families[REWRITE]

    assert shares["short"].planned == 300
    rest = [shares[family].planned for family in ("a", "b", "c")]
    assert sum(rest) == 3700 and max(rest) - min(rest) <= 1
    assert {row.share for row in shares.values()} == {1000}
    short_cell = Cell(REWRITE, family="short")
    assert decisions.plans[short_cell, TEXT_KEYS["t1"]].base == 4


def test_a_family_splits_its_share_across_signatures(fake_sidecars):
    """Spec Story 4 scenario 2: outcome signatures are balanced the same way."""
    cells = {
        (Cell(EFFECT, family="Destroy", signature="died:changed"), TEXT_KEYS["t1"]): 1000,
        (Cell(EFFECT, family="Destroy", signature="stayed:changed"), TEXT_KEYS["t2"]): 1000,
    }
    survey = a_survey(cells=cells, heap_cap=NO_CAP)
    decisions = _decide(
        survey, fake_sidecars, text_cap=0, class_mix={EFFECT: 1.0}, training_records=400,
    )

    for cell, key in cells:
        assert _written(decisions, survey, cell, key) == 200


def test_real_legality_decisions_fill_half_the_budget_first(fake_sidecars):
    """Spec Story 4 scenario 3, with fewer real decisions than half the budget."""
    real, what_if = Cell(LEGALITY, half="real", family="none"), Cell(
        LEGALITY, half="what-if", family="none",
    )
    survey = a_survey(keyless={real: 100, what_if: 10_000})
    decisions = _decide(
        survey, fake_sidecars, class_mix={LEGALITY: 1.0}, training_records=1000,
        reuse_cap=1,
    )

    assert decisions.families[LEGALITY]["real:none"].planned == 100
    assert decisions.families[LEGALITY]["what-if:none"].planned == 900


def test_text_cap_and_reuse_cap_bound_a_short_familys_texts(fake_sidecars):
    """Spec Story 4 scenario 4: texts of 10, 150 and 1,000 give 40, 200, 200."""
    cell = Cell(REWRITE, family="short")
    cells = {
        (cell, TEXT_KEYS["t1"]): 10,
        (cell, TEXT_KEYS["t2"]): 150,
        (cell, TEXT_KEYS["t3"]): 1000,
        # A big family of twenty texts, each capped at 200 in its cell, so the
        # class can supply far more than the short family's share.
        **{(Cell(REWRITE, family="big"), TEXT_KEYS[f"t{i}"]): 1000 for i in range(4, 24)},
    }
    survey = a_survey(cells=cells, heap_cap=200)
    decisions = _decide(
        survey, fake_sidecars, text_cap=200, reuse_cap=4,
        class_mix={REWRITE: 1.0}, training_records=2000,
    )

    assert _written(decisions, survey, cell, TEXT_KEYS["t1"]) == 40
    assert _written(decisions, survey, cell, TEXT_KEYS["t2"]) == 200
    assert _written(decisions, survey, cell, TEXT_KEYS["t3"]) == 200
    # 10 records, each written four times; 200 over 150 records, each once or twice.
    assert decisions.plans[cell, TEXT_KEYS["t1"]].base == 4
    plan = decisions.plans[cell, TEXT_KEYS["t2"]]
    copies = [plan.copies(v) for v in survey.cell_key_heaps[cell, TEXT_KEYS["t2"]].values()]
    assert set(copies) == {1, 2}
    # The 1,000-record text is capped: 200 of its smallest hashes, once each.
    plan = decisions.plans[cell, TEXT_KEYS["t3"]]
    assert plan.base == 0


def test_a_family_with_no_records_takes_no_share(fake_sidecars):
    survey = a_survey(cells=_families({"a": [500], "b": [500]}), heap_cap=NO_CAP)
    decisions = _decide(
        survey, fake_sidecars, text_cap=0, class_mix={REWRITE: 1.0}, training_records=600,
    )

    assert set(decisions.families[REWRITE]) == {"a", "b"}
    assert decisions.families[REWRITE]["a"].share == 300


def test_two_decisions_with_one_seed_are_identical(fake_sidecars):
    """FR-052: selection is a function of the corpus, the flags and the seed."""
    survey = a_survey(cells=_families({"a": [700, 20], "b": [3]}), heap_cap=200)
    first = _decide(survey, fake_sidecars, class_mix={REWRITE: 1.0}, training_records=500)
    again = _decide(survey, fake_sidecars, class_mix={REWRITE: 1.0}, training_records=500)

    assert first.plans == again.plans
    assert first.keyless_plans == again.keyless_plans
    assert first.families == again.families
