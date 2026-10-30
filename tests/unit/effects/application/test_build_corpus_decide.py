"""``decide``: fold the survey into copy plans, class budgets and the strata.

The survey (``run_survey``) keys everything by acting-key tuple and never
resolves a key to ability text (see ``test_build_corpus_survey.py``); these
tests build ``Survey`` objects directly and exercise the fold that happens
once, here, against a fake ``SidecarCache``.

Gen-2 selection (FR-049) is a cell-by-cell plan: a class budget split equally
across rule families, a family's across outcome signatures, the legality class
across its real and what-if halves, and a cell's across its ability texts, each
level capped by what its members can supply with ``--reuse-cap`` repeats and
``--text-cap`` per text per cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from effects.application.build_corpus import BuildCorpusConfig, decide
from effects.application.corpus_survey import GameTally, RarityTally, ShardSurvey, Survey
from effects.domain.corpus_plan import Cell, TrainingSupply
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


def _keys(label: str) -> tuple[ProvenanceKey, ...]:
    """``label`` as the acting-key tuple a survey keys on."""
    return (_provenance_key(label),)


REPRINT_A = _keys("reprint-a")
REPRINT_B = _keys("reprint-b")
UNKNOWN_KEY = _keys("unknown-key")
MISMATCH_KEY = (ProvenanceKey("cardsfolder/x/reprint_a.txt", 0, "trigger", 0),)
#: Texts for the family tests, one key each.
TEXT_KEYS = {f"t{i}": _keys(f"t{i}") for i in range(1, 31)}


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
        *((label, f"text {label}") for label in TEXT_KEYS),
    ):
        key = _provenance_key(label)
        lines[key] = _line(text, key)
    return FakeSidecarCache(lines=lines)


def a_survey(
    *, heap_cap: int = 200, cells: dict | None = None,
    keyless: dict | None = None, key_games: dict | None = None,
    held_out: frozenset[str] = frozenset(), game_disjoint: frozenset[str] = frozenset(),
    held_out_text_games: dict | None = None,
) -> Survey:
    """A real ``Survey`` whose supply holds ``cells`` and ``keyless``.

    ``cells`` maps ``(Cell, keys)`` to a record count, or to ``(count,
    hashes)``; without explicit hashes the slot's records get the hashes
    ``0 .. count-1`` offset by the slot's position, so two slots never
    collide. Absent both, the supply holds one keyless rewrite record, so the
    class mixture has a class to size against.
    """
    cells = dict(cells or {})
    supply = TrainingSupply(heap_cap)
    for cell, count in (keyless or ({} if cells else {Cell(REWRITE): 1})).items():
        supply.keyless[cell] += count
    rarity = RarityTally()
    for position, ((cell, keys), value) in enumerate(cells.items()):
        count, hashes = value if isinstance(value, tuple) else (
            value, range(position * 10_000, position * 10_000 + value),
        )
        for item in hashes:
            supply.add(cell, keys, item)
        supply.keyed[cell, keys] = count
        rarity.key_records[keys] += count
    for keys, games in (key_games or {}).items():
        rarity.key_records.setdefault(keys, 0)
        rarity.key_games[keys] = set(games)
    return Survey(
        totals=ShardSurvey(
            name="", size=0, supply=supply, rarity=rarity,
            games=GameTally(
                held_out=set(held_out), game_disjoint=set(game_disjoint),
                held_out_text_games={
                    text: set(games) for text, games in (held_out_text_games or {}).items()
                },
            ),
        ),
        shards=(),
    )


def a_config(**overrides) -> BuildCorpusConfig:
    overrides.setdefault("keyword_definitions", None)
    return BuildCorpusConfig(records_dir=Path("."), **overrides)


def _decide(survey, fake_sidecars, **config):
    return decide(survey, sidecars=fake_sidecars, surface="script", config=a_config(**config))


def _written(decisions, survey, cell: Cell, keys) -> int:
    """Copies a plan writes for every hash the survey kept for ``(cell, keys)``."""
    plan = decisions.selection.plans[cell, keys]
    return sum(plan.copies(value) for value in survey.totals.supply.heaps[cell, keys].values())


# ── the fold ────────────────────────────────────────────────────────────


def test_two_keys_folding_to_one_text_share_one_cap(fake_sidecars):
    """The cap is per text: two keys of one text share its ``--text-cap``."""
    cell = Cell(REWRITE, family="Draw")
    survey = a_survey(
        cells={(cell, REPRINT_A): 300, (cell, REPRINT_B): 300}, heap_cap=100,
    )
    decisions = _decide(survey, fake_sidecars, text_cap=100, class_mix={REWRITE: 1.0})
    plans = decisions.selection.plans

    assert plans[cell, REPRINT_A] == plans[cell, REPRINT_B]
    total = _written(decisions, survey, cell, REPRINT_A) + _written(
        decisions, survey, cell, REPRINT_B,
    )
    assert total == 100


def test_rarity_counts_games_over_the_whole_corpus(fake_sidecars):
    survey = a_survey(key_games={REPRINT_A: {1, 2}, REPRINT_B: {2, 3}})
    decisions = _decide(survey, fake_sidecars)

    assert decisions.rarity["deals 3 damage"] == 3


def test_keys_no_sidecar_line_carries_are_sampled_with_the_cells_keyless_records(
    fake_sidecars,
):
    cell = Cell(REWRITE, family="Draw")
    survey = a_survey(cells={(cell, UNKNOWN_KEY): 500}, heap_cap=10)
    decisions = _decide(survey, fake_sidecars, text_cap=10, class_mix={REWRITE: 1.0})

    assert (cell, UNKNOWN_KEY) not in decisions.selection.plans
    assert decisions.selection.keyless[cell].base == 1
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


# ── the strata ─────────────────────────────────────────────────────────


def test_the_game_disjoint_stratum_is_the_surveys_never_a_held_out_game(fake_sidecars):
    """The survey routes each game once; decide applies it rather than redrawing."""
    survey = a_survey(
        game_disjoint=frozenset({"g1", "g2", "g3"}), held_out=frozenset({"g3"}),
    )
    decisions = _decide(survey, fake_sidecars)

    assert decisions.strata.game_disjoint == frozenset({"g1", "g2"})


def test_keyword_threshold_games_are_the_ones_over_the_plain_share(fake_sidecars):
    from effects.domain.corpus_curation import game_disjoint_draw

    games = {f"g{i}" for i in range(400)}
    between = {g for g in games if 0.01 <= game_disjoint_draw(g) < 0.15}
    below = {g for g in games if game_disjoint_draw(g) < 0.01}
    assert between and below, "the fixture needs games on both sides of 0.01"
    survey = a_survey(game_disjoint=frozenset(between | below))
    decisions = _decide(survey, fake_sidecars)

    assert decisions.keyword_threshold_games == frozenset(between)


def test_the_card_disjoint_stratum_admits_by_held_out_text(fake_sidecars):
    games = {f"g{i}" for i in range(20)}
    survey = a_survey(
        held_out=frozenset(games), held_out_text_games={"held out text": games},
    )
    decisions = _decide(survey, fake_sidecars, card_disjoint_text_cap=2)

    assert len(decisions.strata.card_disjoint) == 2
    assert decisions.strata.held_out == frozenset(games)


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

    assert decisions.selection.available == {REWRITE: 100, "combat": 50}


def test_a_class_with_no_acting_ability_is_never_capped(fake_sidecars):
    survey = a_survey(keyless={Cell("combat", family="none"): 500}, heap_cap=1)
    decisions = _decide(survey, fake_sidecars, text_cap=1)

    assert decisions.selection.available["combat"] == 500


# ── family and signature balancing (FR-049, spec Story 4) ───────────────


def _families(per_family: dict[str, list[int]], *, klass=REWRITE, signature=""):
    """``(Cell, keys) -> count`` for families whose texts hold these counts."""
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
    shares = decisions.selection.families[REWRITE]

    assert shares["short"].planned == 300
    rest = [shares[family].planned for family in ("a", "b", "c")]
    assert sum(rest) == 3700 and max(rest) - min(rest) <= 1
    assert {row.share for row in shares.values()} == {1000}
    short_cell = Cell(REWRITE, family="short")
    assert decisions.selection.plans[short_cell, TEXT_KEYS["t1"]].base == 4


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
    families = decisions.selection.families[LEGALITY]

    assert families["real:none"].planned == 100
    assert families["what-if:none"].planned == 900


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
    plans = decisions.selection.plans

    assert _written(decisions, survey, cell, TEXT_KEYS["t1"]) == 40
    assert _written(decisions, survey, cell, TEXT_KEYS["t2"]) == 200
    assert _written(decisions, survey, cell, TEXT_KEYS["t3"]) == 200
    # 10 records, each written four times; 200 over 150 records, each once or twice.
    assert plans[cell, TEXT_KEYS["t1"]].base == 4
    plan = plans[cell, TEXT_KEYS["t2"]]
    copies = [plan.copies(v) for v in survey.totals.supply.heaps[cell, TEXT_KEYS["t2"]].values()]
    assert set(copies) == {1, 2}
    # The 1,000-record text is capped: 200 of its smallest hashes, once each.
    assert plans[cell, TEXT_KEYS["t3"]].base == 0


def test_a_family_with_no_records_takes_no_share(fake_sidecars):
    survey = a_survey(cells=_families({"a": [500], "b": [500]}), heap_cap=NO_CAP)
    decisions = _decide(
        survey, fake_sidecars, text_cap=0, class_mix={REWRITE: 1.0}, training_records=600,
    )

    assert set(decisions.selection.families[REWRITE]) == {"a", "b"}
    assert decisions.selection.families[REWRITE]["a"].share == 300


def test_two_decisions_with_one_seed_are_identical(fake_sidecars):
    """FR-052: selection is a function of the corpus, the flags and the seed."""
    survey = a_survey(cells=_families({"a": [700, 20], "b": [3]}), heap_cap=200)
    first = _decide(survey, fake_sidecars, class_mix={REWRITE: 1.0}, training_records=500)
    again = _decide(survey, fake_sidecars, class_mix={REWRITE: 1.0}, training_records=500)

    assert first.selection == again.selection


def test_the_class_targets_are_planned_in_full(fake_sidecars):
    """Every target is placed somewhere in the class's tree, nothing past it."""
    cells = _families({"a": [700, 20], "b": [3]})
    survey = a_survey(
        cells=cells, keyless={Cell("combat", family="none"): 90}, heap_cap=200,
    )
    decisions = _decide(survey, fake_sidecars, class_mix={REWRITE: 0.5, "combat": 0.5})
    selection = decisions.selection

    for klass, target in selection.targets.items():
        planned = sum(row.planned for row in selection.families[klass].values())
        assert planned == target


def test_a_mixture_naming_no_class_the_corpus_holds_is_an_operator_error(fake_sidecars):
    from effects.application.build_corpus import BuildCorpusError

    with pytest.raises(BuildCorpusError):
        _decide(a_survey(), fake_sidecars, class_mix={"trigger": 1.0})
