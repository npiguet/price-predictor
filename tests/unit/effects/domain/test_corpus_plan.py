"""The corpus plan's pure rules: the allocation tree, legality cells, admission, strata."""

from __future__ import annotations

from collections import Counter

from effects.domain.corpus_curation import CapHeap
from effects.domain.corpus_plan import (
    AllocationNode,
    Cell,
    GameStrata,
    TrainingSupply,
    admit_card_disjoint,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.sampling import SamplingClass
from effects.domain.strata import Stratum

LEGALITY = SamplingClass.PLAYABILITY_LEGALITY


def _leaf(supply: int, once: int | None = None) -> AllocationNode:
    return AllocationNode(supply=supply, once=supply if once is None else once)


class TestAllocationNode:
    def test_a_short_child_is_filled_and_its_remainder_passed_on(self):
        root = AllocationNode()
        families = root.child("")
        for name, capacity in (("short", 300), ("a", 9000), ("b", 9000), ("c", 9000)):
            families.children[name] = AllocationNode(children={"x": _leaf(capacity)})

        assert root.allocate(4000) == 4000
        rows = families.children
        assert rows["short"].planned == 300
        assert sorted(rows[name].planned for name in "abc") == [1233, 1233, 1234]
        assert {row.share for row in rows.values()} == {1000}

    def test_shares_halve_and_split_down_the_levels(self):
        root = AllocationNode()
        for half in ("real", "what-if"):
            for family in ("f1", "f2"):
                root.child(half).child(family).children["x"] = _leaf(10_000)

        root.allocate(1000)
        assert root.share == 1000
        assert root.children["real"].share == 500
        assert root.children["real"].children["f1"].share == 250

    def test_members_are_used_once_before_any_is_repeated(self):
        """FR-049 step 4: no record repeats while another of the cell goes unused."""
        cell = AllocationNode(children={
            "rare": _leaf(supply=40, once=10),       # 10 records, reuse cap 4
            "common": _leaf(supply=200, once=200),
        })
        cell.allocate(150)
        assert cell.children["rare"].planned == 10
        assert cell.children["common"].planned == 140

    def test_a_budget_past_every_capacity_plans_only_the_capacity(self):
        root = AllocationNode(children={"x": _leaf(30), "y": _leaf(20)})
        assert root.allocate(1000) == 50


class TestLegalityCells:
    """A legality record is placed in its rarest mode's family (FR-047a)."""

    def test_a_cell_folds_into_its_rarest_mode(self):
        counts = {"CantAttack": 35, "CantBlock": 7, "Defender": 42}
        cell = Cell(LEGALITY, "real", modes=frozenset({"CantAttack", "CantBlock"}))
        assert cell.resolved(counts) == Cell(LEGALITY, "real", "CantBlock")

    def test_a_cell_with_no_modes_is_already_resolved(self):
        cell = Cell(LEGALITY, "real", "none")
        assert cell.resolved({"CantAttack": 1}) is cell

    def test_cells_folding_to_one_family_merge_their_supply(self):
        key = (ProvenanceKey("cardsfolder/p/pacifism.txt", 0, "static", 0),)
        supply = TrainingSupply(heap_cap=10)
        both = Cell(LEGALITY, "real", modes=frozenset({"CantAttack", "CantBlock"}))
        alone = Cell(LEGALITY, "real", modes=frozenset({"CantBlock"}))
        supply.add(both, None, 0)
        supply.add(alone, None, 0)
        supply.add(both, key, 5)
        supply.add(alone, key, 6)
        supply.add(Cell("combat", "", "none"), None, 0)

        folded = supply.resolved({"CantAttack": 35, "CantBlock": 7})
        block = Cell(LEGALITY, "real", "CantBlock")
        assert folded.keyless == Counter({block: 2, Cell("combat", "", "none"): 1})
        assert folded.keyed == Counter({(block, key): 2})
        assert sorted(folded.heaps[block, key].values()) == [5, 6]


class TestCardDisjointAdmission:
    """FR-142: at most the cap of games per held-out text, every text under it."""

    def test_the_cap_admits_exactly_its_cap_of_games(self):
        games = {f"g{i}" for i in range(20)}
        admitted = admit_card_disjoint(games, {"text": games}, cap=2, seed=1)
        assert len(admitted) == 2

    def test_a_game_is_declined_once_any_of_its_texts_is_full(self):
        games = {f"g{i}" for i in range(10)}
        texts = {"busy": games, **{f"own{i}": {f"g{i}"} for i in range(10)}}
        admitted = admit_card_disjoint(games, texts, cap=3, seed=1)
        assert len(admitted) == 3

    def test_a_held_out_game_resolving_no_held_out_text_is_not_admitted(self):
        admitted = admit_card_disjoint({"g1", "g2"}, {"text": {"g1"}}, cap=0, seed=1)
        assert admitted == frozenset({"g1"})

    def test_only_held_out_games_are_admitted(self):
        """A variant printing a held-out template can resolve in a training game."""
        admitted = admit_card_disjoint({"g1"}, {"text": {"g1", "g9"}}, cap=0, seed=1)
        assert admitted == frozenset({"g1"})

    def test_the_draw_is_seeded(self):
        games = {f"g{i}" for i in range(20)}
        first = admit_card_disjoint(games, {"text": games}, cap=2, seed=7)
        again = admit_card_disjoint(games, {"text": games}, cap=2, seed=7)
        assert first == again


def test_a_held_out_game_the_cap_declined_is_dropped_never_validated():
    strata = GameStrata(
        card_disjoint=frozenset({"g1"}), game_disjoint=frozenset({"g2", "g3"}),
        held_out=frozenset({"g1", "g3"}),
    )
    assert strata.stratum_of("g1") is Stratum.CARD_DISJOINT
    assert strata.stratum_of("g2") is Stratum.GAME_DISJOINT
    assert strata.stratum_of("g3") is Stratum.DROPPED_HELD_OUT
    assert strata.stratum_of("g4") is Stratum.TRAINING


def test_the_supply_keeps_each_slots_smallest_hashes():
    key = (ProvenanceKey("cardsfolder/b/bolt.txt", 0, "spell", 0),)
    supply = TrainingSupply(heap_cap=2)
    for value in (9, 1, 5):
        supply.add(Cell("rewrite"), key, value)
    assert supply.keyed[Cell("rewrite"), key] == 3
    assert sorted(supply.heaps[Cell("rewrite"), key].values()) == [1, 5]
    assert isinstance(supply.heaps[Cell("rewrite"), key], CapHeap)
