"""The knowledge probes' combat and evasion grids."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from effect_knowledge_probes import (
    combat_grid,
    evasion_grid,
)
from effect_knowledge_probes.board_edits import BoardEdit

from tests.unit.scripts.knowledge_probe_support import (
    LETHAL_STEP,
    _block_pair,
    _differences,
    _keywords,
    _one_on_one,
    _outcome_by_the_rule,
)

# ── the combat grid ─────────────────────────────────────────────────────


class TestCombatGrid:
    @pytest.mark.parametrize("attacker, blocker, keywords, step, expected", [
        ((3, 3, 0), (2, 2, 0), ({}, {}), "regular", (False, True)),
        ((2, 2, 0), (2, 3, 0), ({}, {}), "regular", (True, False)),
        ((2, 2, 0), (1, 3, 1), ({}, {}), "regular", (False, True)),
        ((1, 5, 0), (1, 5, 0), ({"deathtouch"}, {}), "regular", (False, True)),
        ((5, 5, 0), (5, 5, 0), ({}, {"indestructible"}), "regular", (True, False)),
        ((3, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "first_strike", (False, True)),
        ((2, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "first_strike", (False, False)),
        ((2, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "regular", (True, False)),
        ((5, 3, 0), (3, 3, 0), ({}, {"first_strike"}), "first_strike", (True, False)),
        ((2, 2, 0), (1, 4, 0), ({"double_strike"}, {}), "first_strike", (False, False)),
        ((2, 2, 0), (1, 4, 2), ({"double_strike"}, {}), "regular", (False, True)),
        ((2, 2, 0), (0, 4, 0), ({}, {}), "first_strike", (False, False)),
    ])
    def test_the_rule(self, attacker, blocker, keywords, step, expected):
        assert combat_grid.step_deaths(
            attacker, blocker, frozenset(keywords[0]), frozenset(keywords[1]), step) == expected

    def test_power_edits_one_entity_only(self, records):
        record, attacker, _ = _one_on_one(records)
        edited = BoardEdit(record).set_power(attacker, 7).record
        assert edited.state.entity(attacker).pt.total[0] == 7
        assert (edited.state.entity(attacker).pt.total[1]
                == record.state.entity(attacker).pt.total[1])
        assert _differences(record, edited) == {"entities"}

    def test_a_granted_keyword_is_added_once(self, records):
        record, attacker, _ = _one_on_one(records)
        once = BoardEdit(record).grant_keyword(attacker, "deathtouch")
        twice = once.grant_keyword(attacker, "deathtouch").record
        assert twice.state.entity(attacker).granted_temporary.keywords.count("deathtouch") == 1

    def test_a_one_on_one_block_qualifies_when_its_outcome_obeys_the_rule(
        self, records, monkeypatch,
    ):
        from effects.domain.effect_head_layout import ZONE_OUTCOMES

        record, attacker, blocker = _one_on_one(records)
        outcome = _outcome_by_the_rule(monkeypatch, record, attacker, blocker)
        assert combat_grid.single_block(record, _keywords) == (attacker, blocker)
        dies = bool(outcome[blocker].fields)
        outcome[blocker] = SimpleNamespace(
            fields={} if dies else {"zone_outcome": ZONE_OUTCOMES.index("died")}, unknown=set())
        assert combat_grid.single_block(record, _keywords) is None

    def test_a_probe_fork_or_two_blockers_never_qualify(self, records):
        from effects.domain.records import RecordKind

        for record in records:
            if record.kind is RecordKind.COMBAT and (
                    record.fork or any(len(b) > 1 for b in record.payload.blocks.values())):
                assert combat_grid.single_block(record, _keywords) is None

    def test_grid_rules_follow_the_keyword(self, records):
        record, attacker, blocker = _one_on_one(records)
        rows, rule = combat_grid.grid_records(record, attacker, blocker, "blocker", "none")
        assert len(rows) == 8 and all(len(row) == 8 for row in rows)
        assert np.array_equal(rule, LETHAL_STEP)
        assert rows[2][5].state.entity(attacker).pt.total[0] == 3
        assert rows[2][5].state.entity(blocker).pt.total[1] == 6
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "attacker-deathtouch")
        assert rule.min() == 1.0
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "blocker-indestructible")
        assert rule.max() == 0.0

    def test_a_first_striker_is_read_in_the_first_strike_step(self, records):
        record, attacker, blocker = _one_on_one(records)
        rows, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "attacker-first-strike")
        assert rows[0][0].state.global_.combat_substep == "first_strike"
        assert np.array_equal(rule, LETHAL_STEP)
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "attacker", "attacker-first-strike")
        assert rule.max() == 0.0
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "blocker-first-strike")
        assert rule.max() == 0.0
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "attacker", "blocker-first-strike")
        assert np.array_equal(rule, LETHAL_STEP)

    def test_a_double_strikers_second_hit_lands_on_the_first(self, records):
        record, attacker, blocker = _one_on_one(records)
        rows, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "attacker-double-strike-second-hit")
        assert rows[2][5].state.global_.combat_substep == "regular"
        assert rows[2][5].state.entity(blocker).damage == 3
        assert np.array_equal(rule, np.array([[float(2 * p >= t) for t in range(1, 9)]
                                              for p in range(1, 9)]))

    def test_the_assignment_follows_the_edited_board(self, records):
        record, attacker, blocker = _one_on_one(records)
        rows, _ = combat_grid.grid_records(record, attacker, blocker, "blocker", "none")
        choices = rows[4][1].payload.assignment_choices
        assert choices[attacker] == {blocker: 5}
        assert choices[blocker] == {attacker: rows[4][1].state.entity(blocker).pt.total[0]}
        rows, _ = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "attacker-first-strike")
        choices = rows[2][0].payload.assignment_choices
        assert choices[attacker] == {blocker: 3}
        assert blocker not in choices, "the blocker deals nothing in the first-strike step"

    def test_a_first_strike_step_record_never_qualifies(self, records, monkeypatch):
        record, attacker, blocker = _one_on_one(records)
        _outcome_by_the_rule(monkeypatch, record, attacker, blocker)
        assert combat_grid.single_block(record, _keywords) == (attacker, blocker)
        first = BoardEdit(record).set_step("first_strike").record
        assert combat_grid.single_block(first, _keywords) is None

    def test_damage_the_step_already_marked_is_cleared_and_other_damage_disqualifies(
        self, records, monkeypatch,
    ):
        record, attacker, blocker = _one_on_one(records)
        power = record.state.entity(attacker).pt.total[0]
        marked = BoardEdit(record).set_marked_damage(blocker, power).record
        _outcome_by_the_rule(monkeypatch, record, attacker, blocker)
        assert combat_grid.single_block(marked, _keywords) == (attacker, blocker)
        earlier = BoardEdit(record).set_marked_damage(blocker, power + 1).record
        assert combat_grid.single_block(earlier, _keywords) is None
        rows, rule = combat_grid.grid_records(marked, attacker, blocker, "attacker", "none")
        assert all(cell.state.entity(blocker).damage == 0 for row in rows for cell in row)
        assert np.array_equal(rule, LETHAL_STEP)


# ── the evasion grid ────────────────────────────────────────────────────

class TestEvasionGrid:
    @pytest.mark.parametrize("attacker, blocker, colours, types, expected", [
        ((), (), ("G",), ("creature",), True),
        (("flying",), (), ("G",), ("creature",), False),
        (("flying",), ("reach",), ("G",), ("creature",), True),
        (("horsemanship",), ("flying",), ("G",), ("creature",), False),
        (("fear",), (), ("B",), ("creature",), True),
        (("fear",), (), ("G",), ("artifact", "creature"), True),
        (("fear",), (), ("G",), ("creature",), False),
        (("intimidate",), (), ("R",), ("creature",), True),
        (("intimidate",), (), ("G",), ("creature",), False),
        (("shadow",), (), ("G",), ("creature",), False),
        ((), ("shadow",), ("G",), ("creature",), False),
        (("unblockable",), ("reach",), ("G",), ("creature",), False),
        (("protection_from_green",), (), ("G",), ("creature",), False),
        (("protection_from_blue",), (), ("G",), ("creature",), True),
    ])
    def test_the_rule(self, attacker, blocker, colours, types, expected):
        assert evasion_grid.may_block(attacker, ("R",), blocker, colours, types) is expected

    def test_a_cell_grants_the_keyword_and_the_trait_and_scores_them(self, records):
        record, attacker, blocker = _block_pair(records)
        cell, rule = evasion_grid.edited(record, attacker, blocker, "flying", None)
        assert "flying" in cell.state.entity(attacker).granted_temporary.keywords
        assert rule is False
        cell, rule = evasion_grid.edited(record, attacker, blocker, "flying", "reach")
        assert "reach" in cell.state.entity(blocker).granted_temporary.keywords
        assert rule is True
        _, rule = evasion_grid.edited(record, attacker, blocker, "intimidate", "shares-colour")
        assert rule is True
        _, rule = evasion_grid.edited(record, attacker, blocker, "fear", "artifact")
        assert rule is True

    def test_protection_resolves_against_the_blockers_colour(self, records):
        record, attacker, blocker = _block_pair(records)
        cell, rule = evasion_grid.edited(record, attacker, blocker, "protection-blocker", None)
        assert "protection_from_green" in cell.state.entity(attacker).granted_temporary.keywords
        assert rule is False
        cell, rule = evasion_grid.edited(record, attacker, blocker, "protection-other", None)
        granted = cell.state.entity(attacker).granted_temporary.keywords
        assert any(k.startswith("protection_from_") and k != "protection_from_green"
                   and k != "protection_from_red" for k in granted)
        assert rule is True

    def test_qualification(self, records):
        record, attacker, blocker = _block_pair(records)
        assert evasion_grid.qualifying_pair(record, _keywords) == (attacker, blocker)
        assert evasion_grid.qualifying_pair(replace(record, fork=True), _keywords) is None
        flying = BoardEdit(record).grant_keyword(attacker, "flying").record
        assert evasion_grid.qualifying_pair(flying, _keywords) is None
        black = BoardEdit(record).set_colours(blocker, ("B",)).record
        assert evasion_grid.qualifying_pair(black, _keywords) is None
