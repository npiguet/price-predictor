"""The game-ending rules are not trained (decision D13: game-ending records are not curated).

Read against the real rules map: no heads line reaches the removed ``game_lost`` output,
and the rules whose only outcome is the game's end are not learnable."""

from __future__ import annotations

import json

from rules_catalogue_parts import locations
from rules_catalogue_parts.gen3_fields import GEN3_FIELDS_NAME, LEARNABLE
from rules_catalogue_parts.gen3_lines import learnable_today
from rules_catalogue_parts.reading import read_jsonl

#: The names a game's end was recorded under, which no output reads any more.
GAME_END_NAMES = {"new:game_result", "record:event:player_removed", "record:event:player_won"}
#: Rules that also train a real output: the game's end left their outcome, they stay trained.
STAYS_TRAINED = {"118.3b", "405.3", "603.4", "608.2a"}


def _heads():
    rows = [row for path in sorted((locations.RULES / "heads").glob("*.jsonl"))
            for row in read_jsonl(path)]
    return {row["id"]: row for row in rows}


def _registry():
    return json.loads((locations.RULES / GEN3_FIELDS_NAME).read_text(encoding="utf-8"))


def test_no_heads_line_names_the_game_lost_output():
    assert [rule for rule, row in _heads().items() if "out.game_lost" in row["gen3_outputs"]] == []


def test_the_registry_holds_no_game_lost_output():
    assert "out.game_lost" not in {entry["id"] for entry in _registry()["outputs"]}


def test_a_rule_that_trains_a_real_output_does_not_list_the_game_end():
    heads = _heads()
    assert [rule for rule in STAYS_TRAINED if GAME_END_NAMES & set(heads[rule]["outputs"])] == []


def test_the_game_end_only_rules_are_not_trained():
    ended = [rule for rule, row in _heads().items()
             if "game-ending records are not curated (D13)" in row["note"]]
    learnable = set(_registry()[LEARNABLE])
    assert len(ended) == 22
    assert [rule for rule in ended if learnable_today(_heads()[rule]) or rule in learnable] == []
    assert not any(_heads()[rule]["learnable_after_gen3"] for rule in ended)


def test_the_rules_that_train_a_real_output_stay_trained():
    heads, learnable = _heads(), set(_registry()[LEARNABLE])
    assert all(learnable_today(heads[rule]) and rule in learnable for rule in STAYS_TRAINED)
