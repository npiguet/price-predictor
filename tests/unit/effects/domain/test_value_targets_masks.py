"""What the value head's targets mask rather than state (task G13b, finding 1.10).

Lines are real gen-3 sidecar lines: read from the fixture where it holds one,
and pasted from ``output/effects/gen3/converted`` where it holds none (the coin
flips, which no fixture card has).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.domain.value_targets import VALUE_TARGET_NAMES, ValueLine

_SIDECARS = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-sidecars" / "cardsfolder"

_AMOUNTS = (
    "damage", "power_change", "toughness_change", "counters_placed", "counters_removed",
    "cards_drawn",
)
_MANA = ("mana_w", "mana_u", "mana_b", "mana_r", "mana_g", "mana_c", "mana_generic")

#: Orcish Captain: a flip whose win gives +2/+0 and whose loss gives -0/-2.
_ORCISH_CAPTAIN = (
    "AB$ FlipCoin | AILogic$ KillOrcs | Cost$ 1 | LoseSubAbility$ SV1 | SpellDescription$ "
    "Flip a coin. If you win the flip, target Orc creature gets +2/+0 until end of turn. "
    "If you lose the flip, it gets -0/-2 until end of turn. | TgtPrompt$ Select target Orc "
    "creature | ValidTgts$ Creature.Orc | WinSubAbility$ SV2 [SEG] SV1: DB$ Pump | Defined$ "
    "Targeted | NumDef$ -2 [SEG] SV2: DB$ Pump | Defined$ Targeted | NumAtt$ +2"
)
#: Goblin Bangchuckers: a win deals 2 to any target, a loss deals 2 to itself.
_BANGCHUCKERS = (
    "AB$ FlipCoin | AILogic$ Bangchuckers | Cost$ T | LoseSubAbility$ SV1 | SpellDescription$ "
    "Flip a coin. If you win the flip, CARDNAME deals 2 damage to any target. If you lose the "
    "flip, CARDNAME deals 2 damage to itself. | ValidTgts$ Any | WinSubAbility$ SV2 [SEG] SV1: "
    "DB$ DealDamage | Defined$ Self | NumDmg$ 2 [SEG] SV2: DB$ DealDamage | Defined$ Targeted "
    "| NumDmg$ 2"
)
#: Goblin Festival: 1 damage, then a flip whose loss hands the card over.
_FESTIVAL = (
    "AB$ DealDamage | Cost$ 2 | NumDmg$ 1 | SpellDescription$ CARDNAME deals 1 damage to any "
    "target. Flip a coin. If you lose the flip, choose one of your opponents. That player "
    "gains control of CARDNAME. | SubAbility$ SV1 | ValidTgts$ Any [SEG] SV1: DB$ FlipCoin | "
    "Defined$ You | LoseSubAbility$ SV2 [SEG] SV2: AILogic$ Curse | Choices$ Opponent | DB$ "
    "ChoosePlayer | Defined$ You | SubAbility$ SV3 [SEG] SV3: DB$ GainControl | Defined$ Self "
    "| NewController$ ChosenPlayer | SubAbility$ SV4 [SEG] SV4: ClearChosenPlayer$ True | DB$ "
    "Cleanup"
)


def _line(card: str, position: int) -> ValueLine:
    """The fixture card's ``position``-th sidecar line, as the value head reads it."""
    path = _SIDECARS / f"{card}.provenance.json"
    found = json.loads(path.read_text(encoding="utf-8"))["lines"][position]
    return ValueLine("", found["script_text"], found["script_api_type"] == "Keyword")


def _stated(line: ValueLine) -> dict[str, float | None]:
    targets = line.targets
    return {
        name: value if stated else None
        for name, value, stated in zip(VALUE_TARGET_NAMES, targets.values, targets.mask)
    }


def _script(script: str) -> dict[str, float | None]:
    return _stated(ValueLine("", script, False))


def test_a_line_with_no_script_masks_every_target():
    """An option line of a charm has no script of its own: it stated six zeros."""
    line = _line("a/aberrant_mind_sorcerer", 1)
    assert line.script is None
    assert set(_stated(line).values()) == {None}


@pytest.mark.parametrize("card, index", [
    ("a/abundant_maw", 0),
    ("a/alchemists_greeting", 0),
    ("a/apex_hawks", 0),
])
def test_a_keyword_line_masks_every_target(card, index):
    """``Emerge:6 B`` has no ``$`` parameter: it stated power change 0."""
    line = _line(card, index)
    assert line.keyword
    assert set(_stated(line).values()) == {None}


@pytest.mark.parametrize("card, index", [
    ("d/dreamshackle_geist", 1),
    ("t/trufflesnout", 0),
    ("s/sentinel_of_lost_lore", 0),
])
def test_a_charm_below_the_root_masks_the_amounts(card, index):
    stated = _stated(_line(card, index))
    assert [stated[name] for name in _AMOUNTS] == [None] * len(_AMOUNTS)


def test_a_chooser_below_the_root_masks_the_amounts_it_summed():
    """Zoyowa's end-step trigger: each opponent chooses a discard or a sacrifice, and
    on the other branch takes 3 damage. The choice's options were summed."""
    stated = _stated(_line("z/zoyowa_lava_tongue", 1))
    assert stated["damage"] is None


def test_a_chooser_in_the_middle_of_a_chain_masks_the_amounts():
    stated = _stated(_line("s/sweep_away", 0))
    assert stated["cards_drawn"] is None
    assert stated["damage"] is None


def test_a_coin_flips_branches_are_not_summed():
    """Orcish Captain stated +2 power and -2 toughness, both outcomes at once."""
    stated = _script(_ORCISH_CAPTAIN)
    assert stated["power_change"] is None
    assert stated["toughness_change"] is None
    assert stated["damage"] == 0


def test_a_coin_flips_two_damage_branches_are_not_summed():
    """Goblin Bangchuckers stated 4 damage, 2 to each side of the flip."""
    assert _script(_BANGCHUCKERS)["damage"] is None


def test_what_a_coin_flip_does_not_branch_on_stays_stated():
    """Goblin Festival's own 1 damage comes before the flip."""
    assert _script(_FESTIVAL)["damage"] == 1


def test_a_conditional_sub_ability_is_still_summed():
    """Aggressive Sabotage: the kicker's 3 damage is a second effect of the same
    spell, not an alternative to another branch (the plan's reading)."""
    stated = _stated(_line("a/aggressive_sabotage", 1))
    assert stated["damage"] == 3
    assert stated["cards_drawn"] == 0


def test_a_hybrid_cost_masks_its_mana_and_keeps_the_rest():
    """Cabaretti Initiate's ``2 RW`` stated 1 R and 1 W: one symbol, either colour."""
    stated = _stated(_line("c/cabaretti_initiate", 0))
    assert [stated[name] for name in _MANA] == [None] * len(_MANA)
    assert stated["cost_taps"] == 0.0
    assert stated["cost_sacrifices"] == 0.0


def test_a_hybrid_cost_masks_its_mana_and_still_states_the_amount():
    """Foxfire Oak's ``RG RG RG``: +3/+0, three hybrid symbols."""
    stated = _stated(_line("f/foxfire_oak", 0))
    assert stated["mana_r"] is None
    assert stated["mana_g"] is None
    assert stated["power_change"] == 3.0


@pytest.mark.parametrize("shard", ["2W", "WP", "WB"])
def test_each_kind_of_hybrid_shard_masks_the_mana(shard):
    stated = _script(f"AB$ Draw | Cost$ {shard} T | NumCards$ 1")
    assert stated["mana_w"] is None
    assert stated["cost_taps"] == 1.0


def test_a_line_with_a_script_states_its_cost_as_before():
    stated = _script("AB$ Draw | Cost$ 2 U T | NumCards$ 1")
    assert stated["mana_u"] == 1.0
    assert stated["cards_drawn"] == 1.0


def test_value_line_of_a_keyword_fragment_is_a_keyword_line_with_no_script():
    assert ValueLine.of("kicker 2", None) == ValueLine("kicker 2", None, True)
