"""The value head's targets, read from a line's script (T091, FR-058, FR-058a).

Texts are real script lines from the converted tree, chained by hand in the
gen-2 form where a test needs more than one segment.
"""

from __future__ import annotations

import json
from pathlib import Path

from effects.domain.value_targets import (
    VALUE_TARGET_NAMES,
    params,
    segments,
    value_targets,
)

#: Lightning Bolt's script line, as gen-1's sidecar carries it.
_BOLT = (
    "NumDmg$ 3 | SP$ DealDamage | SpellDescription$ CARDNAME deals 3 damage to "
    "any target. | ValidTgts$ Any"
)
#: Bone Splinters: an additional cost of a sacrifice, with mana in Cost$.
_BONE_SPLINTERS = (
    "Cost$ B Sac<1/Creature> | SP$ Destroy | SpellDescription$ Destroy target creature. "
    "| TgtPrompt$ Select target creature | ValidTgts$ Creature"
)
#: Prodigal Sorcerer's activated ability.
_TIM = (
    "AB$ DealDamage | Cost$ T | NumDmg$ 1 | SpellDescription$ CARDNAME deals 1 "
    "damage to any target. | ValidTgts$ Any"
)
#: Cryptic Command's charm root under gen-2's chain rendering.
_CHARM = "CharmNum$ 2 | Choices$ SV1,SV2,SV3,SV4 | SP$ Charm"


def test_a_one_segment_gen1_text_reads_its_amount():
    targets = value_targets(_BOLT)
    assert targets.get("damage") == 3
    assert targets.get("cards_drawn") == 0


def test_amounts_sum_over_segments():
    """Spec Story 5 scenario 5: NumDmg$ 3 in one segment and 2 in another."""
    text = (
        "AB$ DealDamage | Cost$ 2 R | NumDmg$ 3 | SubAbility$ SV1 | ValidTgts$ Creature"
        " [SEG] SV1: DB$ DealDamage | Defined$ You | NumDmg$ 2"
    )
    assert value_targets(text).get("damage") == 5


def test_a_variable_amount_in_any_segment_masks_the_target():
    text = (
        "AB$ DealDamage | Cost$ X R | NumDmg$ X | SubAbility$ SV1"
        " [SEG] SV1: DB$ DealDamage | NumDmg$ 2"
    )
    assert value_targets(text).get("damage") is None


def test_a_count_svar_amount_is_masked():
    text = "NumDmg$ Count$Valid Creature.YouCtrl | SP$ DealDamage | ValidTgts$ Any"
    assert value_targets(text).get("damage") is None


def test_a_signed_pump_reads_plus_and_minus_literals():
    text = "AB$ Pump | Cost$ G | NumAtt$ +2 | NumDef$ -1 | ValidTgts$ Creature"
    targets = value_targets(text)
    assert targets.get("power_change") == 2
    assert targets.get("toughness_change") == -1


def test_cards_drawn_counts_draw_segments_only():
    text = (
        "SP$ Draw | NumCards$ 2 | SubAbility$ SV1"
        " [SEG] SV1: DB$ Discard | Mode$ TgtChoose | NumCards$ 1"
    )
    assert value_targets(text).get("cards_drawn") == 2


def test_bone_splinters_masks_mana_and_sets_the_sacrifice():
    """A spell line's mana is masked even where Cost$ carries mana (FR-058a)."""
    targets = value_targets(_BONE_SPLINTERS)
    for name in ("mana_w", "mana_u", "mana_b", "mana_r", "mana_g", "mana_c",
                 "mana_generic"):
        assert targets.get(name) is None, name
    assert targets.get("cost_sacrifices") == 1.0
    assert targets.get("cost_taps") == 0.0


def test_an_activated_cost_reads_its_mana_and_tap():
    text = "AB$ Draw | Cost$ 2 U T | NumCards$ 1"
    targets = value_targets(text)
    assert targets.get("mana_u") == 1
    assert targets.get("mana_generic") == 2
    assert targets.get("cost_taps") == 1.0
    assert targets.get("cost_sacrifices") == 0.0


def test_a_tap_only_cost_states_zero_mana():
    targets = value_targets(_TIM)
    assert targets.get("mana_generic") == 0
    assert targets.get("cost_taps") == 1.0
    assert targets.get("damage") == 1


def test_an_x_cost_masks_the_mana_targets():
    targets = value_targets("AB$ DealDamage | Cost$ X R | NumDmg$ 1")
    assert targets.get("mana_r") is None


def test_no_cost_masks_every_cost_target():
    targets = value_targets(_BOLT)
    for name in ("mana_r", "cost_taps", "cost_sacrifices"):
        assert targets.get(name) is None


def test_a_charm_root_masks_every_amount():
    targets = value_targets(_CHARM)
    for name in ("damage", "power_change", "toughness_change",
                 "counters_placed", "cards_drawn"):
        assert targets.get(name) is None, name


def test_the_mask_and_values_align_with_the_target_list():
    targets = value_targets(_TIM)
    assert len(targets.values) == len(targets.mask) == len(VALUE_TARGET_NAMES)


def test_segments_drop_their_labels_and_params_split_at_the_first_dollar():
    parts = segments("SP$ Draw | SubAbility$ SV1 [SEG] SV1: DB$ Pump | NumAtt$ +1")
    assert parts == ["SP$ Draw | SubAbility$ SV1", "DB$ Pump | NumAtt$ +1"]
    assert params("NumDmg$ Count$Valid Creature | SP$ DealDamage") == {
        "NumDmg": "Count$Valid Creature", "SP": "DealDamage",
    }


# ── real gen-2 sidecars ─────────────────────────────────────────────────

_GEN2 = Path(__file__).parents[3] / "fixtures" / "effects" / "gen2-sidecars" / "cardsfolder"


def _texts(card: str) -> list[str]:
    data = json.loads((_GEN2 / f"{card}.provenance.json").read_text(encoding="utf-8"))
    return [line["script_text"] for line in data["lines"] if line["script_text"]]


def test_arc_trail_sums_its_two_segments():
    assert value_targets(_texts("a/arc_trail")[0]).get("damage") == 3


def test_a_trigger_reads_the_damage_of_its_executed_segment():
    (text,) = [t for t in _texts("a/abraded_bluffs") if "NumDmg" in t]
    assert value_targets(text).get("damage") == 1


def test_cryptic_commands_root_masks_amounts_and_its_draw_mode_reads_one():
    root, *modes = _texts("c/cryptic_command")
    assert value_targets(root).get("damage") is None
    assert value_targets(modes[3]).get("cards_drawn") == 1
    assert value_targets(modes[0]).get("cards_drawn") == 0
