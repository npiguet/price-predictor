"""``acting_lines`` finds the source lines of a chain whose segments carry a
label the converter renamed (``ResultSubAbilities$``, ``Trigger$``...), not
only the ones after ``Execute$``/``SubAbility$``.

The scripts are Forge's own card scripts, and the ``script_text`` strings are
the provenance sidecar lines the converter wrote for them at commit 3a473430
(task AR6); each is copied from the cards named in the comments.
"""

from __future__ import annotations

from effects.domain.variant_rounds import acting_lines

# forge-gui/res/cardsfolder/c/contact_other_plane.txt
_CONTACT_OTHER_PLANE = [
    "Name:Contact Other Plane",
    "A:SP$ RollDice | Sides$ 20 | ResultSubAbilities$ 1-9:DBDraw2,10-19:DBScry2,20:DBScry3 "
    "| SpellDescription$ Roll a d20.",
    "SVar:DBDraw2:DB$ Draw | NumCards$ 2 | SpellDescription$ 1—9 VERT Draw two cards.",
    "SVar:DBScry2:DB$ Scry | ScryNum$ 2 | SubAbility$ DBDraw2 "
    "| SpellDescription$ 10—19 VERT Scry 2, then draw two cards.",
    "SVar:DBScry3:DB$ Scry | ScryNum$ 3 | SubAbility$ DBDraw3 "
    "| SpellDescription$ 20 VERT Scry 3, then draw three cards.",
    "SVar:DBDraw3:DB$ Draw | NumCards$ 3",
]
_CONTACT_OTHER_PLANE_TEXT = (
    "ResultSubAbilities$ 1-9:SV1,10-19:SV2,20:SV3 | SP$ RollDice | Sides$ 20 "
    "| SpellDescription$ Roll a d20. "
    "[SEG] SV1: DB$ Draw | NumCards$ 2 | SpellDescription$ 1—9 VERT Draw two cards. "
    "[SEG] SV2: DB$ Scry | ScryNum$ 2 | SpellDescription$ 10—19 VERT Scry 2, then draw two "
    "cards. | SubAbility$ SV1 "
    "[SEG] SV3: DB$ Scry | ScryNum$ 3 | SpellDescription$ 20 VERT Scry 3, then draw three "
    "cards. | SubAbility$ SV4 "
    "[SEG] SV4: DB$ Draw | NumCards$ 3"
)

# forge-gui/res/cardsfolder/a/ahn_crop_champion.txt
_AHN_CROP_CHAMPION = [
    "Name:Ahn-Crop Champion",
    "S:Mode$ OptionalAttackCost | ValidCard$ Card.Self | Trigger$ TrigUntapAll "
    "| Cost$ Exert<1/CARDNAME> | Description$ You may exert CARDNAME as it attacks. "
    "When you do, untap all other creatures you control.",
    "SVar:TrigUntapAll:DB$ UntapAll | ValidCards$ Creature.YouCtrl+StrictlyOther "
    "| SpellDescription$ When you do, untap all other creatures you control.",
    "SVar:AIExertCondition:NumCreats GE3",
]
_AHN_CROP_CHAMPION_TEXT = (
    "Cost$ Exert<1/CARDNAME> | Description$ You may exert CARDNAME as it attacks. When you "
    "do, untap all other creatures you control. | Mode$ OptionalAttackCost | Trigger$ SV1 "
    "| ValidCard$ Card.Self "
    "[SEG] SV1: DB$ UntapAll | SpellDescription$ When you do, untap all other creatures you "
    "control. | ValidCards$ Creature.YouCtrl+StrictlyOther"
)

# forge-gui/res/cardsfolder/r/ranger_class.txt, the level-2 line's sidecar text.
_RANGER_CLASS = [
    "Name:Ranger Class",
    "K:Class:2:1 G:AddTrigger$ TriggerAttackersDeclared",
    "SVar:TriggerAttackersDeclared:Mode$ AttackersDeclared | AttackingPlayer$ You "
    "| TriggerZones$ Battlefield | Execute$ TrigPutCounter | Secondary$ True "
    "| TriggerDescription$ Whenever you attack, put a +1/+1 counter on target attacking "
    "creature.",
    "SVar:TrigPutCounter:DB$ PutCounter | ValidTgts$ Creature.attacking "
    "| TgtPrompt$ Select target attacking creature | CounterType$ P1P1 | CounterNum$ 1",
]
_RANGER_CLASS_LEVEL_2_TEXT = (
    "AB$ ClassLevelUp | ClassLevel$ EQ1 | Cost$ 1 G | SorcerySpeed$ True "
    "| SpellDescription$ Level 2 | StackDescription$ SpellDescription "
    "[SEG] SV1: AddTrigger$ SV2 | Affected$ Card.Self | ClassLevel$ 2 | Description$ "
    "Whenever you attack, put a +1/+1 counter on target attacking creature. "
    "| Mode$ Continuous "
    "[SEG] SV2: AttackingPlayer$ You | Execute$ SV3 | Mode$ AttackersDeclared "
    "| Secondary$ True | TriggerDescription$ Whenever you attack, put a +1/+1 counter on "
    "target attacking creature. | TriggerZones$ Battlefield "
    "[SEG] SV3: CounterNum$ 1 | CounterType$ P1P1 | DB$ PutCounter "
    "| TgtPrompt$ Select target attacking creature | ValidTgts$ Creature.attacking"
)


def test_a_die_roll_s_root_line_is_found_beside_its_results():
    assert acting_lines(_CONTACT_OTHER_PLANE, _CONTACT_OTHER_PLANE_TEXT) == [1, 2, 3, 4, 5]


def test_a_static_s_root_line_is_found_beside_its_trigger_sub_ability():
    assert acting_lines(_AHN_CROP_CHAMPION, _AHN_CROP_CHAMPION_TEXT) == [1, 2]


def test_a_class_level_finds_the_granted_trigger_and_its_effect():
    """The level's own ``AB$ ClassLevelUp`` and its ``AddTrigger$`` carrier are
    built by the converter, not written in the script: only the SVars match."""
    assert acting_lines(_RANGER_CLASS, _RANGER_CLASS_LEVEL_2_TEXT) == [2, 3]
