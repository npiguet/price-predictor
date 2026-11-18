"""Every renamed chain label (``SV1``, ``SV2``...) is one token on the script
surface, whichever parameter names it (AR13).

The script texts are lines of the converter's output at commit 3a473430, descriptions abridged
(Contact Other Plane, Ahn-Crop Champion, Ranger Class levels 2 and 3).
"""

from __future__ import annotations

import re

import pytest

from effects.domain.ability_tokenizer import AbilityTokenizer

_WORDS = (
    "draw a card scry untap all other creatures you control wolf token "
    "whenever attack put counter on target attacking creature look at top "
    "of your library any time may cast spells from roll"
).split()

# contact_other_plane.provenance.json, line 3 (ResultSubAbilities$).
_CONTACT_OTHER_PLANE = (
    "ResultSubAbilities$ 1-9:SV1,10-19:SV2,20:SV3 | SP$ RollDice | Sides$ 20 | "
    "SpellDescription$ Roll a d20. [SEG] SV1: DB$ Draw | NumCards$ 2 | "
    "SpellDescription$ 1-9 Draw two cards. [SEG] SV2: DB$ Scry | ScryNum$ 2 | "
    "SpellDescription$ 10-19 Scry 2, then draw two cards. | SubAbility$ SV1 "
    "[SEG] SV3: DB$ Scry | ScryNum$ 3 | SpellDescription$ 20 Scry 3. | "
    "SubAbility$ SV4 [SEG] SV4: DB$ Draw | NumCards$ 3"
)

# ahn_crop_champion.provenance.json, line 4 (Trigger$).
_AHN_CROP_CHAMPION = (
    "Cost$ Exert<1/CARDNAME> | Description$ You may exert CARDNAME as it "
    "attacks. When you do, untap all other creatures you control. | "
    "Mode$ OptionalAttackCost | Trigger$ SV1 | ValidCard$ Card.Self [SEG] "
    "SV1: DB$ UntapAll | SpellDescription$ When you do, untap all other "
    "creatures you control. | ValidCards$ Creature.YouCtrl+StrictlyOther"
)

# ranger_class.provenance.json, line 4 (level 2: AddTrigger$).
_RANGER_CLASS_LEVEL_2 = (
    "AB$ ClassLevelUp | ClassLevel$ EQ1 | Cost$ 1 G | SorcerySpeed$ True | "
    "SpellDescription$ Level 2 | StackDescription$ SpellDescription [SEG] "
    "SV1: AddTrigger$ SV2 | Affected$ Card.Self | ClassLevel$ 2 | "
    "Mode$ Continuous [SEG] SV2: AttackingPlayer$ You | Execute$ SV3 | "
    "Mode$ AttackersDeclared | Secondary$ True [SEG] SV3: CounterNum$ 1 | "
    "CounterType$ P1P1 | DB$ PutCounter"
)

# ranger_class.provenance.json, line 5 (level 3: AddStaticAbility$ SV2 & SV3).
_RANGER_CLASS_LEVEL_3 = (
    "AB$ ClassLevelUp | ClassLevel$ EQ2 | Cost$ 3 G | SorcerySpeed$ True | "
    "SpellDescription$ Level 3 | StackDescription$ SpellDescription [SEG] "
    "SV1: AddStaticAbility$ SV2 & SV3 | Affected$ Card.Self | ClassLevel$ 3 "
    "| Mode$ Continuous [SEG] SV2: Affected$ Card.TopLibrary+YouCtrl | "
    "MayLookAt$ You | Mode$ Continuous | Secondary$ True [SEG] SV3: "
    "Affected$ Creature.nonLand+TopLibrary+YouCtrl | MayPlay$ True | "
    "Mode$ Continuous | Secondary$ True"
)


@pytest.fixture(scope="module")
def tokenizer() -> AbilityTokenizer:
    vocab = {t: i for i, t in enumerate(
        ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEG]", "cardname", *_WORDS],
    )}
    return AbilityTokenizer(vocab, {}, surface="script")


@pytest.mark.parametrize("script_text", [
    _CONTACT_OTHER_PLANE, _AHN_CROP_CHAMPION,
    _RANGER_CLASS_LEVEL_2, _RANGER_CLASS_LEVEL_3,
])
def test_every_chain_label_is_one_token(tokenizer, script_text):
    texts = [token.text for token in tokenizer.tokenize(script_text)]
    labels = [label.lower() for label in re.findall(r"SV\d+", script_text)]
    assert [t for t in texts if re.fullmatch(r"sv\d+", t)] == labels
    assert "sv" not in texts
