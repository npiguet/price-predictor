"""The value head's targets come from a line's script, whatever surface encodes it.

The prose surface encodes "deals 3 damage to any target." and holds no ``$``;
reading the amounts from it would train every prose line on stated zeros
(FR-058).
"""

from __future__ import annotations

from effects.application.step_preparation import value_batch
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.value_targets import VALUE_TARGET_NAMES

_KEY = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
_LINE = SidecarLine(
    line_index=0, line_kind="spell", provenance=(_KEY,),
    script_text="SP$ DealDamage | NumDmg$ 2 | ValidTgts$ Any",
)


def test_a_prose_texts_amounts_come_from_its_lines_script():
    prose = "cardname deals 2 damage to any target."
    values, mask = value_batch([prose], {prose: _LINE})
    damage = VALUE_TARGET_NAMES.index("damage")
    assert bool(mask[0, damage])
    assert float(values[0, damage]) == 2.0


def test_a_text_with_no_line_reads_itself():
    script = "SP$ DealDamage | NumDmg$ 4"
    values, _mask = value_batch([script], {})
    assert float(values[0, VALUE_TARGET_NAMES.index("damage")]) == 4.0
