"""Which text a line is encoded from, on each surface.

The surface itself is the vocabulary file's header
(``tests/unit/effects/infrastructure/test_vocabulary_header.py``).
"""

from __future__ import annotations

from effects.domain.ability_encoder import encoding_text
from effects.domain.ability_tokenizer import Surface
from effects.domain.script_text import script_reading


class _Line:
    def __init__(self, script_text: str) -> None:
        self.script_text = script_text


class TestPairingLoss:
    def test_a_variant_has_only_the_script_surface(self):
        """So it contributes no pairing term: there is no prose to pair with,
        and both surfaces read its script reading (finding 1.5)."""
        line = _Line("NumDmg$ 7")
        assert encoding_text(line, None, Surface.SCRIPT) == script_reading("NumDmg$ 7")
        assert encoding_text(line, None, Surface.PROSE) == script_reading("NumDmg$ 7")

    def test_a_real_card_has_both_surfaces_and_can_pair(self):
        line = _Line("NumDmg$ 3")
        prose = "deal 3 damage to any target"
        assert encoding_text(line, prose, Surface.SCRIPT) == "NumDmg$ 3"
        assert encoding_text(line, prose, Surface.PROSE) == prose
