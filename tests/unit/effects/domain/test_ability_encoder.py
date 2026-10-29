"""Which encoding surface a vocabulary path implies."""

from __future__ import annotations

from effects.domain.ability_encoder import surface_of
from effects.domain.ability_tokenizer import Surface


class TestSurfaceFollowsTheVocabulary:
    def test_a_script_vocabulary_path_selects_the_script_surface(self):
        """The surface follows the loaded --vocab-path rather than a flag of
        its own, so the two cannot disagree."""
        assert surface_of("models/effects/vocab-script.txt") == Surface.SCRIPT
        assert surface_of("models/effects/vocab.txt") == Surface.PROSE

    def test_an_unrecognised_path_reads_as_prose(self):
        assert surface_of("models/effects/anything.txt") == Surface.PROSE


class TestPairingLoss:
    def test_a_variant_has_only_the_script_surface(self):
        """So it contributes no pairing term: there is no prose to pair with."""
        from effects.domain.ability_encoder import encoding_text
        from effects.domain.ability_tokenizer import Surface

        class _Line:
            script_text = "NumDmg$ 7"

        line = _Line()
        assert encoding_text(line, None, Surface.SCRIPT) == "NumDmg$ 7"
        # No prose exists, so the prose surface falls back to the script rather
        # than pairing two different readings of one line.
        assert encoding_text(line, None, Surface.PROSE) == "NumDmg$ 7"

    def test_a_real_card_has_both_surfaces_and_can_pair(self):
        from effects.domain.ability_encoder import encoding_text
        from effects.domain.ability_tokenizer import Surface

        class _Line:
            script_text = "NumDmg$ 3"

        line = _Line()
        prose = "deal 3 damage to any target"
        assert encoding_text(line, prose, Surface.SCRIPT) == "NumDmg$ 3"
        assert encoding_text(line, prose, Surface.PROSE) == prose
