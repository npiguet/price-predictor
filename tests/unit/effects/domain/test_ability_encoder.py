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
