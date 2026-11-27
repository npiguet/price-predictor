"""What a checkpoint or a run sets on the tokenizer: the recorded tokenizer
rules and the withheld keyword."""

from __future__ import annotations

import pytest

from effects.domain.ability_tokenizer import withheld_keyword_tokens


class TestTokenizerRulesByCheckpoint:
    """A loaded checkpoint is read with the grammar it trained under, and one
    that records none, or rules this code does not know, is refused."""

    def test_a_checkpoint_recording_no_rules_is_refused(self):
        from effects.domain.ability_tokenizer import tokenizer_surface
        from effects.domain.recorded_version import UnknownRecordedVersion

        for settings in ({}, None):
            with pytest.raises(UnknownRecordedVersion, match="records no tokenizer_rules"):
                tokenizer_surface("script", settings)

    def test_a_checkpoint_reads_its_vocabularys_surface(self):
        from effects.domain.ability_tokenizer import tokenizer_surface

        settings = {"tokenizer_rules": "gen-3"}
        assert tokenizer_surface("script", settings) == "script"
        assert tokenizer_surface("prose", settings) == "prose"

    def test_rules_this_code_does_not_know_are_refused(self):
        from effects.domain.ability_tokenizer import tokenizer_surface
        from effects.domain.recorded_version import UnknownRecordedVersion

        # gen-2 read a variant's script by the prose grammar on the prose
        # surface, descriptions and all.
        for rules in ("gen-2", "gen-4"):
            with pytest.raises(UnknownRecordedVersion, match=rules):
                tokenizer_surface("script", {"tokenizer_rules": rules})


class TestWithheldKeyword:
    def test_no_withholding_hides_nothing(self):
        assert withheld_keyword_tokens(None) == frozenset()

    def test_the_token_form_matches_the_vocabularys(self):
        assert withheld_keyword_tokens("First Strike") == {"first_strike"}
