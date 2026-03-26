"""Tests for the check-convert module."""

from pathlib import Path

import pytest

from price_predictor.application.check_convert import (
    CardCheckResult,
    check_all,
    check_card,
    format_report,
)


class TestCheckCard:
    """Tests for check_card()."""

    def test_high_similarity_for_matching_card(self):
        forge = (
            "Name:Laboratory Maniac\n"
            "ManaCost:2 U\n"
            "Types:Creature Human Wizard\n"
            "PT:2/2\n"
            "Oracle:If you would draw a card while your library has no cards "
            "in it, you win the game instead.\n"
        )
        converted = (
            "name: laboratory maniac\n"
            "mana cost: {2}{U}\n"
            "types: creature human wizard\n"
            "power toughness: 2/2\n"
            "replacement: if you would draw a card while your library has "
            "no cards in it, you win the game instead.\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.8
        assert not result.duplicate_lines
        assert not result.empty_lines

    def test_low_similarity_for_mismatched_card(self):
        forge = (
            "Name:Lightning Bolt\n"
            "ManaCost:R\n"
            "Types:Instant\n"
            "Oracle:Lightning Bolt deals 3 damage to any target.\n"
        )
        converted = (
            "name: lightning bolt\n"
            "mana cost: {R}\n"
            "types: instant\n"
            "spell[1]: something completely different and unrelated.\n"
        )
        result = check_card(converted, forge)
        assert result.similarity < 0.5

    def test_detects_duplicate_lines(self):
        forge = (
            "Name:Test Card\n"
            "Types:Creature\n"
            "Oracle:Flying\n"
        )
        converted = (
            "name: test card\n"
            "types: creature\n"
            "keyword: flying\n"
            "keyword: flying\n"
        )
        result = check_card(converted, forge)
        assert len(result.duplicate_lines) == 1

    def test_no_oracle_text_passes(self):
        forge = (
            "Name:Grizzly Bears\n"
            "ManaCost:1 G\n"
            "Types:Creature Bear\n"
            "PT:2/2\n"
            "Oracle:\n"
        )
        converted = (
            "name: grizzly bears\n"
            "mana cost: {1}{G}\n"
            "types: creature bear\n"
            "power toughness: 2/2\n"
        )
        result = check_card(converted, forge)
        assert result.similarity == 1.0
        assert not result.has_oracle

    def test_oracle_but_no_abilities_flags(self):
        forge = (
            "Name:Test Card\n"
            "Types:Instant\n"
            "Oracle:Deal 3 damage to any target.\n"
        )
        converted = (
            "name: test card\n"
            "types: instant\n"
        )
        result = check_card(converted, forge)
        assert result.similarity == 0.0
        assert result.converted_lines == 0

    def test_reminder_text_stripped_from_oracle(self):
        forge = (
            "Name:Test Creature\n"
            "Types:Creature\n"
            "Oracle:Trample (This creature can deal excess combat damage "
            "to the player it's attacking.)\n"
        )
        converted = (
            "name: test creature\n"
            "types: creature\n"
            "keyword: trample\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.8

    def test_cardname_replacement_in_oracle(self):
        forge = (
            "Name:Labyrinth Champion\n"
            "Types:Creature Human Warrior\n"
            "Oracle:Heroic \u2014 Whenever you cast a spell that targets "
            "Labyrinth Champion, Labyrinth Champion deals 2 damage to any target.\n"
        )
        converted = (
            "name: labyrinth champion\n"
            "types: creature human warrior\n"
            "triggered: heroic \u2014 whenever you cast a spell that targets "
            "CARDNAME, CARDNAME deals 2 damage to any target.\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.8

    def test_order_independent_comparison(self):
        forge = (
            "Name:Test Card\n"
            "Types:Creature\n"
            "Oracle:Flying\\nVigilance\n"
        )
        # Converted has abilities in reverse order
        converted = (
            "name: test card\n"
            "types: creature\n"
            "keyword: vigilance\n"
            "keyword: flying\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.9

    def test_land_intrinsic_mana_included(self):
        forge = (
            "Name:Forest\n"
            "ManaCost:no cost\n"
            "Types:Basic Land Forest\n"
            "Oracle:({T}: Add {G}.)\n"
        )
        converted = (
            "name: forest\n"
            "types: basic land forest\n"
            "activated[1]: {T}: add {G}\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.7
        assert result.has_oracle

    def test_dual_land_intrinsic_mana(self):
        forge = (
            "Name:Bayou\n"
            "ManaCost:no cost\n"
            "Types:Land Swamp Forest\n"
            "Oracle:({T}: Add {B} or {G}.)\n"
        )
        converted = (
            "name: bayou\n"
            "types: land swamp forest\n"
            "activated[1]: {T}: add {B} or {G}\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.7

    def test_reminder_text_stripped_from_oracle_lines(self):
        forge = (
            "Name:Test Flyer\n"
            "Types:Creature\n"
            "Oracle:Flying (This creature can deal excess combat damage.)\\n"
            "Trample (It can trample.)\n"
        )
        converted = (
            "name: test flyer\n"
            "types: creature\n"
            "keyword: flying\n"
            "keyword: trample\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.9

    def test_landwalk_portmanteau_normalized(self):
        # Oracle uses portmanteau forms ("Swampwalk", "Forestwalk", etc.)
        # but our converter outputs the split form ("landwalk swamp", "landwalk forest").
        # The checker must normalize oracle portmanteaus before comparing.
        for oracle_word, converted_word in [
            ("Swampwalk", "landwalk swamp"),
            ("Forestwalk", "landwalk forest"),
            ("Islandwalk", "landwalk island"),
            ("Mountainwalk", "landwalk mountain"),
            ("Plainswalk", "landwalk plains"),
            ("Desertwalk", "landwalk desert"),
            ("Legendary landwalk", "landwalk legendary land"),
            ("Nonbasic landwalk", "landwalk nonbasic land"),
        ]:
            forge = f"Name:Test Card\nTypes:Creature\nOracle:{oracle_word}\n"
            converted = f"name: test card\ntypes: creature\nstatic: {converted_word}\n"
            result = check_card(converted, forge)
            assert result.similarity > 0.8, (
                f"Oracle '{oracle_word}' vs converted '{converted_word}' "
                f"should normalize to same text, got {result.similarity:.2%}"
            )

    def test_text_key_counted_as_ability_line(self):
        # "text" is intentionally NOT in _HEADER_KEYS so text: values are treated
        # as oracle-relevant content (conspiracy abilities, casting restrictions, etc.)
        forge = (
            "Name:Conspiracy Card\n"
            "Types:Conspiracy\n"
            "Oracle:Hidden agenda (Start the game with this conspiracy face down in the command zone.)\n"
        )
        converted = (
            "name: conspiracy card\n"
            "types: conspiracy\n"
            "text: hidden agenda\n"
        )
        result = check_card(converted, forge)
        assert result.converted_lines == 1, "text: line must be counted as an ability line"
        assert result.similarity > 0.8

    def test_additional_cost_prefix_stripped_from_oracle(self):
        # Oracle says "As an additional cost to cast this spell, sacrifice a creature."
        # but our converter outputs just "sacrifice a creature." under the key "additional cost:".
        # The checker must strip the oracle prefix so both sides compare equal.
        forge = (
            "Name:Bone Splinters\n"
            "ManaCost:B\n"
            "Types:Sorcery\n"
            "Oracle:As an additional cost to cast this spell, sacrifice a creature."
            "\\nDestroy target creature.\n"
        )
        converted = (
            "name: bone splinters\n"
            "mana cost: {B}\n"
            "types: sorcery\n"
            "additional cost: sacrifice a creature.\n"
            "spell[1]: destroy target creature.\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.8, (
            f"Additional cost prefix should be stripped from oracle before comparison, "
            f"got {result.similarity:.2%}"
        )

    def test_long_identical_strings_score_high(self):
        # Word-bag Jaccard on long but identical ability text must still score high.
        long_ability = (
            "whenever a creature enters the battlefield under your control, "
            "you may pay {1}{G}. if you do, draw a card and you gain 1 life. "
            "this ability triggers only once each turn and cannot be countered "
            "by spells or abilities your opponents control during their turns."
        )
        forge = f"Name:Long Card\nTypes:Enchantment\nOracle:{long_ability}\n"
        converted = f"name: long card\ntypes: enchantment\nstatic: {long_ability.lower()}\n"
        result = check_card(converted, forge)
        assert result.similarity > 0.8, (
            f"Long similar strings should score high similarity, got {result.similarity:.2%}."
        )

    def test_protection_from_normalized_to_match_converter(self):
        # Oracle says "Protection from creatures" but converter outputs "protection:creature"
        # (Forge internal format: no "from", singular type).  The checker must normalise
        # "protection from Xs" → "protection X" so both sides compare equal.
        for oracle_text, converted_text in [
            ("Protection from creatures",     "protection:creature"),
            ("Protection from artifacts",     "protection:artifact"),
            ("Protection from enchantments",  "protection:enchantment"),
            ("Protection from Dragons",       "protection:dragon"),
            ("Protection from instants",      "protection:instant"),
            ("Protection from spells",        "protection:spell"),
            ("Protection from multicolored",  "protection:multicolored"),
            ("Protection from monocolored",   "protection:monocolored"),
            ("Protection from everything",    "protection:everything"),
        ]:
            forge = f"Name:Test Card\nTypes:Creature\nOracle:{oracle_text}\n"
            converted = f"name: test card\ntypes: creature\nstatic: {converted_text}\n"
            result = check_card(converted, forge)
            assert result.similarity > 0.8, (
                f"Oracle '{oracle_text}' vs converted '{converted_text}' "
                f"should normalise to same text, got {result.similarity:.2%}"
            )

    def test_protection_plus_keyword_normalized(self):
        # "Flying, protection from enchantments." (one oracle line split into two)
        # must compare well against two separate static: lines.
        forge = (
            "Name:Azorius First-Wing\n"
            "Types:Creature Griffin\n"
            "Oracle:Flying, protection from enchantments.\n"
        )
        converted = (
            "name: azorius first-wing\n"
            "types: creature griffin\n"
            "static: flying\n"
            "static: protection:enchantment\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.8, (
            f"Flying + protection from enchantments should normalise correctly, "
            f"got {result.similarity:.2%}"
        )

    def test_perfect_similarity_with_intentional_duplicate_lines_not_flagged(self):
        # Cards like Bounty of Might have 3 identical oracle lines (same effect,
        # different targets). The converter correctly emits 3 identical spell[N]: lines.
        # duplicate_lines fires, but similarity is 100%: should NOT count as an issue.
        oracle_line = "Target creature gets +3/+3 until end of turn."
        forge = (
            f"Name:Bounty of Might\nTypes:Instant\n"
            f"Oracle:{oracle_line}\\n{oracle_line}\\n{oracle_line}\n"
        )
        converted = (
            "name: bounty of might\ntypes: instant\n"
            f"spell[1]: {oracle_line.lower()}\n"
            f"spell[2]: {oracle_line.lower()}\n"
            f"spell[3]: {oracle_line.lower()}\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.99, f"Expected ~100% similarity, got {result.similarity:.2%}"
        assert len(result.duplicate_lines) == 2, "Should detect 2 duplicates"
        # The real test: check_all must NOT treat this as an issue.
        from price_predictor.application.check_convert import check_all
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            tp = pathlib.Path(tmp)
            out_dir = tp / "output" / "b"
            cards_dir = tp / "cards" / "b"
            out_dir.mkdir(parents=True)
            cards_dir.mkdir(parents=True)
            (out_dir / "bounty_of_might.txt").write_text(converted)
            (cards_dir / "bounty_of_might.txt").write_text(forge)
            issues = check_all(tp / "output", tp / "cards", threshold=0.5)
        assert len(issues) == 0, (
            f"100%-similarity card with intentional duplicate lines should not be flagged; "
            f"got issues: {issues}"
        )

    def test_specialize_oracle_uses_front_face(self):
        """_extract_oracle() must not overwrite oracle with a SPECIALIZE face's Oracle line."""
        forge = (
            "Name:Test Specialize Card\n"
            "Types:Creature\n"
            "Oracle:Front face oracle text.\n"
            "SPECIALIZE:WHITE\n"
            "Name:Test Specialize Card\n"
            "Types:Creature\n"
            "Oracle:Specialize oracle text that should be ignored.\n"
        )
        converted = (
            "name: test specialize card\n"
            "types: creature\n"
            "spell[1]: front face oracle text.\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.8, (
            f"Should compare against front-face oracle only, got {result.similarity:.2%}"
        )

    def test_multi_face_only_checks_first_face(self):
        forge = (
            "Name:Front Face\n"
            "Types:Creature\n"
            "Oracle:Flying\n"
        )
        converted = (
            "layout: transform\n"
            "name: front face\n"
            "types: creature\n"
            "keyword: flying\n"
            "\n"
            "ALTERNATE\n"
            "\n"
            "name: back face\n"
            "types: creature\n"
            "keyword: totally different ability text that is wrong\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.8


    def test_comma_separated_keywords_split_into_individual_lines(self):
        # Oracle "Flying, reach, trample." is 1 line but converter emits 3 separate
        # static: lines.  The checker must split oracle so line counts match.
        forge = (
            "Name:Dragon Sniper\n"
            "Types:Creature Dragon\n"
            "Oracle:Flying, reach, trample.\n"
        )
        converted = (
            "name: dragon sniper\n"
            "types: creature dragon\n"
            "static: flying\n"
            "static: reach\n"
            "static: trample\n"
        )
        result = check_card(converted, forge)
        assert result.oracle_lines == 3, (
            f"Oracle 'Flying, reach, trample.' should split into 3 lines, "
            f"got {result.oracle_lines}"
        )
        assert result.similarity > 0.8, f"Got {result.similarity:.2%}"

    def test_comma_separated_sentence_not_split(self):
        # "Sacrifice a creature, then draw two cards." should NOT be split —
        # "a" is a stop word that indicates it's a sentence, not a keyword list.
        forge = (
            "Name:Test Card\n"
            "Types:Sorcery\n"
            "Oracle:Sacrifice a creature, then draw two cards.\n"
        )
        converted = (
            "name: test card\n"
            "types: sorcery\n"
            "spell[1]: sacrifice a creature, then draw two cards.\n"
        )
        result = check_card(converted, forge)
        assert result.oracle_lines == 1, (
            f"Sentence with comma should not be split, got {result.oracle_lines} lines"
        )

    def test_card_name_with_comma_not_split(self):
        # Oracle: "{G}: Regenerate Silvos, Rogue Elemental." — the comma is inside the card name.
        # After card-name substitution (Silvos, Rogue Elemental → CARDNAME), there is no comma
        # left, so _split_keyword_line must not produce more than 1 oracle line.
        forge = (
            "Name:Silvos, Rogue Elemental\n"
            "ManaCost:3 G G G\n"
            "Types:Legendary Creature Elemental\n"
            "Oracle:Trample\\n{G}: Regenerate Silvos, Rogue Elemental.\n"
        )
        converted = (
            "name: silvos, rogue elemental\n"
            "mana cost: {3}{G}{G}{G}\n"
            "types: legendary creature elemental\n"
            "static: trample\n"
            "activated[1]: {G}: regenerate CARDNAME.\n"
        )
        result = check_card(converted, forge)
        assert result.oracle_lines == 2, (
            f"Oracle should have 2 lines (trample + regenerate CARDNAME), got {result.oracle_lines}"
        )
        assert result.similarity > 0.8, f"Got {result.similarity:.2%}"

    def test_nickname_in_converter_matches_cardname_in_oracle(self):
        # Converter may output NICKNAME; oracle uses the card name; both should normalize to CARDNAME.
        forge = (
            "Name:Test Card\n"
            "Types:Creature\n"
            "Oracle:Test Card gets +2/+2 until end of turn.\n"
        )
        converted = (
            "name: test card\n"
            "types: creature\n"
            "activated[1]: {1}: NICKNAME gets +2/+2 until end of turn.\n"
        )
        result = check_card(converted, forge)
        assert result.similarity > 0.8, (
            f"NICKNAME in converter should match card name in oracle, got {result.similarity:.2%}"
        )

    def test_developer_note_in_text_not_counted(self):
        # A text: line containing only [Developer's note: …] should not be counted
        # as an ability line for comparison purposes (it is not oracle content).
        forge = (
            "Name:Celestine Cave Witch\n"
            "Types:Creature Human Warlock\n"
            "Oracle:When Celestine Cave Witch enters, create two 1/1 black Insect tokens."
            "\\nWhenever Celestine Cave Witch attacks, you may sacrifice an Insect."
            " When you do, curse defending player.\n"
        )
        converted = (
            "name: celestine cave witch\n"
            "types: creature human warlock\n"
            "text: [developer's note: while intent is clear, the card doesn't work as intended"
            " if the rules text is followed exactly as printed.]\n"
            "triggered: when CARDNAME enters, create two 1/1 black insect tokens.\n"
            "triggered: whenever CARDNAME attacks, you may sacrifice an insect."
            " when you do, curse defending player.\n"
        )
        result = check_card(converted, forge)
        assert result.converted_lines == 2, (
            f"Developer's note text: line should not count as an ability line, "
            f"got {result.converted_lines}"
        )
        assert result.similarity > 0.8, f"Got {result.similarity:.2%}"

    def test_class_level_lines_dropped_from_oracle(self):
        # Class/Talent cards have "{cost}: Level N" lines in oracle that the
        # converter omits entirely.  The checker must drop them before comparison.
        forge = (
            "Name:Blacksmith's Talent\n"
            "Types:Enchantment Class\n"
            "Oracle:First level ability."
            "\\n{1}{R}: Level 2"
            "\\nSecond level ability."
            "\\n{3}{R}: Level 3"
            "\\nThird level ability.\n"
        )
        converted = (
            "name: blacksmith's talent\n"
            "types: enchantment class\n"
            "level[1]: first level ability.\n"
            "level[2]: second level ability.\n"
            "level[3]: third level ability.\n"
        )
        result = check_card(converted, forge)
        assert result.oracle_lines == 3, (
            f"Class level lines should be dropped: expected 3 oracle lines, "
            f"got {result.oracle_lines}"
        )
        assert result.similarity > 0.8, f"Got {result.similarity:.2%}"


class TestCheckAll:
    """Tests for check_all() with filesystem."""

    def test_flags_low_similarity(self, tmp_path: Path):
        out = tmp_path / "output" / "a"
        cards = tmp_path / "cards" / "a"
        out.mkdir(parents=True)
        cards.mkdir(parents=True)

        (cards / "bolt.txt").write_text(
            "Name:Lightning Bolt\nManaCost:R\nTypes:Instant\n"
            "Oracle:Lightning Bolt deals 3 damage to any target.\n"
        )
        (out / "bolt.txt").write_text(
            "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
            "spell[1]: something totally wrong.\n"
        )

        results = check_all(out.parent, cards.parent, threshold=0.5)
        assert len(results) == 1
        assert results[0].similarity < 0.5

    def test_passes_good_cards(self, tmp_path: Path):
        out = tmp_path / "output" / "a"
        cards = tmp_path / "cards" / "a"
        out.mkdir(parents=True)
        cards.mkdir(parents=True)

        (cards / "bolt.txt").write_text(
            "Name:Lightning Bolt\nManaCost:R\nTypes:Instant\n"
            "Oracle:Lightning Bolt deals 3 damage to any target.\n"
        )
        (out / "bolt.txt").write_text(
            "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
            "spell[1]: CARDNAME deals 3 damage to any target.\n"
        )

        results = check_all(out.parent, cards.parent, threshold=0.5)
        assert len(results) == 0

    def test_skips_missing_forge_file(self, tmp_path: Path):
        out = tmp_path / "output" / "a"
        cards = tmp_path / "cards" / "a"
        out.mkdir(parents=True)
        cards.mkdir(parents=True)

        (out / "missing.txt").write_text("name: missing\ntypes: creature\n")

        results = check_all(out.parent, cards.parent, threshold=0.5)
        assert len(results) == 0


class TestFormatReport:
    """Tests for format_report()."""

    def test_empty_results(self):
        assert format_report([]) == "All cards passed checks."

    def test_formats_results(self):
        results = [
            CardCheckResult(
                filename="a/bolt.txt",
                card_name="lightning bolt",
                similarity=0.3,
                oracle_lines=1,
                converted_lines=1,
                duplicate_lines=[],
                empty_lines=False,
                has_oracle=True,
            ),
        ]
        report = format_report(results)
        assert "30.00%" in report
        assert "lightning bolt" in report

    def test_limit(self):
        results = [
            CardCheckResult(
                filename=f"a/card{i}.txt",
                card_name=f"card {i}",
                similarity=0.1 * i,
                oracle_lines=1,
                converted_lines=1,
                duplicate_lines=[],
                empty_lines=False,
                has_oracle=True,
            )
            for i in range(5)
        ]
        report = format_report(results, limit=2)
        assert "showing top 2" in report
        assert "card 0" in report
        assert "card 1" in report
        assert "card 4" not in report
