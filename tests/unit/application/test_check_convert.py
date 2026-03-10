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
