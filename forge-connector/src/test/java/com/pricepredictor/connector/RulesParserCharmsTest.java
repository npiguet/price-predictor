package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.pricepredictor.connector.AbilityRendering.lineOf;
import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static com.pricepredictor.connector.FaceAbilities.allOptions;
import static com.pricepredictor.connector.FaceAbilities.rendered;
import static org.junit.jupiter.api.Assertions.*;

/** The parser's charms: the choose header, the ABILITY placeholder and each mode's option line. */
@ExtendWith(ForgeExtension.class)
class RulesParserCharmsTest {

    private final ConvertedCards cards = new ConvertedCards();

    @Test
    void pawprintCharmOptions() {
        CardFace card = cards.face("s/season_of_the_burrow.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());

        // Charm options are sub-abilities of the spell
        var options = spells.get(0).subAbilities();
        assertEquals(3, options.size());
        assertTrue(lineOf(options.get(0)).contains("{P} \u2014"));
        assertTrue(lineOf(options.get(0)).contains("create a 1/1"));
        assertTrue(lineOf(options.get(1)).contains("{P}{P} \u2014"));
        assertTrue(lineOf(options.get(1)).contains("exile target"));
        assertTrue(lineOf(options.get(2)).contains("{P}{P}{P} \u2014"));
        assertTrue(lineOf(options.get(2)).contains("return target"));
    }

    @Test
    void charmWithSubAbilityDescription() {
        // "What Must Be Done" has no SpellDescription on the Charm itself,
        // so choices become top-level OPTION abilities (no parent SPELL).
        var options = allOptions(cards.face("w/what_must_be_done.txt"));
        assertEquals(2, options.size());
        assertTrue(lineOf(options.get(0)).contains("destroy all artifacts and creatures"));
        assertTrue(lineOf(options.get(1)).contains("release juno"));
        assertTrue(lineOf(options.get(1)).contains("return target historic permanent card"));
        assertFalse(lineOf(options.get(1)).contains("Artifacts, legendaries, and Sagas are historic"),
                "Reminder text should be stripped");
    }

    @Test
    void charmChoiceWithNoDescriptionAnywhereIsLeftOut() {
        // A mode with nothing to print has no option line, rather than
        // failing the whole card's conversion.
        var options = allOptions(cards.convert(
                "Name:Bad Charm", "ManaCost:1 U", "Types:Sorcery",
                "A:SP$ Charm | Choices$ DBNoop,DBDraw",
                "SVar:DBNoop:DB$ Pump | Defined$ Self | NumAtt$ 0 | NumDef$ 0",
                "SVar:DBDraw:DB$ Draw | NumCards$ 1 | SpellDescription$ Draw a card.",
                "Oracle:").faces().get(0));
        assertEquals(1, options.size(), "Only the described mode prints: " + options);
        assertEquals("draw a card.", options.get(0).descriptionText());
    }

    @Test
    void pawprintModeWithNoDescriptionPrintsItsCostAlone() {
        var options = allOptions(cards.convert(
                "Name:Paw Charm", "ManaCost:1 G", "Types:Sorcery",
                "A:SP$ Charm | Pawprint$ 3 | Choices$ DBNoop,DBDraw",
                "SVar:DBNoop:DB$ Pump | Defined$ Self | NumAtt$ 0 | NumDef$ 0 | Pawprint$ 2",
                "SVar:DBDraw:DB$ Draw | NumCards$ 1 | Pawprint$ 1 | SpellDescription$ Draw a card.",
                "Oracle:").faces().get(0));
        assertEquals(2, options.size(), "Both modes print: " + options);
        assertEquals("{P}{P}", options.get(0).descriptionText());
        assertEquals("{P} \u2014 draw a card.", options.get(1).descriptionText());
    }

    @Test
    void aTriggeredCharmPrefixesItsModesLikeAModalSpell() {
        // A trigger's modes come through optionsFrom rather than
        // fromSpellAbility, and print the same mode-cost prefix.
        CardFace card = cards.convert(
                "Name:Costly Trigger", "ManaCost:2 W", "Types:Creature Human",
                "PT:2/2",
                "T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield"
                        + " | ValidCard$ Card.Self | Execute$ TrigCharm"
                        + " | TriggerDescription$ When CARDNAME enters, ABILITY",
                "SVar:TrigCharm:DB$ Charm | Choices$ DBDraw,DBGain",
                "SVar:DBDraw:DB$ Draw | NumCards$ 1 | ModeCost$ 1 W | SpellDescription$ Draw a card.",
                "SVar:DBGain:DB$ GainLife | LifeAmount$ 3 | SpellDescription$ Gain 3 life.",
                "Oracle:").faces().get(0);
        var options = allOptions(card);
        assertEquals(2, options.size(), "Both modes print: " + card.abilities());
        assertEquals("{1}{W} \u2014 draw a card.", options.get(0).descriptionText());
        assertEquals("gain 3 life.", options.get(1).descriptionText());
    }

    @Test
    void charmChoiceDescriptionOnSubAbility() {
        // Charm without SpellDescription → choices become top-level OPTION abilities
        var options = allOptions(cards.convert(
                "Name:Chain Charm", "ManaCost:1 G", "Types:Sorcery",
                "A:SP$ Charm | Choices$ DBChainTop,DBDirect",
                "SVar:DBChainTop:DB$ Pump | Defined$ Self | NumAtt$ 0 | NumDef$ 0 | SubAbility$ DBChainDesc",
                "SVar:DBChainDesc:DB$ Draw | NumCards$ 1 | SpellDescription$ Draw a card from the chain.",
                "SVar:DBDirect:DB$ GainLife | LifeAmount$ 3 | SpellDescription$ Gain 3 life.",
                "Oracle:").faces().get(0));
        assertEquals(2, options.size());
        assertTrue(lineOf(options.get(0)).contains("draw a card from the chain"));
        assertTrue(lineOf(options.get(1)).contains("gain 3 life"));
    }

    @Test
    void chooseOneTriggerExpandsAbilityPlaceholder() {
        // Suncleanser: "When CARDNAME enters, ABILITY" where Execute$ is a DB$ Charm.
        // Should emit: TRIGGERED with "choose one" + 2 OPTION sub-abilities.
        CardFace card = cards.face("s/suncleanser.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Should have triggered ability: " + card.abilities());
        Ability t = triggered.get(0);
        assertTrue(t.descriptionText().contains("choose one"),
                "ABILITY placeholder must be replaced with 'choose one': " + t.descriptionText());
        List<Ability> options = t.subAbilities();
        assertEquals(2, options.size(), "Should have 2 charm options: " + options);
        assertTrue(options.stream().allMatch(o -> o.type() == AbilityType.OPTION),
                "Options must be OPTION type: " + options);
        assertTrue(options.stream().anyMatch(o -> o.descriptionText().contains("remove all counters")),
                "Option 1 must mention counter removal: " + options);
        assertTrue(options.stream().anyMatch(o -> o.descriptionText().contains("loses all counters")),
                "Option 2 must mention opponent losing counters: " + options);
    }

    @Test
    void tieredCharmIncludesPrecostDescModeCostAndAdditionalDescription() {
        // Vincent's Limit Break: Tiered charm with PrecostDesc$ on each choice SVar.
        // Option text must include the PrecostDesc name, not just the raw P/T string.
        CardFace card = cards.face("v/vincents_limit_break.txt");
        String formatted = rendered(card);
        assertTrue(formatted.toLowerCase().contains("galian beast"),
                "Tiered charm option should include 'Galian Beast': " + formatted);
        assertTrue(formatted.toLowerCase().contains("death gigas"),
                "Tiered charm option should include 'Death Gigas': " + formatted);
        assertTrue(formatted.toLowerCase().contains("hellmasker"),
                "Tiered charm option should include 'Hellmasker': " + formatted);

        // AdditionalDescription$ on the Charm SA must become the main spell line.
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertTrue(spells.stream().anyMatch(
                a -> a.descriptionText().toLowerCase().contains("until end of turn")),
                "vincent's limit break missing AdditionalDescription: " + spells);
        // And its options include the ModeCost in "Name — {cost} — P/T" format.
        var options = spells.get(0).subAbilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertTrue(options.stream().anyMatch(
                a -> a.descriptionText().contains("{0}") || a.descriptionText().contains("{1}")),
                "vincent's limit break options missing ModeCost: " + options);
    }

    @Test
    void chooseTwoCharmEmitsChooseTwoHeader() {
        // Atarka's Command: CharmNum$ 2 (no MinCharmNum) — exactly two modes.
        // Expected: 1 SPELL with "choose two" in description + 4 OPTION sub-abilities.
        CardFace card = cards.face("a/atarkas_command.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have 1 SPELL: " + card.abilities());
        Ability spell = spells.get(0);
        assertTrue(spell.descriptionText().contains("choose two"),
                "Spell description should contain 'choose two': " + spell.descriptionText());
        List<Ability> options = spell.subAbilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(4, options.size(), "Should have 4 OPTION sub-abilities: " + spell.subAbilities());
    }

    @Test
    void chooseOneOrBothCharmEmitsCorrectHeader() {
        // Against All Odds: MinCharmNum$ 1, CharmNum$ 2 — "choose one or both".
        CardFace card = cards.face("a/against_all_odds.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have 1 SPELL: " + card.abilities());
        assertTrue(spells.get(0).descriptionText().contains("choose one or both"),
                "Spell description should contain 'choose one or both': " + spells.get(0).descriptionText());
    }

    @Test
    void chooseUpToOneCharmEmitsCorrectHeader() {
        // Ertai Resurrected trigger: MinCharmNum$ 0, no CharmNum — "choose up to one".
        // The charm is in a triggered sub-ability; check the full formatted text.
        CardFace card = cards.face("e/ertai_resurrected.txt");
        String formatted = rendered(card);
        assertTrue(formatted.contains("choose up to one"),
                "Output should contain 'choose up to one': " + formatted);
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource({
            // Caesar: ImmediateTrigger → Charm(CharmNum=2)
            "choose two,                  c/caesar_legions_emperor.txt,    '',                        choose two, 3",
            // Cemetery Desecrator: ChangeZone → SubAbility → ImmediateTrigger → Charm
            "nested choose one,           c/cemetery_desecrator.txt,       '',                        choose one, 2",
            // Hylda: ImmediateTrigger → Charm (default choose one); her other trigger is not the charm
            "choose one among triggers,   h/hylda_of_the_icy_crown.txt,    tap an untapped creature,  choose one, 3",
    })
    void immediatelyTriggeredCharmExpandsAbilityPlaceholder(
            String ignoredCase, String file, String triggerText, String chooseText, int optionCount) {
        // TriggerDescription ends with "ABILITY", which must be replaced by the charm's header.
        CardFace card = cards.face(file);
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty());
        Ability t = triggered.stream()
                .filter(a -> a.descriptionText().contains(triggerText))
                .findFirst().orElseThrow();
        assertFalse(t.descriptionText().contains("ABILITY"), "ABILITY must be replaced: " + t.descriptionText());
        assertTrue(t.descriptionText().contains(chooseText),
                "Must say '" + chooseText + "': " + t.descriptionText());
        List<Ability> options = t.subAbilities();
        assertEquals(optionCount, options.size(), "Must have " + optionCount + " options: " + options);
        assertTrue(options.stream().allMatch(o -> o.type() == AbilityType.OPTION));
    }

    @Test
    void hauntCharmExpandsAbilityPlaceholder() {
        // Orzhov Pontiff: the Haunt SVar is a DB$ Charm with no SpellDescription, so the
        // ABILITY placeholder expands to the charm's "choose one" header and its options.
        CardFace card = cards.face("o/orzhov_pontiff.txt");
        // The haunted-dies triggered ability should mention "choose one", not "ABILITY"
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        Ability hauntTrig = triggered.stream()
                .filter(a -> a.descriptionText().contains("haunts dies"))
                .findFirst().orElseThrow();
        assertFalse(hauntTrig.descriptionText().contains("ABILITY"),
                "ABILITY must be replaced: " + hauntTrig.descriptionText());
        assertTrue(hauntTrig.descriptionText().contains("choose one"),
                "Must say 'choose one': " + hauntTrig.descriptionText());
        assertEquals(2, hauntTrig.subAbilities().size(), "Must have 2 options: " + hauntTrig.subAbilities());
        assertTrue(hauntTrig.subAbilities().stream().allMatch(o -> o.type() == AbilityType.OPTION));
    }

    @Test
    void chapterCharmExpandsAbilityPlaceholder() {
        // Life of Toshiro: Chapter I/II execute is DB$ Charm, SpellDescription$ ABILITY
        MultiCard card = cards.convertFromFile("l/life_of_toshiro_umezawa_memory_of_toshiro.txt");
        CardFace front = card.faces().get(0);
        var chapters = abilitiesOfType(front, AbilityType.CHAPTER);
        // Find chapter I/II (the charm chapter)
        Ability charmChapter = chapters.stream()
                .filter(c -> c.descriptionText().contains("I") && c.descriptionText().contains("II"))
                .findFirst().orElseThrow();
        assertFalse(charmChapter.descriptionText().contains("ABILITY"),
                "ABILITY must be replaced: " + charmChapter.descriptionText());
        assertTrue(charmChapter.descriptionText().contains("choose one"),
                "Must say 'choose one': " + charmChapter.descriptionText());
        assertEquals(3, charmChapter.subAbilities().size(),
                "Must have 3 charm options: " + charmChapter.subAbilities());
        assertTrue(charmChapter.subAbilities().stream().allMatch(o -> o.type() == AbilityType.OPTION));
    }
}
