package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ParameterContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.aggregator.AggregateWith;
import org.junit.jupiter.params.aggregator.ArgumentsAccessor;
import org.junit.jupiter.params.aggregator.ArgumentsAggregator;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.pricepredictor.connector.AbilityRendering.lineOf;
import static com.pricepredictor.connector.ConvertedTextAssertions.assertContainsAll;
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
        assertLinesHold(spells.get(0).subAbilities(), List.of(
                List.of("{P} \u2014", "create a 1/1"),
                List.of("{P}{P} \u2014", "exile target"),
                List.of("{P}{P}{P} \u2014", "return target")));
    }

    @Test
    void charmWithSubAbilityDescription() {
        // "What Must Be Done" has no SpellDescription on the Charm itself,
        // so choices become top-level OPTION abilities (no parent SPELL).
        var options = allOptions(cards.face("w/what_must_be_done.txt"));
        assertLinesHold(options, List.of(
                List.of("destroy all artifacts and creatures"),
                List.of("release juno", "return target historic permanent card")));
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
        assertOptions(t.subAbilities(), 2);
        assertSomeOptionSays(t.subAbilities(), "remove all counters", "loses all counters");
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
            @AggregateWith(ImmediateCharm.FromCsv.class) ImmediateCharm charm) {
        // TriggerDescription ends with "ABILITY", which must be replaced by the charm's header.
        CardFace card = cards.face(charm.file());
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty());
        Ability t = triggered.stream()
                .filter(a -> a.descriptionText().contains(charm.triggerText()))
                .findFirst().orElseThrow();
        assertExpandsToOptions(t, charm.chooseText(), charm.optionCount());
    }

    /**
     * One immediately triggered charm: the trigger of the card in {@code file}
     * whose text holds {@code triggerText} reads {@code chooseText} and has
     * {@code optionCount} options.
     */
    record ImmediateCharm(String file, String triggerText, String chooseText, int optionCount) {

        /** Reads a case from a CSV row whose first column only names the case. */
        static final class FromCsv implements ArgumentsAggregator {
            @Override
            public ImmediateCharm aggregateArguments(ArgumentsAccessor row, ParameterContext context) {
                return new ImmediateCharm(row.getString(1), row.getString(2), row.getString(3), row.getInteger(4));
            }
        }
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
        assertExpandsToOptions(hauntTrig, "choose one", 2);
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
        assertExpandsToOptions(charmChapter, "choose one", 3);
    }

    /**
     * {@code options} are one line per entry of {@code expectedParts}, in
     * order, each line holding every part of its entry.
     */
    private static void assertLinesHold(List<Ability> options, List<List<String>> expectedParts) {
        assertEquals(expectedParts.size(), options.size(), options.toString());
        for (int i = 0; i < options.size(); i++) {
            assertContainsAll(lineOf(options.get(i)), expectedParts.get(i).toArray(String[]::new));
        }
    }

    /** The ability's ABILITY placeholder reads {@code chooseText}, and it has {@code optionCount} options. */
    private static void assertExpandsToOptions(Ability ability, String chooseText, int optionCount) {
        String text = ability.descriptionText();
        assertFalse(text.contains("ABILITY"), "ABILITY must be replaced: " + text);
        assertTrue(text.contains(chooseText), "Must say '" + chooseText + "': " + text);
        assertOptions(ability.subAbilities(), optionCount);
    }

    /** {@code options} are {@code count} abilities, each an OPTION. */
    private static void assertOptions(List<Ability> options, int count) {
        assertEquals(count, options.size(), "Must have " + count + " options: " + options);
        assertTrue(options.stream().allMatch(o -> o.type() == AbilityType.OPTION),
                "Options must be OPTION type: " + options);
    }

    /** Each of {@code texts} is in the description of one of {@code options}. */
    private static void assertSomeOptionSays(List<Ability> options, String... texts) {
        for (String text : texts) {
            assertTrue(options.stream().anyMatch(o -> o.descriptionText().contains(text)),
                    "An option must say '" + text + "': " + options);
        }
    }
}
