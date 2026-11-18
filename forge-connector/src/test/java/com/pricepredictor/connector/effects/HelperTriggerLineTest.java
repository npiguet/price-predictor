package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A static secondary trigger, which Forge's card view never shows, prints no
 * line: its description is a script author's note or repeats a static's text.
 */
@ExtendWith(ForgeExtension.class)
class HelperTriggerLineTest {

    private static final Path CARDS = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser parser = new RulesParser();

    private static ProvenanceKey trigger(String relativePath, int index) {
        return new ProvenanceKey("cardsfolder/" + relativePath, 0, TraitKind.TRIGGER, index);
    }

    @Test
    void aBookkeepingTriggerPrintsNoLineAndIsDropped() {
        // Diabolic Servitude's three "Forget remembered card if it goes into ..." triggers.
        String path = "d/diabolic_servitude.txt";
        ConvertedScript servitude = ConvertedScript.fromFile(parser, CARDS, path);

        assertEquals(List.of(
                "name: diabolic servitude",
                "mana cost: {3}{B}",
                "types: enchantment",
                "triggered: when CARDNAME enters, return target creature card from your"
                        + " graveyard to the battlefield.",
                "triggered: when the creature put onto the battlefield with CARDNAME dies,"
                        + " exile it and return CARDNAME to its owner's hand.",
                "triggered: when CARDNAME leaves the battlefield, exile the creature put onto"
                        + " the battlefield with CARDNAME."), servitude.texts());
        // Its permanent spell states nothing, as every permanent's.
        ProvenanceKey spell = new ProvenanceKey("cardsfolder/" + path, 0, TraitKind.SPELL, 0);
        assertEquals(List.of(spell, trigger(path, 3), trigger(path, 4), trigger(path, 5)),
                servitude.sidecar().droppedKeys());
    }

    @Test
    void theStaticAHelperRepeatedPrintsItsOwnLine() {
        // Serra Paragon: "Once during each of your turns, you may play a land from your graveyard ..."
        List<String> lines = ConvertedScript.fromFile(parser, CARDS, "s/serra_paragon.txt").texts();
        assertTrue(lines.contains("static: once during each of your turns, you may play a land"
                + " from your graveyard or cast a permanent spell with mana value 3 or less from"
                + " your graveyard. if you do, it gains \"when this permanent is put into a"
                + " graveyard from the battlefield, exile it and you gain 2 life.\""), lines.toString());
        assertTrue(lines.stream().noneMatch(l -> l.startsWith("replacement:")), lines.toString());
    }
    @Test
    void aSecondaryTriggerAnotherLineRepeatsPrintsNoLine() {
        // Coffin Queen's three "when ... exile creature controlled by CARDNAME" triggers
        // restate its activated line's "when CARDNAME becomes untapped or you lose control of ...".
        String path = "c/coffin_queen.txt";
        ConvertedScript queen = ConvertedScript.fromFile(parser, CARDS, path);

        assertEquals(List.of(
                "name: coffin queen",
                "mana cost: {2}{B}",
                "types: creature zombie wizard",
                "power toughness: 1/1",
                "static: you may choose not to untap CARDNAME during your untap step.",
                "activated[1]: {2}{B}, {T}: put target creature card from a graveyard onto the"
                        + " battlefield under your control. when CARDNAME becomes untapped or you"
                        + " lose control of CARDNAME, exile that creature."), queen.texts());
        assertTrue(queen.sidecar().droppedKeys().containsAll(
                List.of(trigger(path, 0), trigger(path, 1), trigger(path, 2), trigger(path, 3))),
                queen.sidecar().droppedKeys().toString());
    }

    @Test
    void aSecondaryTriggerWithTextOfItsOwnKeepsItsLine() {
        // Free for All's upkeep trigger is secondary, and no other line states it.
        List<String> lines = ConvertedScript.fromFile(parser, CARDS, "f/free_for_all.txt").texts();
        assertTrue(lines.contains("triggered: at the beginning of each player's upkeep, that"
                + " player chooses a card exiled with CARDNAME at random and puts it onto the"
                + " battlefield."), lines.toString());
    }

    @Test
    void aSelfNameWrittenInTheScriptsOwnCaseReadsCardname() {
        List<String> lines = ConvertedScript.fromFile(parser, CARDS, "f/friendly_fire.txt").texts();
        assertTrue(lines.stream().anyMatch(l -> l.contains("CARDNAME deals damage")), lines.toString());
    }
}
