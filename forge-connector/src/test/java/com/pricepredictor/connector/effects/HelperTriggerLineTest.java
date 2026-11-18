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
}
