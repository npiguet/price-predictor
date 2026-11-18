package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Forge's line-break marker ({@code ,,,} and {@code ,,,,,,}) in a description
 * reads as one space on its line, as Forge's own stack text reads it.
 */
@ExtendWith(ForgeExtension.class)
class LineBreakMarkerTest {

    private static final Path CARDS = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser parser = new RulesParser();

    private List<String> texts(String relativePath) {
        List<String> lines = ConvertedScript.fromFile(parser, CARDS, relativePath).texts();
        for (String line : lines) assertFalse(line.contains(",,,"), line);
        return lines;
    }

    @Test
    void aSpellLineReadsTheMarkerAsASpace() {
        // Bounty of Might: "Target creature gets +3/+3 until end of turn." three times.
        assertTrue(texts("b/bounty_of_might.txt").contains(
                "spell[1]: target creature gets +3/+3 until end of turn."
                        + " target creature gets +3/+3 until end of turn."
                        + " target creature gets +3/+3 until end of turn."));
    }

    @Test
    void aStaticLineReadsTheMarkerAsASpace() {
        // Coralhelm Commander: "LEVEL 4+ / 4/4 / Flying / Other Merfolk creatures you control get +1/+1."
        assertTrue(texts("c/coralhelm_commander.txt").contains(
                "static: level 4+ 4/4 flying other merfolk creatures you control get +1/+1."));
    }

    @Test
    void anActivatedLineReadsTheMarkerAsASpace() {
        // Item Crate: "... with the listed name and ability. • Banana with ..."
        List<String> lines = texts("i/item_crate.txt");
        assertTrue(lines.stream().anyMatch(l -> l.contains(
                "with the listed name and ability. • banana with \"{T}, sacrifice this token:")),
                lines.toString());
    }

    @Test
    void aChapterLineReadsTheMarkerAsASpace() {
        // The Battle of Dragon Brothers: chapter I's modes "Bolas — ..." and "Ugin — ..."
        List<String> lines = texts("t/the_battle_of_dragon_brothers_fate_reforged.txt");
        assertTrue(lines.stream().anyMatch(l -> l.contains(
                "you draw seven cards. ugin — manifest your hand")), lines.toString());
    }
}
