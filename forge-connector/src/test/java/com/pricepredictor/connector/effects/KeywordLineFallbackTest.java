package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A keyword's line carries the script of the trait whose text it prints, and
 * a keyword-built trait whose script states nothing still prints its Oracle
 * sentence.
 *
 * <p>A several-line keyword's line is claimed by its built traits in index
 * order, so without the rendering's own choice the first claimant's chain
 * won: haunt's line carried a description-less trigger that ends the
 * haunting. A chapter or an ETB replacement whose effect has no
 * {@code SpellDescription$} printed its numerals alone, or nothing, and a
 * record of it read no text.
 */
@ExtendWith(ForgeExtension.class)
class KeywordLineFallbackTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private static ConvertedScript convert(String relativePath) {
        return ConvertedScript.fromFile(new RulesParser(), CARDS_FOLDER, relativePath);
    }

    /** The sidecar line whose text is {@code text}. */
    private static ProvenanceSidecar.Line lineOf(ConvertedScript card, String text) {
        List<ProvenanceSidecar.Line> lines = card.sidecar().lines().stream()
                .filter(line -> card.texts().get(line.lineIndex()).equals(text)).toList();
        assertEquals(1, lines.size(), text + " in " + card.texts());
        return lines.get(0);
    }

    @Test
    void theHauntLineCarriesTheHauntingTriggersChain() {
        // Triggers 0 and 1 end the haunting (TriggerDescription$ Blank) and
        // claim the line first; trigger 3, which haunts, prints it.
        ConvertedScript pontiff = convert("o/orzhov_pontiff.txt");
        String script = lineOf(pontiff, "triggered: haunt").script().scriptText();
        assertTrue(script.contains("TriggerDescription$ Haunt"), script);
        assertFalse(script.contains("TriggerDescription$ Blank"), script);
    }

    @Test
    void aChapterWhoseEffectStatesNothingPrintsItsOracleSentence() {
        // Oracle: "II, III — Each player returns two land cards from their
        // graveyard to the battlefield."
        assertTrue(convert("f/fall_of_the_thran.txt").texts().contains(
                "chapter: II, III — each player returns two land cards from their graveyard"
                        + " to the battlefield."));
        // Oracle: "I — For each opponent, you create a 1/1 white Human Soldier
        // creature token."
        assertTrue(convert("v/vault_11_voters_dilemma.txt").texts().contains(
                "chapter: I — for each opponent, you create a 1/1 white human soldier"
                        + " creature token."));
    }

    @Test
    void anEtbReplacementWithNoDescriptionPrintsItsOracleSentenceUnderItsBareKey() {
        // K:ETBReplacement:Other:ETBPrep, whose effect has no SpellDescription$.
        ConvertedScript ghoul = convert("s/sutured_ghoul.txt");
        ProvenanceKey keyword = new ProvenanceKey(
                "cardsfolder/s/sutured_ghoul.txt", 0, TraitKind.KEYWORD, 0);
        ProvenanceSidecar.Line line = lineOf(ghoul,
                "replacement: as CARDNAME enters, exile any number of creature cards"
                        + " from your graveyard.");
        assertEquals(List.of(keyword), line.provenance());
        assertFalse(ghoul.sidecar().droppedKeys().contains(keyword),
                "dropped: " + ghoul.sidecar().droppedKeys());
    }
}
