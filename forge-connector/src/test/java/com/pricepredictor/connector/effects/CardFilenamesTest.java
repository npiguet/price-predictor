package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Java/Python agreement on card filenames.
 *
 * <p>A provenance key's {@code script_file} is written here and resolved by
 * {@code sealed.infrastructure.converted_card_locator}, and the join fails
 * loudly on a mismatch. The cases below are the ones the Python suite pins in
 * {@code tests/unit/sealed/infrastructure/test_converted_card_locator.py}, so a
 * change to either sanitizer that is not made to both shows up here.
 */
class CardFilenamesTest {

    @ParameterizedTest
    @CsvSource({
            "Lightning Bolt,           lightning_bolt",
            "Urza's Saga,              urzas_saga",
            "Dandân,                   dandan",
            "Glassworks // Shattered Yard, glassworks_shattered_yard",
            "'Borborygmos, Enraged!',  borborygmos_enraged",
            "Anchovy & Banana Pizza,   anchovy_banana_pizza",
            "'Don & Leo, Problem Solvers', don_leo_problem_solvers",
            "Lim-Dûl's Vault,          lim_duls_vault",
            "Ajani's Pridemate,        ajanis_pridemate",
    })
    void sanitizeMatchesThePythonSanitizer(String cardName, String expected) {
        assertEquals(expected, CardFilenames.sanitize(cardName));
    }

    @Test
    void aCardsfolderPathIsLetterKeyed() {
        assertEquals("cardsfolder/a/ajanis_pridemate.txt",
                CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "Ajani's Pridemate"));
    }

    @Test
    void aTokenPathIsFlat() {
        assertEquals("tokenscripts/ajanis_pridemate.txt",
                CardFilenames.scriptFile(SourceTree.TOKENSCRIPTS, "Ajani's Pridemate"));
    }

    @Test
    void aVariantPathIsFlatToo() {
        assertEquals("variant-scripts/serra_angel.txt",
                CardFilenames.scriptFile(SourceTree.VARIANT_SCRIPTS, "Serra Angel"));
    }

    @Test
    void rebalancedCardsKeepTheirLiteralPrefixAndOwnDirectory() {
        // The general sanitizer would turn the hyphen into an underscore.
        assertEquals("cardsfolder/rebalanced/a-akki_ronin.txt",
                CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "A-Akki Ronin"));
    }

    @Test
    void aPathAlwaysUsesForwardSlashes() {
        String path = CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "Serra Angel");
        assertEquals(-1, path.indexOf('\\'), path);
    }

    @Test
    void theThreeTreesAreKnownAndTwoAreFlat() {
        assertEquals(true, SourceTree.isKnown(SourceTree.CARDSFOLDER));
        assertEquals(false, SourceTree.isFlat(SourceTree.CARDSFOLDER));
        assertEquals(true, SourceTree.isFlat(SourceTree.TOKENSCRIPTS));
        assertEquals(true, SourceTree.isFlat(SourceTree.VARIANT_SCRIPTS));
        assertEquals(false, SourceTree.isKnown("cardsfolder-512"));
    }
}
