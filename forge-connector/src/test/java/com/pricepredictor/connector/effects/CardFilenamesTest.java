package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

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

    // -- the tree decides where a card lives ----------------------------

    @AfterEach
    void forgetTheTemporaryTree() {
        SourceTree.forgetTrees();
    }

    /** A tree holding one ordinary card and the three shapes derivation misses. */
    private static Path tree(Path root) throws IOException {
        file(root, "a/ajanis_pridemate.txt");
        file(root, "p/+2_mace.txt");
        file(root, "n/1996_world_champion.txt");
        file(root, "upcoming/xenobotanist.txt");
        SourceTree.useTree(SourceTree.CARDSFOLDER, root);
        return root;
    }

    private static void file(Path root, String relative) throws IOException {
        Path path = root.resolve(relative);
        Files.createDirectories(path.getParent());
        Files.writeString(path, "Name:test\n");
    }

    /**
     * The defect this lookup exists for. {@code +2 Mace} derives to
     * {@code cardsfolder/+/} and Forge files it under {@code p/}, and because
     * only {@code keyword} keys are checked against their sidecar the wrong
     * path joins to nothing and says nothing about it.
     */
    @Test
    void aCardTheTreeFilesElsewhereGetsThePathTheTreeUses(@TempDir Path root)
            throws IOException {
        tree(root);
        assertEquals("cardsfolder/p/+2_mace.txt",
                CardFilenames.scriptFileForStem(SourceTree.CARDSFOLDER, "+2_mace"));
    }

    /** A digit initial is a directory that exists but holds a different card. */
    @Test
    void aCardFiledUnderALetterItDoesNotStartWithFollowsTheTree(@TempDir Path root)
            throws IOException {
        tree(root);
        assertEquals("cardsfolder/n/1996_world_champion.txt",
                CardFilenames.scriptFileForStem(
                        SourceTree.CARDSFOLDER, "1996_world_champion"));
    }

    /** Unreleased cards sit in one directory whatever their initial. */
    @Test
    void anUpcomingCardKeepsItsOwnDirectory(@TempDir Path root) throws IOException {
        tree(root);
        assertEquals("cardsfolder/upcoming/xenobotanist.txt",
                CardFilenames.scriptFileForStem(
                        SourceTree.CARDSFOLDER, "xenobotanist"));
    }

    /** The 99.7% the convention does describe are untouched by the lookup. */
    @Test
    void aCardFiledUnderItsInitialIsUnchanged(@TempDir Path root) throws IOException {
        tree(root);
        assertEquals("cardsfolder/a/ajanis_pridemate.txt",
                CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "Ajani's Pridemate"));
        assertNull(SourceTree.directoryOf(SourceTree.CARDSFOLDER, "ajanis_pridemate"),
                "the tree only answers where it disagrees with the convention");
    }

    /**
     * A stem the tree does not hold still has to name something: the collector
     * sees cards Forge built without a script, and derivation is the only
     * answer left for them.
     */
    @Test
    void aStemTheTreeDoesNotHoldFallsBackToTheConvention(@TempDir Path root)
            throws IOException {
        tree(root);
        assertEquals("cardsfolder/s/serra_angel.txt",
                CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "Serra Angel"));
    }

    /** An unlocatable tree must not take the collector down with it. */
    @Test
    void anAbsentTreeLeavesTheConventionInPlace(@TempDir Path root) {
        SourceTree.useTree(SourceTree.CARDSFOLDER, root.resolve("no-such-tree"));
        assertEquals("cardsfolder/s/serra_angel.txt",
                CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "Serra Angel"));
    }

    // -- the path the card reader recorded ------------------------------

    /**
     * The authoritative route: what the loader read, relativised the way the
     * converter relativises it. Anchoring on the tree segment rather than on a
     * configured root is what makes the two agree when Forge was pointed at an
     * absolute path and the converter at a relative one.
     */
    @ParameterizedTest
    @CsvSource({
            "C:\\Users\\x\\forge\\forge-gui\\res\\cardsfolder\\p\\+2_mace.txt, p/+2_mace.txt",
            "/home/x/forge/forge-gui/res/cardsfolder/p/+2_mace.txt,           p/+2_mace.txt",
            "../forge/forge-gui/res/cardsfolder/upcoming/horta.txt,           upcoming/horta.txt",
            "cardsfolder/a/ajanis_pridemate.txt,                              a/ajanis_pridemate.txt",
            "p/+2_mace.txt,                                                   p/+2_mace.txt",
    })
    void aLoaderPathBecomesTheTreeRelativePath(String loaderPath, String expected) {
        assertEquals(expected,
                SourceTree.relativePathIn(SourceTree.CARDSFOLDER, loaderPath));
    }

    /**
     * A path from somewhere else answers nothing rather than guessing. Forge
     * also reads the user's custom-cards directory, and a card from there is
     * not a card in this tree.
     */
    @ParameterizedTest
    @CsvSource({
            "C:\\Users\\x\\AppData\\Roaming\\Forge\\Cards\\my_card.txt",
            "/home/x/.forge/Cards/deep/my_card.txt",
            "ajanis_pridemate.txt",
            "cardsfolder/a/ajanis_pridemate.json",
    })
    void aPathOutsideTheTreeAnswersNothing(String loaderPath) {
        assertNull(SourceTree.relativePathIn(SourceTree.CARDSFOLDER, loaderPath));
    }

    @Test
    void anAbsentLoaderPathAnswersNothing() {
        assertNull(SourceTree.relativePathIn(SourceTree.CARDSFOLDER, null));
        assertNull(SourceTree.relativePathIn(SourceTree.CARDSFOLDER, ""));
    }
}
