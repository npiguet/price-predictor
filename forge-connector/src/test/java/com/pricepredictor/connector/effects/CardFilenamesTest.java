package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.stream.Stream;

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
                CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "Ajani's Pridemate").key());
    }

    @Test
    void aTokenPathIsFlat() {
        assertEquals("tokenscripts/ajanis_pridemate.txt",
                CardFilenames.scriptFile(SourceTree.TOKENSCRIPTS, "Ajani's Pridemate").key());
    }

    @Test
    void aVariantPathIsFlatToo() {
        assertEquals("variant-scripts/serra_angel.txt",
                CardFilenames.scriptFile(SourceTree.VARIANT_SCRIPTS, "Serra Angel").key());
    }

    @Test
    void rebalancedCardsKeepTheirLiteralPrefixAndOwnDirectory() {
        // The general sanitizer would turn the hyphen into an underscore.
        assertEquals("cardsfolder/rebalanced/a-akki_ronin.txt",
                CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "A-Akki Ronin").key());
    }

    // -- the tree decides where a card lives ----------------------------
    //
    // Asked of Forge's own cardsfolder, the tree the collector reads, so each
    // case names a file that tree actually holds.

    /**
     * The case this lookup exists for. {@code +2 Mace} derives to
     * {@code cardsfolder/+/} and Forge files it under {@code p/}, and because
     * only {@code keyword} keys are checked against their sidecar the wrong
     * path joins to nothing and says nothing about it.
     */
    @Test
    void aCardTheTreeFilesElsewhereGetsThePathTheTreeUses() {
        assertEquals("cardsfolder/p/+2_mace.txt",
                CardFilenames.scriptFileForStem(SourceTree.CARDSFOLDER, "+2_mace").key());
    }

    /** A digit initial is a directory that exists but holds a different card. */
    @Test
    void aCardFiledUnderALetterItDoesNotStartWithFollowsTheTree() {
        assertEquals("cardsfolder/n/1996_world_champion.txt",
                CardFilenames.scriptFileForStem(
                        SourceTree.CARDSFOLDER, "1996_world_champion").key());
    }

    /**
     * Unreleased cards sit in one directory whatever their initial.
     *
     * <p>Whichever card the tree holds there, because which cards are
     * unreleased changes with every Forge upgrade.
     */
    @Test
    void anUpcomingCardKeepsItsOwnDirectory() throws IOException {
        Path upcoming = ForgeEnvironmentInitializer.findCardsFolder().resolve("upcoming");
        Assumptions.assumeTrue(Files.isDirectory(upcoming), "this Forge has no upcoming cards");
        String stem;
        try (Stream<Path> files = Files.list(upcoming)) {
            stem = files.map(path -> path.getFileName().toString())
                    .filter(name -> name.endsWith(".txt"))
                    .map(name -> name.substring(0, name.length() - ".txt".length()))
                    .sorted()
                    .findFirst().orElse(null);
        }
        Assumptions.assumeTrue(stem != null, "this Forge has no upcoming cards");

        assertEquals("cardsfolder/upcoming/" + stem + ".txt",
                CardFilenames.scriptFileForStem(SourceTree.CARDSFOLDER, stem).key());
    }

    /** The cards the convention does describe are untouched by the lookup. */
    @Test
    void aCardFiledUnderItsInitialIsUnchanged() {
        assertEquals("cardsfolder/a/ajanis_pridemate.txt",
                CardFilenames.scriptFile(SourceTree.CARDSFOLDER, "Ajani's Pridemate").key());
        assertEquals(Optional.empty(),
                SourceTree.directoryOf(SourceTree.CARDSFOLDER, "ajanis_pridemate"),
                "the tree only answers where it disagrees with the convention");
    }

    /**
     * A stem the tree does not hold still has to name something: the collector
     * sees cards Forge built without a script, and derivation is the only
     * answer left for them.
     */
    @Test
    void aStemTheTreeDoesNotHoldFallsBackToTheConvention() {
        assertEquals("cardsfolder/n/no_such_card_was_ever_printed.txt",
                CardFilenames.scriptFile(
                        SourceTree.CARDSFOLDER, "No Such Card Was Ever Printed").key());
    }

    /**
     * A tree with nowhere to look must not take the collector down with it:
     * only {@code cardsfolder} has a root to index, so any other name answers
     * nothing and the caller keeps the convention.
     */
    @Test
    void aTreeWithNoRootAnswersNothing() {
        assertEquals(Optional.empty(), SourceTree.directoryOf("cardsfolder-512", "+2_mace"));
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
        assertEquals(Optional.of(expected),
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
        assertEquals(Optional.empty(),
                SourceTree.relativePathIn(SourceTree.CARDSFOLDER, loaderPath));
    }

    @Test
    void anAbsentLoaderPathAnswersNothing() {
        assertEquals(Optional.empty(), SourceTree.relativePathIn(SourceTree.CARDSFOLDER, null));
        assertEquals(Optional.empty(), SourceTree.relativePathIn(SourceTree.CARDSFOLDER, ""));
    }
}
