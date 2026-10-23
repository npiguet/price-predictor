package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** The one way a provenance key's script path is assembled and taken apart. */
class ScriptFileTest {

    /** The key is compared as a string on both sides, so the separator is fixed. */
    @Test
    void theKeyIsTreePrefixedWithForwardSlashesWhateverThePlatformWrote() {
        ScriptFile script = new ScriptFile(SourceTree.CARDSFOLDER, "a\\ajanis_pridemate.txt");
        assertEquals("cardsfolder/a/ajanis_pridemate.txt", script.key());
        assertEquals("a/ajanis_pridemate.txt", script.relativePath());
    }

    @Test
    void aLetterKeyedFileSplitsIntoItsDirectoryAndStem() {
        ScriptFile script = new ScriptFile(SourceTree.CARDSFOLDER, "p/+2_mace.txt");
        assertEquals("p", script.directory());
        assertEquals("+2_mace", script.stem());
    }

    @Test
    void aFlatTreesFileHasNoDirectory() {
        ScriptFile script = new ScriptFile(SourceTree.TOKENSCRIPTS, "c_a_food_sac.txt");
        assertEquals("", script.directory());
        assertEquals("c_a_food_sac", script.stem());
        assertEquals("tokenscripts/c_a_food_sac.txt", script.key());
    }
}
