package com.pricepredictor.connector;

import forge.localinstance.properties.ForgePreferences;
import forge.localinstance.properties.ForgePreferences.FPref;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The game-rule preferences are pinned, not read from the machine's Forge profile (finding 11.2). */
@ExtendWith(ForgeExtension.class)
class ForgeEnvironmentInitializerTest {

    private static final PinnedForgePreferences PINNED =
            new PinnedForgePreferences("London", false, false, "en-US");

    @Test
    void pinningOverwritesPreferencesSetToOtherValues() {
        ForgePreferences prefs = new ForgePreferences();
        prefs.setPref(FPref.MULLIGAN_RULE, "Vancouver");
        prefs.setPref(FPref.FILTERED_HANDS, "true");
        prefs.setPref(FPref.PERFORMANCE_MODE, "true");
        prefs.setPref(FPref.UI_LANGUAGE, "de-DE");

        PinnedForgePreferences.apply(prefs);

        assertEquals("London", prefs.getPref(FPref.MULLIGAN_RULE));
        assertEquals("false", prefs.getPref(FPref.FILTERED_HANDS));
        assertEquals("false", prefs.getPref(FPref.PERFORMANCE_MODE));
        assertEquals("en-US", prefs.getPref(FPref.UI_LANGUAGE));
    }

    @Test
    void theEngineRunsWithThePinnedValuesAndTheInitializerReportsThem() {
        assertEquals(PINNED, PinnedForgePreferences.inEffect());
        assertEquals(PINNED, ForgeEnvironmentInitializer.pinnedPreferences());
    }

    @Test
    void aSecondInitializationWithOtherArgumentsThrows(@TempDir Path variants) {
        assertDoesNotThrow(() -> ForgeEnvironmentInitializer.initialize());
        assertDoesNotThrow(() -> ForgeEnvironmentInitializer.initialize(null, false));
        IllegalStateException e = assertThrows(
                IllegalStateException.class,
                () -> ForgeEnvironmentInitializer.initialize(variants));
        assertTrue(e.getMessage().contains("already initialized"), e.getMessage());
    }

    @Test
    void aForeignScriptInACustomDirectoryRefusesStartup(@TempDir Path root) throws IOException {
        Path cards = Files.createDirectories(root.resolve("cards"));
        Path tokens = Files.createDirectories(root.resolve("tokens"));
        Files.writeString(cards.resolve("mine.txt"), "Name:Mine\n");
        Files.writeString(cards.resolve("staged.txt"), "Name:Staged\n");
        Files.write(cards.resolve(ForgeEnvironmentInitializer.STAGED_MANIFEST), java.util.List.of("staged.txt"));

        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> PinnedForgePreferences.requireNoForeignScripts(
                        cards, tokens, ForgeEnvironmentInitializer.STAGED_MANIFEST));
        assertTrue(e.getMessage().contains(cards.toString()), e.getMessage());

        Files.delete(cards.resolve("mine.txt"));
        assertDoesNotThrow(() -> PinnedForgePreferences.requireNoForeignScripts(
                cards, tokens, ForgeEnvironmentInitializer.STAGED_MANIFEST));
        assertDoesNotThrow(() -> PinnedForgePreferences.requireNoForeignScripts(
                root.resolve("absent"), root.resolve("absent2"),
                ForgeEnvironmentInitializer.STAGED_MANIFEST));
    }
}
