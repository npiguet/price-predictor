package com.pricepredictor.connector;

import forge.game.spellability.Spell;
import forge.localinstance.properties.ForgePreferences;
import forge.localinstance.properties.ForgePreferences.FPref;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * The Forge user preferences that change what a game or a conversion
 * produces, pinned to fixed values instead of read from the machine's Forge
 * profile.
 *
 * <p>The values are applied through {@code FModel.initialize}'s
 * preference-adjusting hook, which runs right after the preferences load and
 * before anything reads them: the language reaches the {@code Localizer} and
 * the card translation, and the mulligan rule, filtered hands and performance
 * mode reach the card database and {@code Spell} from the pinned object. The
 * profile file is never written.
 *
 * @param mulliganRule    the mulligan rule's name
 * @param filteredHands   whether opening hands are smoothed
 * @param performanceMode whether {@code Spell} skips some replacement checks
 * @param uiLanguage      the language of engine-generated text
 */
public record PinnedForgePreferences(
        String mulliganRule, boolean filteredHands, boolean performanceMode,
        String uiLanguage) {

    /** The values every worker runs with. */
    public static final PinnedForgePreferences PINNED =
            new PinnedForgePreferences("London", false, false, "en-US");

    /** Overwrite the four preferences in memory; the file is not saved. */
    public static Void apply(ForgePreferences preferences) {
        preferences.setPref(FPref.MULLIGAN_RULE, PINNED.mulliganRule);
        preferences.setPref(FPref.FILTERED_HANDS, String.valueOf(PINNED.filteredHands));
        preferences.setPref(FPref.PERFORMANCE_MODE, String.valueOf(PINNED.performanceMode));
        preferences.setPref(FPref.UI_LANGUAGE, PINNED.uiLanguage);
        return null;
    }

    /** The values the engine holds now, as read back after initialization. */
    public static PinnedForgePreferences inEffect() {
        forge.StaticData data = forge.StaticData.instance();
        return new PinnedForgePreferences(
                data.getMulliganRule().name(), data.getFilteredHandsEnabled(),
                Spell.isPerformanceMode(),
                forge.model.FModel.getPreferences().getPref(FPref.UI_LANGUAGE));
    }

    /**
     * Refuse to start while a custom card or token directory holds a script
     * this project did not stage: Forge reads every script there into the card
     * database, so one the user put there changes the cards of every run.
     * Scripts named by {@code stagedManifest} (the variant staging) are ours.
     */
    static void requireNoForeignScripts(Path cardsDir, Path tokensDir, String stagedManifest) {
        for (Path dir : List.of(cardsDir, tokensDir)) {
            List<Path> foreign = foreignScripts(dir, stagedManifest);
            if (!foreign.isEmpty()) {
                throw new IllegalStateException(
                        "Forge's custom directory " + dir + " holds " + foreign.size()
                                + " script(s) not staged by this project (first: "
                                + foreign.get(0).getFileName() + "); they would join the card "
                                + "database of every run. Move them out of the directory.");
            }
        }
    }

    private static List<Path> foreignScripts(Path dir, String stagedManifest) {
        if (!Files.isDirectory(dir)) {
            return List.of();
        }
        try {
            Set<String> staged = new HashSet<>();
            Path manifest = dir.resolve(stagedManifest);
            if (Files.isRegularFile(manifest)) {
                staged.addAll(Files.readAllLines(manifest));
            }
            return ScriptFiles.textFilesUnder(dir).stream()
                    .filter(p -> !staged.contains(p.getFileName().toString()))
                    .toList();
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }
}
