package com.pricepredictor.connector;

import forge.gui.GuiBase;
import forge.model.FModel;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Shared Forge initialization — used by all entry points (ConvertMain, PoolMain, MatchWorkerMain) and tests.
 *
 * <p>Uses FModel.initialize() + GuiHeadless to set up the full Forge environment including
 * StaticData, AI subsystems, and the game rules engine. Auto-detects the Forge installation
 * by searching for a "forge" directory starting from the current working directory and walking
 * up to parent directories.
 */
public class ForgeEnvironmentInitializer {
    private static volatile boolean initialized = false;

    private static final String RES_SUBPATH = "forge-gui/res";
    private static final String GUI_SUBPATH = "forge-gui";

    public static synchronized void initialize() {
        initialize(null);
    }

    /**
     * Initialize Forge, optionally loading an extra directory of card scripts.
     *
     * <p>Stage four's synthetic variants live in
     * {@code output/effects/variant-scripts/}, and Forge cannot see them
     * otherwise: it resolves only {@code forge-gui/res/cardsfolder} plus the
     * fixed {@code ForgeConstants.USER_CUSTOM_CARDS_DIR}, neither of which this
     * project writes to. Without this, {@code collect-variants} could generate
     * perturbed scripts and never get one into a game.
     *
     * <p>The extra directory is staged into Forge's own custom-cards directory
     * rather than passed to the card reader, because the reader's source path
     * is a constant the engine resolves at class-init time. Staging is a copy,
     * so the variant tree stays the artifact the trainer and the encoder read.
     *
     * @param extraCardSource an additional card-script directory, or null
     */
    public static synchronized void initialize(Path extraCardSource) {
        if (initialized) return;

        Path forgeDir = findForgeDir();
        Path forgeGuiDir = forgeDir.resolve(GUI_SUBPATH);

        if (extraCardSource != null) {
            stageCustomCards(extraCardSource);
        }

        GuiBase.setInterface(new GuiHeadless(forgeGuiDir + File.separator));
        FModel.initialize(null, null);

        initialized = true;
    }

    /**
     * Copy an extra card-script directory into Forge's custom-cards directory.
     *
     * <p>Must run <b>before</b> {@code FModel.initialize}: the card database is
     * read once at startup, and a script that appears afterwards is invisible
     * for the life of the JVM.
     */
    private static void stageCustomCards(Path source) {
        if (!Files.isDirectory(source)) {
            return;
        }
        Path target = Path.of(forge.localinstance.properties.ForgeConstants
                .USER_CUSTOM_CARDS_DIR);
        try {
            Files.createDirectories(target);
            try (var scripts = Files.walk(source)) {
                for (Path script : scripts.filter(Files::isRegularFile).toList()) {
                    if (!script.toString().endsWith(".txt")) {
                        continue;
                    }
                    Files.copy(
                            script, target.resolve(script.getFileName()),
                            java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                }
            }
            System.out.println(
                    "Staged custom card scripts from " + source + " into " + target);
        } catch (java.io.IOException e) {
            System.err.println(
                    "Could not stage custom cards from " + source + ": "
                            + e.getMessage() + "; variant cards will not load");
        }
    }

    public static Path findCardsFolder() {
        return findForgeDir().resolve(RES_SUBPATH).resolve("cardsfolder");
    }

    private static Path findForgeDir() {
        Path dir = Path.of("").toAbsolutePath();
        while (dir != null) {
            Path candidate = dir.resolve("forge").resolve(RES_SUBPATH);
            if (Files.isDirectory(candidate)) {
                return dir.resolve("forge");
            }
            dir = dir.getParent();
        }
        throw new IllegalStateException(
                "Could not find forge/" + RES_SUBPATH + " in any parent of " + Path.of("").toAbsolutePath());
    }
}
