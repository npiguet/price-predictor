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
        if (initialized) return;

        Path forgeDir = findForgeDir();
        Path forgeGuiDir = forgeDir.resolve(GUI_SUBPATH);

        GuiBase.setInterface(new GuiHeadless(forgeGuiDir + File.separator));
        FModel.initialize(null, null);

        initialized = true;
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
