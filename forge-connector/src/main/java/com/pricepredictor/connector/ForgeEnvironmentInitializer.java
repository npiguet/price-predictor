package com.pricepredictor.connector;

import forge.CardStorageReader;
import forge.ImageKeys;
import forge.StaticData;
import forge.util.Lang;
import forge.util.Localizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;

/**
 * Shared Forge initialization — used by both ConvertMain and tests.
 * Auto-detects the Forge installation by searching for a "forge" directory
 * starting from the current working directory and walking up to parent directories.
 */
public class ForgeEnvironmentInitializer {
    private static volatile boolean initialized = false;

    private static final String RES_SUBPATH = "forge-gui/res";

    public static synchronized void initialize() {
        if (initialized) return;
        Lang.createInstance("en-US");
        Path resDir = findForgeResDir();
        Localizer.getInstance().initialize("en-US", resDir.resolve("languages").toString() + "/");
        ImageKeys.initializeDirs("", new HashMap<>(), "", "", "", "", "", "", "");
        initStaticData(resDir);
        initialized = true;
    }

    private static void initStaticData(Path resDir) {
        String cardsDir = resDir.resolve("cardsfolder").toString();
        String editionsDir = resDir.resolve("editions").toString();
        String blockDataDir = resDir.resolve("blockdata").toString();

        CardStorageReader cardReader = new CardStorageReader(cardsDir, null, false);
        new StaticData(cardReader, null, editionsDir, editionsDir, blockDataDir, "", true, true);
    }

    private static Path findForgeResDir() {
        Path dir = Path.of("").toAbsolutePath();
        while (dir != null) {
            Path candidate = dir.resolve("forge").resolve(RES_SUBPATH);
            if (Files.isDirectory(candidate)) {
                return candidate;
            }
            dir = dir.getParent();
        }
        throw new IllegalStateException(
                "Could not find forge/" + RES_SUBPATH + " in any parent of " + Path.of("").toAbsolutePath());
    }
}
