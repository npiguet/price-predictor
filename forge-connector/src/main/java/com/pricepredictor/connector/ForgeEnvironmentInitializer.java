package com.pricepredictor.connector;

import forge.util.Lang;
import forge.util.Localizer;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Shared Forge initialization — used by both ConvertMain and tests.
 * Auto-detects the Forge installation by searching for a "forge" directory
 * starting from the current working directory and walking up to parent directories.
 */
public class ForgeEnvironmentInitializer {
    private static volatile boolean initialized = false;

    private static final String LANG_SUBPATH = "forge-gui/res/languages";

    public static synchronized void initialize() {
        if (initialized) return;
        Lang.createInstance("en-US");
        Path langDir = findForgeLangDir();
        Localizer.getInstance().initialize("en-US", langDir.toString() + "/");
        initialized = true;
    }

    private static Path findForgeLangDir() {
        Path dir = Path.of("").toAbsolutePath();
        while (dir != null) {
            Path candidate = dir.resolve("forge").resolve(LANG_SUBPATH);
            if (Files.isDirectory(candidate)) {
                return candidate;
            }
            dir = dir.getParent();
        }
        throw new IllegalStateException(
                "Could not find forge/" + LANG_SUBPATH + " in any parent of " + Path.of("").toAbsolutePath());
    }
}
