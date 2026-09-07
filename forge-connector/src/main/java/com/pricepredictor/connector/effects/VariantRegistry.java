package com.pricepredictor.connector.effects;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The synthetic variant card names staged into this JVM.
 *
 * <p>A provenance key names the converted tree its script came from, and the
 * runtime has no other way to tell a variant from a printed card: a variant is
 * an ordinary non-token {@code Card} once Forge has loaded it, so
 * {@link ProvenanceKey} would key it under {@code cardsfolder/} and the Python
 * reader would look for a sidecar that does not exist there. Registering the
 * names at staging time is what makes the key say {@code variant-scripts/}.
 *
 * <p>Populated by {@code ForgeEnvironmentInitializer} before
 * {@code FModel.initialize}, and empty in every run that stages no variants —
 * so a worker that never saw a variant behaves exactly as it did at stage one.
 */
public final class VariantRegistry {

    private VariantRegistry() {
    }

    private static final Set<String> NAMES =
            Collections.newSetFromMap(new ConcurrentHashMap<>());

    /** Register every card name declared by a {@code .txt} script under {@code source}. */
    public static void registerAll(Path source) {
        if (!Files.isDirectory(source)) {
            return;
        }
        try (var scripts = Files.walk(source)) {
            for (Path script : scripts.filter(Files::isRegularFile).toList()) {
                if (script.toString().endsWith(".txt")) {
                    register(nameOf(script));
                }
            }
        } catch (IOException e) {
            System.err.println(
                    "Could not read variant names from " + source + ": "
                            + e.getMessage() + "; variant records will be keyed "
                            + "under " + SourceTree.CARDSFOLDER);
        }
    }

    public static void register(String cardName) {
        if (cardName != null && !cardName.isEmpty()) {
            NAMES.add(cardName);
        }
    }

    public static boolean isVariant(String cardName) {
        return cardName != null && NAMES.contains(cardName);
    }

    public static int size() {
        return NAMES.size();
    }

    /** Test seam: the registry is JVM-wide static state. */
    public static void clear() {
        NAMES.clear();
    }

    /** The {@code Name:} a Forge source script declares, or null. */
    private static String nameOf(Path script) {
        try {
            List<String> lines = Files.readAllLines(script);
            for (String line : lines) {
                if (line.startsWith("Name:")) {
                    return line.substring("Name:".length()).trim();
                }
            }
        } catch (IOException | RuntimeException e) {
            // A script that cannot be read cannot be staged either, so the
            // registry simply does not learn its name.
        }
        return null;
    }
}
