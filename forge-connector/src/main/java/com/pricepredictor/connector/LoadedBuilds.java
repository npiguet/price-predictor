package com.pricepredictor.connector;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.CodeSource;
import java.util.jar.Attributes;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

/**
 * The builds this JVM runs, as their jars' manifests name them (finding
 * 11.1): the commit the Forge engine it loaded was built at, read from the
 * jar {@code forge.game.Game} came from (Forge's pom writes
 * {@value #BUILD_COMMIT} and {@value #BUILD_DIRTY} into every jar it builds),
 * and the Forge commit the connector JAR was built against
 * ({@value #CONNECTOR_COMMIT}, {@code forge-connector/pom.xml}).
 *
 * <p>Each is null when its manifest names none: a class loaded from a
 * directory (the connector's own classes under test), a Forge built before
 * its pom wrote the entry, a jar that cannot be read. The supervisor refuses
 * a collection on either missing or differing from the run's conversion.
 *
 * @param forgeBuildCommit the loaded Forge engine's build commit, or null
 * @param forgeBuildDirty whether its checkout had uncommitted changes to
 *        tracked files when it was built, or null
 * @param connectorForgeCommit the connector JAR's {@code Forge-Commit}, or null
 */
record LoadedBuilds(String forgeBuildCommit, Boolean forgeBuildDirty, String connectorForgeCommit) {

    /** The Forge jars' manifest attribute: the commit the jar was built at. */
    static final String BUILD_COMMIT = "Forge-Build-Commit";
    /** The Forge jars' manifest attribute: {@code true} when built from a dirty checkout. */
    static final String BUILD_DIRTY = "Forge-Build-Dirty";
    /** The connector JAR's manifest attribute: the Forge commit it was built against. */
    static final String CONNECTOR_COMMIT = "Forge-Commit";

    /** The builds of this JVM: the engine {@code forge.game.Game} was loaded from, and this connector. */
    static LoadedBuilds ofThisJvm() {
        Attributes engine = mainAttributes(forge.game.Game.class);
        return new LoadedBuilds(engine.getValue(BUILD_COMMIT),
                flag(engine.getValue(BUILD_DIRTY)),
                mainAttributes(LoadedBuilds.class).getValue(CONNECTOR_COMMIT));
    }

    /** {@code true} or {@code false} as the manifest spells it; null for anything else. */
    private static Boolean flag(String value) {
        return "true".equals(value) || "false".equals(value) ? Boolean.valueOf(value) : null;
    }

    /**
     * The main attributes of the manifest of the jar {@code type} was loaded
     * from; empty when it came from no jar or the jar cannot be read.
     */
    static Attributes mainAttributes(Class<?> type) {
        CodeSource source = type.getProtectionDomain().getCodeSource();
        if (source == null || source.getLocation() == null) {
            return new Attributes();
        }
        try {
            return mainAttributes(Path.of(source.getLocation().toURI()));
        } catch (URISyntaxException | IllegalArgumentException e) {
            return new Attributes();
        }
    }

    private static Attributes mainAttributes(Path location) {
        if (!Files.isRegularFile(location)) {
            return new Attributes();
        }
        try (JarFile jar = new JarFile(location.toFile())) {
            Manifest manifest = jar.getManifest();
            return manifest == null ? new Attributes() : manifest.getMainAttributes();
        } catch (IOException e) {
            return new Attributes();
        }
    }
}
