package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A variant tree stays stageable after its sidecar pass (FR-058b, python R1).
 *
 * <p>The sidecar pass stages the tree into Forge's custom-cards directory,
 * initializes Forge, and writes the loadable list into the tree. The next run
 * over the same tree stages the tree again, list included; a list Forge reads
 * as a card script (any {@code .txt}) has no {@code Name:} line, and
 * {@code StaticData} throws on it while building the card database, so the
 * second startup, and every Forge startup on the machine after it, dies.
 *
 * <p>Two JVMs, because {@code ForgeEnvironmentInitializer.initialize} runs once
 * per JVM: what fails is the next process's startup. Each runs
 * {@link VariantSidecarMain}, the one process that stages. The staging lands in
 * Forge's shared custom-cards directory, so the test takes it back out by the
 * staging's own manifest whatever happens.
 */
class VariantStagingTest {

    private static final long STARTUP_MINUTES = 5;

    @Test
    void aTreeWithItsLoadableListInitializesForgeTwiceRunning(@TempDir Path tmp)
            throws Exception {
        Path tree = Files.createDirectories(tmp.resolve("variants"));
        Files.writeString(tree.resolve("r1_staging_bolt_variant_0.txt"),
                "Name:R1 Staging Bolt Variant 0\nManaCost:R\nTypes:Instant\n"
                        + "A:SP$ DealDamage | Cost$ R | ValidTgts$ Any | NumDmg$ 4 | "
                        + "SpellDescription$ CARDNAME deals 4 damage to any target.\n"
                        + "Oracle:R1 Staging Bolt Variant 0 deals 4 damage to any target.\n");
        // A stray text file in the tree, as gen-2's loadable.txt was.
        Files.writeString(tree.resolve("notes.txt"), "R1 Staging Bolt Variant 0\n");

        try {
            int first = sidecarPass(tree, tmp.resolve("first.log"));
            assertEquals(0, first, "first startup: " + tail(tmp.resolve("first.log")));
            Path list = tree.resolve(VariantSidecarMain.LOADABLE_FILE);
            assertTrue(Files.isRegularFile(list), "no loadable list written");
            assertFalse(VariantSidecarMain.LOADABLE_FILE.endsWith(".txt"),
                    "Forge reads every staged .txt as a card script");
            assertEquals(List.of("R1 Staging Bolt Variant 0"), Files.readAllLines(list));

            int second = sidecarPass(tree, tmp.resolve("second.log"));
            assertEquals(0, second, "second startup: " + tail(tmp.resolve("second.log")));
        } finally {
            unstage();
        }
    }

    /** Run the sidecar pass over {@code tree} in a JVM of its own; its exit status. */
    private static int sidecarPass(Path tree, Path log) throws IOException, InterruptedException {
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        Process process = new ProcessBuilder(
                java, "-cp", System.getProperty("java.class.path"),
                VariantSidecarMain.class.getName(), "--variants-path", tree.toString())
                .directory(new File(System.getProperty("user.dir")))
                .redirectErrorStream(true)
                .redirectOutput(log.toFile())
                .start();
        if (!process.waitFor(STARTUP_MINUTES, TimeUnit.MINUTES)) {
            process.destroyForcibly().waitFor();
            return -1;
        }
        return process.exitValue();
    }

    /** Remove what the staging copied into Forge's custom-cards directory, by its manifest. */
    private static void unstage() throws IOException {
        Path custom = ForgeEnvironmentInitializer.findCardsFolder()
                .getParent().getParent().resolve("data").resolve("custom").resolve("cards");
        Path manifest = custom.resolve(ForgeEnvironmentInitializer.STAGED_MANIFEST);
        if (!Files.isRegularFile(manifest)) {
            return;
        }
        for (String name : Files.readAllLines(manifest)) {
            if (!name.isBlank()) {
                Files.deleteIfExists(custom.resolve(name));
            }
        }
        Files.write(manifest, List.of());
    }

    private static String tail(Path log) throws IOException {
        if (!Files.isRegularFile(log)) {
            return "(no log)";
        }
        String text = new String(Files.readAllBytes(log), StandardCharsets.UTF_8);
        return text.substring(Math.max(0, text.length() - 3000));
    }
}
