package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.SourceTree;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * CLI entry point writing a provenance sidecar beside every variant script.
 *
 * <p>Invoked by {@code python -m effects collect-variants} after it has
 * generated the perturbed scripts and before it plays them. Separate from
 * {@link ConvertMain} because it writes sidecars <em>only</em>: a variant has no
 * oracle text, so converting it to prose would put text in the corpus that no
 * card has (FR-056).
 *
 * <p>The sidecar has to come from this parser rather than from the Python
 * generator. A provenance key's {@code index_within_kind} is the trait's
 * position in Forge's own runtime trait list, and the collectors read that list
 * directly, so only the parser that builds it can number the traits the way a
 * record will name them.
 */
public class VariantSidecarMain {

    public static void main(String[] args) {
        String variantsPath = "output/effects/variant-scripts/";

        for (int i = 0; i < args.length; i++) {
            if ("--variants-path".equals(args[i]) && i + 1 < args.length) {
                variantsPath = args[++i];
            }
        }

        Path variants = Path.of(variantsPath);
        if (!Files.isDirectory(variants)) {
            System.err.println("No variant scripts at " + variantsPath);
            System.exit(1);
        }

        try {
            // Staged, not bare: the loadable list below has to be taken from a
            // card database that actually holds the variants, and it is the
            // same database a worker will build.
            ForgeEnvironmentInitializer.initialize(variants);

            BatchConverter.BatchResult result = new BatchConverter().convert(
                    variants, variants, SourceTree.VARIANT_SCRIPTS, false);

            System.out.println("Variant sidecars written:");
            System.out.println("  Total files:  " + result.totalFiles());
            System.out.println("  Succeeded:    " + result.succeeded());
            System.out.println("  Warnings:     " + result.warningCount());
            for (String warning : result.warnings()) {
                System.out.println("  " + warning);
            }

            int loadable = writeLoadableNames(variants);
            System.out.println(
                    "  Loadable:     " + loadable + " of " + result.totalFiles()
                            + " (written to " + LOADABLE_FILE + ")");
            System.exit(0);
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /** Where the names Forge's card database accepted are written. */
    static final String LOADABLE_FILE = "loadable.txt";

    /**
     * Write every variant name Forge's card database actually holds.
     *
     * <p>A perturbation can produce a script Forge declines to load — a
     * sub-ability that no longer resolves, a selector the parser rejects. The
     * script is still on disk and still has a sidecar, so nothing before this
     * point can tell the difference, and a deck built over the generated names
     * then carries cards no worker can materialize. A decks-only round refuses
     * such a deck outright rather than playing basics, so on a real run this is
     * not a rounding error: at roughly one variant in nine unloadable, almost
     * every deck of 23 nonlands holds at least one and the round collects
     * nearly nothing while every worker dies on it.
     *
     * <p>The lookup is the one {@code MatchGenerator.materializeDeck} uses, so
     * a name on this list is a name that will resolve there.
     *
     * @return how many of the variant scripts Forge accepted
     */
    private static int writeLoadableNames(Path variants) throws java.io.IOException {
        java.util.List<String> accepted = new java.util.ArrayList<>();
        try (var scripts = Files.walk(variants)) {
            for (Path script : scripts.filter(Files::isRegularFile).toList()) {
                if (!script.toString().endsWith(".txt")) {
                    continue;
                }
                String name = nameLineOf(script);
                if (name == null) {
                    continue;
                }
                if (forge.model.FModel.getMagicDb().getCommonCards()
                        .getCard(name) != null) {
                    accepted.add(name);
                }
            }
        }
        java.util.Collections.sort(accepted);
        Files.write(variants.resolve(LOADABLE_FILE), accepted);
        return accepted.size();
    }

    /** The {@code Name:} a Forge card script declares, or null. */
    private static String nameLineOf(Path script) throws java.io.IOException {
        for (String line : Files.readAllLines(script)) {
            if (line.startsWith("Name:")) {
                return line.substring("Name:".length()).trim();
            }
        }
        return null;
    }
}
