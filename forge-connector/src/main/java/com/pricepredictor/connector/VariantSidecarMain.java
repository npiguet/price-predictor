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
        String variantsPath = ToolMain.args(args, "--variants-path")
                .get("--variants-path", "output/effects/variant-scripts/");

        Path variants = Path.of(variantsPath);
        if (!Files.isDirectory(variants)) {
            System.err.println("No variant scripts at " + variantsPath);
            throw ToolMain.exit(1);
        }

        ToolMain.run(() -> {
            // Staged, not bare: the loadable list below has to be taken from a
            // card database that actually holds the variants, and it is the
            // same database a worker will build.
            ForgeEnvironmentInitializer.initialize(variants);

            BatchConverter.BatchResult result = new BatchConverter().convert(
                    variants, variants, SourceTree.VARIANT_SCRIPTS, false);

            System.out.println("Variant sidecars written:");
            result.report(System.out, SourceTree.VARIANT_SCRIPTS);

            int loadable = writeLoadableNames(variants);
            System.out.println(
                    "  Loadable:     " + loadable + " of " + result.totalFiles()
                            + " (written to " + LOADABLE_FILE + ")");
        });
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
     * such a deck outright rather than playing basics, so this is not a
     * rounding error: with even a modest share of variants unloadable, almost
     * every deck of 23 nonlands holds at least one and the round collects
     * nearly nothing while every worker dies on it.
     *
     * <p>The lookup is the one {@code DeckMaterializer.resolve} uses, so
     * a name on this list is a name that will resolve there.
     *
     * @return how many of the variant scripts Forge accepted
     */
    private static int writeLoadableNames(Path variants) throws java.io.IOException {
        java.util.List<String> accepted = new java.util.ArrayList<>();
        for (Path script : ScriptFiles.under(variants)) {
            String name = ScriptHeader.of(Files.readAllLines(script)).name();
            if (name == null) {
                continue;
            }
            if (forge.model.FModel.getMagicDb().getCommonCards()
                    .getCard(name) != null) {
                accepted.add(name);
            }
        }
        java.util.Collections.sort(accepted);
        Files.write(variants.resolve(LOADABLE_FILE), accepted);
        return accepted.size();
    }
}
