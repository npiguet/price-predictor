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
            ForgeEnvironmentInitializer.initialize();

            BatchConverter.BatchResult result = new BatchConverter().convert(
                    variants, variants, SourceTree.VARIANT_SCRIPTS, false);

            System.out.println("Variant sidecars written:");
            System.out.println("  Total files:  " + result.totalFiles());
            System.out.println("  Succeeded:    " + result.succeeded());
            System.out.println("  Warnings:     " + result.warningCount());
            for (String warning : result.warnings()) {
                System.out.println("  " + warning);
            }
            System.exit(0);
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
