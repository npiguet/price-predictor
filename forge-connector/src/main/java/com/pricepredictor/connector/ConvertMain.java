package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.MissingSVarReport;
import com.pricepredictor.connector.effects.SourceTree;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * CLI entry point for batch card script conversion.
 *
 * <p>Converts two source trees, not one. Token scripts stay out of the
 * converted card tree ({@code --output-path}) because their converted
 * filenames collide with card ones and the sealed pipeline reads that tree as
 * its card corpus; they go to their own tree ({@code --tokens-output-path})
 * with sidecars of their own. A token a card creates is an ability the effect
 * model has to be able to name, so the token tree is part of the encoding
 * surface rather than an extra.
 *
 * <p>An output tree that already holds converted files is refused unless
 * {@code --replace} is given, which sweeps them out first
 * ({@link ConvertedTrees}). The run exits {@link ExitCode#FAILURE} when any
 * script failed to convert, after the counts and the first names.
 */
public class ConvertMain {

    /** How many failed scripts the closing line names. */
    private static final int NAMED_FAILURES = 10;

    public static void main(String[] args) {
        ToolMain.run(() -> work(args));
    }

    /** One source tree and where it is converted to. */
    private record Conversion(Path source, Path output, SourceTree tree) {
    }

    /**
     * The conversion itself; an exit it requests ends the process through
     * {@link ToolMain#run}.
     */
    static void work(String[] args) throws IOException {
        ToolMain.Args cli = ToolMain.args(args, Set.of("--replace"),
                "--cards-path", "--output-path", "--tokens-path", "--tokens-output-path");
        List<Conversion> conversions = conversions(cli);
        List<Path> outputs = conversions.stream().map(Conversion::output).toList();
        boolean replace = cli.has("--replace");

        ConvertedTrees.refuseUnsafe(
                outputs, conversions.stream().map(Conversion::source).toList());
        if (!replace) {
            ConvertedTrees.refuseOccupied(outputs);
        }

        ForgeEnvironmentInitializer.initialize();
        if (replace) {
            for (Path output : outputs) {
                System.out.println("Swept " + ConvertedTrees.sweep(output)
                        + " converted files from " + output);
            }
        }

        List<BatchConverter.BatchResult> results = convertAll(conversions);
        reportMissingSVars(results);
        refuseFailures(results);
    }

    /**
     * The card tree, then the token tree when its source exists. A missing
     * token source is skipped, its output tree neither checked nor touched.
     */
    private static List<Conversion> conversions(ToolMain.Args cli) {
        String cardsPath = cli.get("--cards-path", "../forge/forge-gui/res/cardsfolder/");
        // No default output tree: the run directory belongs to the caller
        // (the artifact layout), and a default would take the conversion
        // wherever a misspelt flag left it.
        String outputPath = cli.require("--output-path");
        String tokensOutputPath = cli.require("--tokens-output-path");
        String tokensPath = cli.get("--tokens-path", defaultSibling(cardsPath, "tokenscripts"));

        List<Conversion> conversions = new ArrayList<>(2);
        conversions.add(new Conversion(Path.of(cardsPath), Path.of(outputPath), SourceTree.CARDSFOLDER));
        Path tokensSource = Path.of(tokensPath);
        if (Files.isDirectory(tokensSource)) {
            conversions.add(new Conversion(
                    tokensSource, Path.of(tokensOutputPath), SourceTree.TOKENSCRIPTS));
        } else {
            System.out.println("tokenscripts: skipped, no such directory (" + tokensPath + ")");
        }
        return conversions;
    }

    private static List<BatchConverter.BatchResult> convertAll(List<Conversion> conversions)
            throws IOException {
        BatchConverter batchConverter = new BatchConverter();
        List<BatchConverter.BatchResult> results = new ArrayList<>(conversions.size());
        for (Conversion conversion : conversions) {
            results.add(batchConverter.convert(
                    conversion.source(), conversion.output(), conversion.tree()));
        }
        System.out.println("Conversion complete:");
        for (int i = 0; i < results.size(); i++) {
            results.get(i).report(System.out, conversions.get(i).tree());
            System.out.println("  Output: " + conversions.get(i).output());
        }
        return results;
    }

    /**
     * One block, on stderr, after both trees: the Python supervisor leaves
     * the JVM's stderr attached to the operator's terminal, so this is what
     * they read — and it is not a warning, because the card converted; only
     * its chain has a hole the script wrote.
     */
    private static void reportMissingSVars(List<BatchConverter.BatchResult> results) {
        List<MissingSVarReport.Entry> missingSVars = new ArrayList<>();
        for (BatchConverter.BatchResult result : results) {
            missingSVars.addAll(result.missingSVars());
        }
        String block = MissingSVarReport.formatBlock(missingSVars);
        if (block != null) {
            System.out.flush();
            System.err.print(block);
            System.err.flush();
        }
    }

    /**
     * End the run with {@link ExitCode#FAILURE} when any script failed,
     * naming how many and the first of them: a tree missing cards is not a
     * conversion any reader should take for whole.
     */
    private static void refuseFailures(List<BatchConverter.BatchResult> results) {
        int total = 0;
        List<String> failed = new ArrayList<>();
        for (BatchConverter.BatchResult result : results) {
            total += result.totalFiles();
            failed.addAll(result.failedScripts());
        }
        if (failed.isEmpty()) {
            return;
        }
        List<String> named = failed.subList(0, Math.min(NAMED_FAILURES, failed.size()));
        System.out.flush();
        System.err.println("Error: " + failed.size() + " of " + total
                + " scripts failed to convert; the first " + named.size() + ": "
                + String.join(", ", named));
        throw ToolMain.exit(ExitCode.FAILURE);
    }

    /** A sibling of {@code path} named {@code name} — the token source's default. */
    private static String defaultSibling(String path, String name) {
        Path resolved = Path.of(path).toAbsolutePath().normalize();
        Path parent = resolved.getParent();
        return (parent == null ? Path.of(name) : parent.resolve(name)).toString();
    }
}
