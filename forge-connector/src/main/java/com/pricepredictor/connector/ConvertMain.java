package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.MissingSVarReport;
import com.pricepredictor.connector.effects.SourceTree;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * CLI entry point for batch card script conversion.
 *
 * <p>Converts two source trees, not one. Token scripts stay out of
 * {@code output/cardsfolder/} because their converted filenames collide with
 * card ones and the sealed pipeline reads that tree as its card corpus; they go
 * to {@code output/tokenscripts/} with sidecars of their own. A token a card
 * creates is an ability the effect model has to be able to name, so the token
 * tree is part of the encoding surface rather than an extra.
 */
public class ConvertMain {

    public static void main(String[] args) {
        ToolMain.Args cli = ToolMain.args(args,
                "--cards-path", "--output-path", "--tokens-path", "--tokens-output-path");
        String cardsPath = cli.get("--cards-path", "../forge/forge-gui/res/cardsfolder/");
        String outputPath = cli.get("--output-path", "./output");
        String tokensPath = cli.get("--tokens-path", defaultSibling(cardsPath, "tokenscripts"));
        String tokensOutputPath = cli.get(
                "--tokens-output-path", defaultSibling(outputPath, "tokenscripts"));

        ToolMain.run(() -> {
            ForgeEnvironmentInitializer.initialize();

            BatchConverter batchConverter = new BatchConverter();
            BatchConverter.BatchResult cards = batchConverter.convert(
                    Path.of(cardsPath), Path.of(outputPath), SourceTree.CARDSFOLDER);

            System.out.println("Conversion complete:");
            cards.report(System.out, SourceTree.CARDSFOLDER);
            List<MissingSVarReport.Entry> missingSVars =
                    new ArrayList<>(cards.missingSVars());

            Path tokensSource = Path.of(tokensPath);
            if (Files.isDirectory(tokensSource)) {
                BatchConverter.BatchResult tokens = batchConverter.convert(
                        tokensSource, Path.of(tokensOutputPath),
                        SourceTree.TOKENSCRIPTS);
                tokens.report(System.out, SourceTree.TOKENSCRIPTS);
                System.out.println("  Token output: " + tokensOutputPath);
                missingSVars.addAll(tokens.missingSVars());
            } else {
                System.out.println("  tokenscripts: skipped, no such directory ("
                        + tokensPath + ")");
            }

            // One block, on stderr, after both trees: the Python supervisor
            // leaves the JVM's stderr attached to the operator's terminal, so
            // this is what they read — and it is not a warning, because the
            // card converted; only its chain has a hole the script wrote.
            String block = MissingSVarReport.formatBlock(missingSVars);
            if (block != null) {
                System.out.flush();
                System.err.print(block);
                System.err.flush();
            }
        });
    }

    /** A sibling of {@code path} named {@code name} — the token tree's default. */
    private static String defaultSibling(String path, String name) {
        Path resolved = Path.of(path).toAbsolutePath().normalize();
        Path parent = resolved.getParent();
        return (parent == null ? Path.of(name) : parent.resolve(name)).toString();
    }
}
