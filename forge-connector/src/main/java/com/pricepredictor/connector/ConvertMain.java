package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.SourceTree;

import java.nio.file.Files;
import java.nio.file.Path;

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
        String cardsPath = "../forge/forge-gui/res/cardsfolder/";
        String outputPath = "./output";
        String tokensPath = null;
        String tokensOutputPath = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--cards-path" -> {
                    if (i + 1 < args.length) cardsPath = args[++i];
                }
                case "--output-path" -> {
                    if (i + 1 < args.length) outputPath = args[++i];
                }
                case "--tokens-path" -> {
                    if (i + 1 < args.length) tokensPath = args[++i];
                }
                case "--tokens-output-path" -> {
                    if (i + 1 < args.length) tokensOutputPath = args[++i];
                }
            }
        }
        if (tokensPath == null) {
            tokensPath = defaultSibling(cardsPath, "tokenscripts");
        }
        if (tokensOutputPath == null) {
            tokensOutputPath = defaultSibling(outputPath, "tokenscripts");
        }

        try {
            ForgeEnvironmentInitializer.initialize();

            BatchConverter batchConverter = new BatchConverter();
            BatchConverter.BatchResult cards = batchConverter.convert(
                    Path.of(cardsPath), Path.of(outputPath), SourceTree.CARDSFOLDER);

            System.out.println("Conversion complete:");
            report("cardsfolder", cards);

            Path tokensSource = Path.of(tokensPath);
            if (Files.isDirectory(tokensSource)) {
                BatchConverter.BatchResult tokens = batchConverter.convert(
                        tokensSource, Path.of(tokensOutputPath),
                        SourceTree.TOKENSCRIPTS);
                report("tokenscripts", tokens);
                System.out.println("  Token output: " + tokensOutputPath);
            } else {
                System.out.println("  tokenscripts: skipped, no such directory ("
                        + tokensPath + ")");
            }

            System.exit(0);
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void report(String tree, BatchConverter.BatchResult result) {
        System.out.println("  [" + tree + "]");
        System.out.println("  Total files:  " + result.totalFiles());
        System.out.println("  Succeeded:    " + result.succeeded());
        System.out.println("  Warnings:     " + result.warningCount());
        if (!result.warnings().isEmpty()) {
            System.out.println("\nWarnings:");
            for (String warning : result.warnings()) {
                System.out.println("  " + warning);
            }
        }
    }

    /** A sibling of {@code path} named {@code name} — the token tree's default. */
    private static String defaultSibling(String path, String name) {
        Path resolved = Path.of(path).toAbsolutePath().normalize();
        Path parent = resolved.getParent();
        return (parent == null ? Path.of(name) : parent.resolve(name)).toString();
    }
}
