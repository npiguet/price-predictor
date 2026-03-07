package com.pricepredictor.connector;

import java.nio.file.Path;

/**
 * CLI entry point for batch card script conversion.
 */
public class ConvertMain {

    public static void main(String[] args) {
        String cardsPath = "../forge/forge-gui/res/cardsfolder/";
        String outputPath = "./output";
        String forgeLangDir = "../forge/forge-gui/res/languages/";

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--cards-path" -> {
                    if (i + 1 < args.length) cardsPath = args[++i];
                }
                case "--output-path" -> {
                    if (i + 1 < args.length) outputPath = args[++i];
                }
            }
        }

        try {
            ForgeEnvironmentInitializer.initialize(forgeLangDir);

            BatchConverter batchConverter = new BatchConverter();
            BatchConverter.BatchResult result = batchConverter.convert(
                    Path.of(cardsPath), Path.of(outputPath));

            System.out.println("Conversion complete:");
            System.out.println("  Total files:  " + result.totalFiles());
            System.out.println("  Succeeded:    " + result.succeeded());
            System.out.println("  Warnings:     " + result.warningCount());

            if (!result.warnings().isEmpty()) {
                System.out.println("\nWarnings:");
                for (String warning : result.warnings()) {
                    System.out.println("  " + warning);
                }
            }

            System.exit(0);
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
