package com.pricepredictor.connector;

import java.nio.file.Path;

/**
 * CLI entry point for batch card script conversion.
 */
public class ConvertMain {

    public static void main(String[] args) {
        CliArgs cli = CliArgs.parse(args);
        String cardsPath = cli.get("--cards-path", "../forge/forge-gui/res/cardsfolder/");
        String outputPath = cli.get("--output-path", "./output");

        try {
            ForgeEnvironmentInitializer.initialize();

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
