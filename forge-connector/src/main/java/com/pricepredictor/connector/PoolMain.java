package com.pricepredictor.connector;

import forge.util.MyRandom;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.StringJoiner;

/**
 * CLI entry point for sealed pool generation.
 *
 * <p>Usage:
 * <pre>
 *   java -cp &lt;classpath&gt; com.pricepredictor.connector.PoolMain
 *       [--set &lt;SET_CODE&gt;]
 *       --size &lt;N&gt;
 *       --pools-path &lt;PATH&gt;
 * </pre>
 *
 * <p>When {@code --set} is omitted, each pool is generated from a randomly
 * selected sealed-legal set (per-pool randomness).
 *
 * <p>Writes {@code pools.txt} to the specified path — one pool per line,
 * formatted as {@code SET_CODE;Card1|Card2|...|CardN}. Streams a progress
 * line to stdout every 1000 pools. Exits 0 on success, 1 on error.
 */
public class PoolMain {

    public static void main(String[] args) {
        String setCode = null;
        int poolCount = 10000;
        String poolsPath = "./output/sealed/pools/";

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--set" -> {
                    if (i + 1 < args.length) setCode = args[++i];
                }
                case "--size" -> {
                    if (i + 1 < args.length) {
                        try {
                            poolCount = Integer.parseInt(args[++i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Error: --size must be an integer");
                            System.exit(1);
                        }
                    }
                }
                case "--pools-path" -> {
                    if (i + 1 < args.length) poolsPath = args[++i];
                }
            }
        }

        try {
            ForgeEnvironmentInitializer.initialize();

            PoolGenerator generator = new PoolGenerator();
            Path outputDir = Path.of(poolsPath);
            Files.createDirectories(outputDir);
            Path outputFile = outputDir.resolve("pools.txt");

            List<String> eligibleSets = null;
            Random random = null;
            if (setCode == null) {
                eligibleSets = MatchGenerator.computeEligibleSets();
                if (eligibleSets.isEmpty()) {
                    System.err.println("Error: no eligible sealed-legal sets found");
                    System.exit(1);
                }
                random = MyRandom.getRandom();
                System.out.println("Generating " + poolCount
                        + " sealed pools from random sets ("
                        + eligibleSets.size() + " eligible)...");
            } else {
                System.out.println("Generating " + poolCount + " " + setCode + " sealed pools...");
            }

            try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
                int written = 0;
                while (written < poolCount) {
                    int batch;
                    String poolSetCode;
                    if (setCode != null) {
                        batch = Math.min(1000, poolCount - written);
                        poolSetCode = setCode;
                    } else {
                        // Random set per pool — generate one at a time.
                        batch = 1;
                        poolSetCode = eligibleSets.get(random.nextInt(eligibleSets.size()));
                    }

                    List<List<String>> pools = generator.generate(poolSetCode, batch);
                    for (List<String> pool : pools) {
                        StringJoiner joiner = new StringJoiner("|");
                        for (String name : pool) {
                            joiner.add(name);
                        }
                        writer.write(poolSetCode);
                        writer.write(";");
                        writer.write(joiner.toString());
                        writer.newLine();
                        written++;
                    }

                    if (written % 1000 == 0) {
                        System.out.println("Generated " + written + "/" + poolCount + " pools");
                        System.out.flush();
                    }
                }
            }

            System.out.println("Done");
            System.out.println("Done: " + poolCount + " pools written to " + outputFile);
            System.exit(0);

        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            System.err.println("Error writing pools file: " + e.getMessage());
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }
}
