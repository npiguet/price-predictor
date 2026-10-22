package com.pricepredictor.connector;

import forge.util.MyRandom;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/**
 * CLI entry point for sealed pool generation.
 *
 * <p>Usage:
 * <pre>
 *   java -cp &lt;classpath&gt; com.pricepredictor.connector.PoolMain
 *       [--set &lt;SET_CODE&gt;]
 *       --size &lt;N&gt;
 *       --pools-path &lt;PATH&gt;
 *       [--exclude-cards &lt;FILE&gt;]
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
        ToolMain.Args cli = ToolMain.args(
                args, "--set", "--size", "--pools-path", "--exclude-cards");
        String setCode = cli.get("--set");
        String poolsPath = cli.get("--pools-path", "./output/sealed/pools/");
        String excludePath = cli.get("--exclude-cards");

        ToolMain.run(() -> {
            int poolCount = cli.intValue("--size", 10000);
            ForgeEnvironmentInitializer.initialize();
            writePools(setCode, poolCount, Path.of(poolsPath), excludePath);
        });
    }

    private static void writePools(
            String setCode, int poolCount, Path outputDir, String excludePath)
            throws java.io.IOException {
        // A depleted run: the named cards never enter a booster, so no
        // training game holds one and none has to be discarded later.
        HoldoutCards excluded = HoldoutCards.NONE;
        if (excludePath != null) {
            excluded = HoldoutCards.load(Path.of(excludePath));
            System.out.println("Depleting pools of " + excluded.size() + " cards from "
                    + excludePath);
        }

        PoolGenerator generator = new PoolGenerator();
        Files.createDirectories(outputDir);
        Path outputFile = outputDir.resolve("pools.txt");

        List<String> eligibleSets = null;
        Random random = null;
        if (setCode == null) {
            eligibleSets = MatchGenerator.computeEligibleSets();
            if (eligibleSets.isEmpty()) {
                throw new IllegalStateException("no eligible sealed-legal sets found");
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

                List<List<String>> pools = generator.generate(poolSetCode, batch, excluded);
                for (List<String> pool : pools) {
                    writer.write(poolSetCode);
                    writer.write(CardNameList.FIELD_SEPARATOR);
                    writer.write(CardNameList.format(pool));
                    writer.newLine();
                    written++;
                }

                if (written % 1000 == 0) {
                    System.out.println("Generated " + written + "/" + poolCount + " pools");
                    System.out.flush();
                }
            }
        }

        System.out.println("Done: " + poolCount + " pools written to " + outputFile);
    }
}
