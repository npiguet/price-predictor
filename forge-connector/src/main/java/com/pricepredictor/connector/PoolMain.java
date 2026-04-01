package com.pricepredictor.connector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * CLI entry point for sealed pool generation.
 *
 * <p>Usage:
 * <pre>
 *   java -cp &lt;classpath&gt; com.pricepredictor.connector.PoolMain
 *       --set &lt;SET_CODE&gt;
 *       --size &lt;N&gt;
 *       --pools-path &lt;PATH&gt;
 * </pre>
 *
 * <p>Writes {@code pools.txt} to the specified path — one pool per line,
 * card names separated by semicolons. Streams a progress line to stdout
 * every 1000 pools. Exits 0 on success, 1 on error.
 */
public class PoolMain {

    public static void main(String[] args) {
        CliArgs cli = CliArgs.parse(args);
        String setCode = cli.get("--set", "RVR");
        int poolCount = parsePoolCount(cli.get("--size", "10000"));
        String poolsPath = cli.get("--pools-path", "./output/sealed/pools/");

        try {
            ForgeEnvironmentInitializer.initialize();
            Path outputDir = Path.of(poolsPath);
            Files.createDirectories(outputDir);
            Path outputFile = outputDir.resolve("pools.txt");

            System.out.println("Generating " + poolCount + " " + setCode + " sealed pools...");
            writePoolsInBatches(new PoolGenerator(), setCode, poolCount, outputFile);
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

    private static int parsePoolCount(String value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            System.err.println("Error: --size must be an integer");
            System.exit(1);
            return 0; // unreachable
        }
    }

    private static void writePoolsInBatches(
            PoolGenerator generator, String setCode, int poolCount, Path outputFile) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            int written = 0;
            while (written < poolCount) {
                int batch = Math.min(1000, poolCount - written);
                List<List<String>> pools = generator.generate(setCode, batch);
                for (List<String> pool : pools) {
                    writer.write(String.join(";", pool));
                    writer.newLine();
                    written++;
                }
                if (written % 1000 == 0) {
                    System.out.println("Generated " + written + "/" + poolCount + " pools");
                    System.out.flush();
                }
            }
        }
    }
}
