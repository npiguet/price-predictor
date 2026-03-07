package com.pricepredictor.connector;

import com.esotericsoftware.minlog.Log;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

/**
 * Batch converts all Forge card scripts in a directory tree.
 */
public class BatchConverter {

    private final CardScriptConverter converter = new CardScriptConverter();

    /**
     * Convert all .txt card scripts in cardsPath, writing output to outputPath
     * with mirrored directory structure.
     */
    public BatchResult convert(Path cardsPath, Path outputPath) throws IOException {
        int totalFiles = 0;
        int succeeded = 0;
        List<String> warnings = new ArrayList<>();

        List<Path> scriptFiles = new ArrayList<>();
        Files.walkFileTree(cardsPath, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                if (file.toString().endsWith(".txt")) {
                    scriptFiles.add(file);
                }
                return FileVisitResult.CONTINUE;
            }
        });

        for (Path scriptFile : scriptFiles) {
            totalFiles++;
            try {
                List<String> lines = Files.readAllLines(scriptFile);
                String filename = scriptFile.getFileName().toString();
                MultiCard result = converter.convertCard(lines, filename);
                if (result == null) {
                    String cardName = scriptFile.getFileName().toString().replace(".txt", "");
                    warnings.add("[" + cardName + "] skipped (unsupported card type)");
                    continue;
                }
                String output = OutputFormatter.formatMultiCard(result);

                // Mirror directory structure
                Path relativePath = cardsPath.relativize(scriptFile);
                Path outputFile = outputPath.resolve(relativePath);
                Files.createDirectories(outputFile.getParent());
                Files.writeString(outputFile, output);
                succeeded++;
            } catch (Exception e) {
                String cardName = scriptFile.getFileName().toString().replace(".txt", "");
                String warning = "[" + cardName + "] " + e.getMessage();
                warnings.add(warning);
                Log.warn("BatchConverter", warning);
            }
        }

        return new BatchResult(totalFiles, succeeded, warnings);
    }

    public record BatchResult(int totalFiles, int succeeded, List<String> warnings) {
        public int warningCount() {
            return warnings.size();
        }
    }
}
