package com.pricepredictor.connector;

import com.esotericsoftware.minlog.Log;
import com.pricepredictor.connector.effects.ProvenanceRecorder;
import com.pricepredictor.connector.effects.ProvenanceSidecar;
import com.pricepredictor.connector.effects.SourceTree;

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
 *
 * <p>Each card produces two files: the converted text and a
 * {@code <name>.provenance.json} sidecar beside it, joining every rendered line
 * back to the runtime traits that produced it. Both come from one render pass,
 * so the sidecar's line indices always describe the file that was written.
 */
public class BatchConverter {

    private final RulesParser converter = new RulesParser();

    /**
     * Convert all .txt card scripts in cardsPath, writing output to outputPath
     * with mirrored directory structure. Sidecars are written for the
     * {@code cardsfolder} tree.
     */
    public BatchResult convert(Path cardsPath, Path outputPath) throws IOException {
        return convert(cardsPath, outputPath, SourceTree.CARDSFOLDER);
    }

    /**
     * Convert a source tree, writing a provenance sidecar beside each card.
     *
     * @param tree which converted source tree this is; it prefixes every
     *             provenance key's script path, because one filename occurs in
     *             more than one tree and must resolve differently in each
     */
    public BatchResult convert(Path cardsPath, Path outputPath, String tree)
            throws IOException {
        return convert(cardsPath, outputPath, tree, true);
    }

    /**
     * Convert a source tree, optionally writing only the sidecars.
     *
     * <p>Stage four's variant tree is the sidecar-only case. A variant has no
     * oracle text — nobody printed the card — so converting it to prose would
     * put text in the corpus no card has (FR-056). The sidecar is still needed,
     * and must come from this parser rather than a reimplementation: a
     * provenance key's {@code index_within_kind} is the trait's position in
     * Forge's own runtime trait list, so only the parser that builds that list
     * can number it the way the collectors will.
     *
     * @param writeConvertedText false to write the sidecar alone, leaving the
     *                           source script in place when {@code outputPath}
     *                           is the source directory
     */
    public BatchResult convert(
            Path cardsPath, Path outputPath, String tree, boolean writeConvertedText)
            throws IOException {
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
                Path relativePath = cardsPath.relativize(scriptFile);
                // The key's script path is the source path as written, tree
                // included — produced identically here and by the Python reader.
                String scriptKeyPath = tree + "/"
                        + relativePath.toString().replace('\\', '/');

                RulesParser.ParsedCard parsed =
                        converter.parseScript(lines, filename, scriptKeyPath);
                MultiCard result = parsed.card();

                List<Ability> owners = new ArrayList<>();
                List<Integer> faceOfLine = new ArrayList<>();
                List<String> renderedLines = result.renderLines(owners, faceOfLine);
                String output = String.join("\n", renderedLines);

                Path outputFile = outputPath.resolve(relativePath);
                Files.createDirectories(outputFile.getParent());
                if (writeConvertedText) {
                    Files.writeString(outputFile, output);
                }

                ProvenanceSidecar sidecar = ProvenanceSidecar.build(
                        result.faces().get(0).name(), scriptKeyPath, result,
                        renderedLines, owners, faceOfLine, parsed.recorders());
                Files.writeString(sidecarPathFor(outputFile), sidecar.toJson());
                succeeded++;
            } catch (Throwable t) {
                // Forge's card-rules parser raises AssertionError on
                // malformed scripts as well as IOException/RuntimeException;
                // catch Throwable so one bad fixture can't abort the batch.
                String cardName = scriptFile.getFileName().toString().replace(".txt", "");
                String warning = "[" + cardName + "] " + t.getMessage();
                warnings.add(warning);
                Log.warn("BatchConverter", warning);
            }
        }

        return new BatchResult(totalFiles, succeeded, warnings);
    }

    /** The sidecar beside a converted {@code .txt}, the pairing {@code .npz} uses. */
    public static Path sidecarPathFor(Path convertedTxt) {
        String name = convertedTxt.getFileName().toString();
        String stem = name.endsWith(".txt")
                ? name.substring(0, name.length() - 4) : name;
        return convertedTxt.resolveSibling(stem + ".provenance.json");
    }

    public record BatchResult(int totalFiles, int succeeded, List<String> warnings) {
        public int warningCount() {
            return warnings.size();
        }
    }
}
