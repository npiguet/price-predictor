package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.RenderedLine;
import com.pricepredictor.connector.RulesParser;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * A parsed script rendered and given its sidecar, as {@code BatchConverter}
 * writes a converted card and its {@code .provenance.json}.
 */
record ConvertedScript(
        RulesParser.ParsedCard parsed, List<RenderedLine> rendered, ProvenanceSidecar sidecar) {

    /**
     * The real Forge scripts captured for feature 024, pinned so that an
     * upstream card edit cannot move a test that reads them; its README lists
     * the few synthetic ones.
     */
    static final Path GOLDEN_024_CARDS = Path.of("src/test/resources/golden-024/cardsfolder");

    /** The lines of the card script at {@code relativePath} under {@code root}. */
    static List<String> source(Path root, String relativePath) {
        try {
            return Files.readAllLines(root.resolve(relativePath));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Read the card script at {@code relativePath} under {@code root}, parse it
     * with {@code parser}, and convert it under its cardsfolder key.
     */
    static ConvertedScript fromFile(RulesParser parser, Path root, String relativePath) {
        String scriptFile = new ScriptFile(SourceTree.CARDSFOLDER, relativePath).key();
        return of(parser.parseScript(source(root, relativePath), scriptFile), scriptFile);
    }

    /** Render {@code parsed} and build its sidecar under {@code scriptFile}. */
    static ConvertedScript of(RulesParser.ParsedCard parsed, String scriptFile) {
        List<RenderedLine> rendered = parsed.card().renderLines();
        ProvenanceSidecar sidecar = ProvenanceSidecar.build(
                parsed.card().faces().get(0).name(), scriptFile,
                rendered, parsed.recorders());
        return new ConvertedScript(parsed, rendered, sidecar);
    }

    /** The rendered lines' text, in order. */
    List<String> texts() {
        return rendered.stream().map(RenderedLine::text).toList();
    }
}
