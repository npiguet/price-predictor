package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The script surface of a line a keyword-built trait prints: the trait's own
 * chain, its effect included, never the keyword's raw text.
 *
 * <p>Forge builds a chapter's trigger, a haunt trigger and an ETB
 * replacement with its effect set directly, so no {@code Execute$} or
 * {@code ReplaceWith$} parameter names it; the chain still follows it.
 */
@ExtendWith(ForgeExtension.class)
class KeywordTraitScriptTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private static ConvertedScript convert(String relativePath) {
        return ConvertedScript.fromFile(new RulesParser(), CARDS_FOLDER, relativePath);
    }

    private static List<ProvenanceSidecar.Line> linesOfKind(ConvertedScript card, String kind) {
        return card.sidecar().lines().stream().filter(line -> line.lineKind().equals(kind)).toList();
    }

    private static String onlyScriptOfKind(ConvertedScript card, String kind) {
        List<ProvenanceSidecar.Line> lines = linesOfKind(card, kind);
        assertEquals(1, lines.size(), kind + " lines in " + card.texts());
        return lines.get(0).script().scriptText();
    }

    @Test
    void eachChapterLineCarriesItsChaptersEffect() {
        List<String> scripts = linesOfKind(convert("t/the_eldest_reborn.txt"), "chapter").stream()
                .map(line -> line.script().scriptText()).toList();
        assertEquals(3, scripts.size(), "chapters: " + scripts);
        List<String> effects = List.of("DB$ Sacrifice", "DB$ Discard", "DB$ ChangeZone");
        for (int chapter = 0; chapter < 3; chapter++) {
            String script = scripts.get(chapter);
            assertTrue(script.contains(" [SEG] SV1: " + effects.get(chapter)), script);
        }
        assertEquals(3, Set.copyOf(scripts).size(), "the three chapters differ: " + scripts);
    }

    @Test
    void aSagaPrintingOneChapterLineCarriesTheChaptersTriggerNotTheKeyword() {
        String script = onlyScriptOfKind(convert("s/summon_magus_sisters.txt"), "chapter");
        assertFalse(script.startsWith("Chapter:"), script);
        assertTrue(script.contains("Mode$ CounterAdded") && script.contains(" [SEG] SV1: "), script);
        assertTrue(script.contains("DB$ Charm"), script);
    }

    @Test
    void anEtbReplacementKeywordLineCarriesTheReplacementsChain() {
        ConvertedScript clone = convert("c/clone.txt");
        ProvenanceSidecar.Line line = linesOfKind(clone, "replacement").get(0);
        String script = line.script().scriptText();
        assertFalse(script.startsWith("ETBReplacement:"), script);
        assertTrue(script.contains("Event$ Moved | ") && script.contains(" [SEG] SV1: ")
                && script.contains("DB$ Clone"), script);
        assertEquals("Clone", line.script().apiType());
        assertEquals(Set.of("keyword 0"), keysOf(line), "the line keeps its bare keyword key");
    }

    @Test
    void anEtbCounterKeywordLineCarriesThePutCounter() {
        ProvenanceSidecar.Line line = linesOfKind(convert("w/walking_ballista.txt"), "replacement").get(0);
        String script = line.script().scriptText();
        assertFalse(script.startsWith("etbCounter:"), script);
        assertTrue(script.contains(" [SEG] SV1: ") && script.contains("DB$ PutCounter"), script);
        assertTrue(script.contains("CounterType$ P1P1"), script);
    }

    @Test
    void aHauntTriggerLineCarriesItsEffect() {
        for (ProvenanceSidecar.Line line : linesOfKind(convert("a/absolver_thrull.txt"), "triggered")) {
            assertTrue(line.script().scriptText().contains(" [SEG] SV1: "), line.script().scriptText());
        }
    }

    private static Set<String> keysOf(ProvenanceSidecar.Line line) {
        return line.provenance().stream()
                .map(key -> key.traitKind().wireName() + " " + key.indexWithinKind()
                        + (key.built() == null ? "" : " built"))
                .collect(Collectors.toSet());
    }
}
