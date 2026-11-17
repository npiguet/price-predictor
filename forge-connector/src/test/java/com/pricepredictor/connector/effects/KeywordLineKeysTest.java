package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The keys a keyword's lines declare (findings 1.1, 1.4). A keyword that
 * prints one line declares its bare key there; one that prints several (a
 * saga's chapters, haunt, a Class level with continuation text) declares on
 * each line the {@code built} keys of the traits that line prints, and its
 * bare key on none. The kind-indexed key of a trait a keyword built is
 * declared nowhere, not even in {@code dropped_keys}.
 */
@ExtendWith(ForgeExtension.class)
class KeywordLineKeysTest {

    /**
     * The keys {@link #noKeyIsDeclaredByTwoLinesOfAFace} checks. Keyword keys
     * for now; every key once each line of a multi-line text declares only
     * its own.
     */
    private static final Predicate<ProvenanceKey> CHECKED =
            key -> key.traitKind() == TraitKind.KEYWORD;

    private final RulesParser parser = new RulesParser();

    private ConvertedScript convert(String relativePath) {
        return ConvertedScript.fromFile(parser, ConvertedScript.GOLDEN_024_CARDS, relativePath);
    }

    /** The keys of the one line whose text starts with {@code prefix}. */
    private static List<ProvenanceKey> keysOfLine(ConvertedScript converted, String prefix) {
        List<ProvenanceSidecar.Line> lines = converted.sidecar().lines().stream()
                .filter(line -> converted.texts().get(line.lineIndex()).startsWith(prefix))
                .toList();
        assertEquals(1, lines.size(), prefix + " in " + converted.texts());
        return lines.get(0).provenance();
    }

    /** Every key a line or {@code dropped_keys} of {@code converted} holds. */
    private static List<ProvenanceKey> everyKey(ConvertedScript converted) {
        List<ProvenanceKey> keys = new ArrayList<>(converted.sidecar().droppedKeys());
        converted.sidecar().lines().forEach(line -> keys.addAll(line.provenance()));
        return keys;
    }

    private static ProvenanceKey keyword(String relativePath, int index) {
        return new ProvenanceKey("cardsfolder/" + relativePath, 0, TraitKind.KEYWORD, index);
    }

    private static ProvenanceKey builtTrigger(ProvenanceKey keyword, int index) {
        return keyword.withBuilt(TraitKind.TRIGGER, index);
    }

    @Test
    void eachChapterLineDeclaresTheBuiltKeysOfItsOwnChapters() {
        ConvertedScript benalia = convert("h/history_of_benalia.txt");
        ProvenanceKey saga = keyword("h/history_of_benalia.txt", 0);

        assertEquals(List.of(builtTrigger(saga, 0), builtTrigger(saga, 1)),
                keysOfLine(benalia, "chapter: I, II "));
        assertEquals(List.of(builtTrigger(saga, 2)), keysOfLine(benalia, "chapter: III "));
        assertFalse(everyKey(benalia).contains(saga), "the saga's bare key: " + everyKey(benalia));
        assertFalse(everyKey(benalia).stream().anyMatch(key -> key.traitKind() == TraitKind.TRIGGER),
                "a chapter's kind-indexed key: " + everyKey(benalia));
    }

    /**
     * Blind Hunter's haunt builds five triggers: the two that run its ability
     * (enters, and the haunted creature dies) print the "when" line; the one
     * that haunts and the two that end the haunting print the haunt line.
     */
    @Test
    void aHauntCardsFiveTriggersEachLandOnTheLineThatStatesThem() {
        ConvertedScript hunter = convert("b/blind_hunter.txt");
        ProvenanceKey haunt = keyword("b/blind_hunter.txt", 1);

        assertEquals(List.of(builtTrigger(haunt, 2), builtTrigger(haunt, 4)),
                keysOfLine(hunter, "triggered: when "));
        assertEquals(List.of(builtTrigger(haunt, 0), builtTrigger(haunt, 1), builtTrigger(haunt, 3)),
                keysOfLine(hunter, "triggered: haunt"));
        assertFalse(everyKey(hunter).contains(haunt), "haunt's bare key: " + everyKey(hunter));
    }

    @Test
    void aCyclingLineDeclaresItsBareKeywordKeyAndNothingElse() {
        ConvertedScript censor = convert("c/censor.txt");
        ProvenanceKey cycling = keyword("c/censor.txt", 0);

        assertEquals(List.of(cycling), keysOfLine(censor, "activated[1]: cycling"));
        assertEquals(1, everyKey(censor).stream().filter(key -> key.traitKind() == TraitKind.SPELL).count(),
                "only the counterspell's own spell key, never cycling's: " + everyKey(censor));
    }

    /**
     * An Attraction's visit is one line whose built trigger is a charm: the
     * mode lines take the keyword's bare key with their option, the key a
     * record naming a mode with {@code built} falls back to.
     */
    @Test
    void aSingleLineKeywordsModeLinesTakeItsBareKeyWithTheirOption() {
        ConvertedScript stand = convert("b/balloon_stand.txt");
        ProvenanceKey visit = keyword("b/balloon_stand.txt", 0);

        assertEquals(List.of(visit), keysOfLine(stand, "triggered: visit"));
        assertEquals(List.of(visit.withOption(0)), keysOfLine(stand, "option[1]: "));
        assertEquals(List.of(visit.withOption(1)), keysOfLine(stand, "option[2]: "));
    }

    /**
     * Ranger Class's level 3 prints two lines: the level-up ability and the
     * level's static are its first line's; the continuation line repeats no
     * keyword key. Level 2 prints one line and keeps its bare key.
     */
    @Test
    void aClassLevelWithContinuationTextDeclaresItsBuiltKeysOnItsFirstLine() {
        ConvertedScript ranger = convert("r/ranger_class.txt");
        ProvenanceKey levelTwo = keyword("r/ranger_class.txt", 0);
        ProvenanceKey levelThree = keyword("r/ranger_class.txt", 1);

        assertEquals(List.of(levelTwo), keysOfLine(ranger, "level[2]: "));
        assertEquals(List.of(levelThree.withBuilt(TraitKind.SPELL, 0),
                        levelThree.withBuilt(TraitKind.STATIC, 0)),
                keysOfLine(ranger, "level[3]: "));
        assertEquals(List.of(), keysOfLine(ranger, "you may cast creature spells"));
    }

    // ── every golden script ─────────────────────────────────────────────

    private static final Path GOLDEN_TOKENS = Path.of("src/test/resources/golden-024/tokenscripts");

    private List<ConvertedScript> everyGoldenScript() throws IOException {
        List<ConvertedScript> converted = new ArrayList<>();
        try (Stream<Path> files = Files.walk(ConvertedScript.GOLDEN_024_CARDS)) {
            for (Path file : files.filter(p -> p.toString().endsWith(".txt")).sorted().toList()) {
                String relativePath = ConvertedScript.GOLDEN_024_CARDS.relativize(file)
                        .toString().replace('\\', '/');
                converted.add(convert(relativePath));
            }
        }
        try (Stream<Path> files = Files.list(GOLDEN_TOKENS)) {
            for (Path file : files.sorted().toList()) {
                String key = new ScriptFile(SourceTree.TOKENSCRIPTS, file.getFileName().toString()).key();
                converted.add(ConvertedScript.of(
                        parser.parseScript(ConvertedScript.source(GOLDEN_TOKENS, file.getFileName().toString()), key),
                        key));
            }
        }
        return converted;
    }

    @Test
    void noKeyIsDeclaredByTwoLinesOfAFace() throws IOException {
        List<ConvertedScript> golden = everyGoldenScript();
        for (ConvertedScript converted : golden) {
            Map<ProvenanceKey, Integer> declaringLine = new HashMap<>();
            for (ProvenanceSidecar.Line line : converted.sidecar().lines()) {
                for (ProvenanceKey key : line.provenance().stream().filter(CHECKED).toList()) {
                    Integer earlier = declaringLine.putIfAbsent(key, line.lineIndex());
                    assertTrue(earlier == null, key.describe() + " is declared by lines "
                            + earlier + " and " + line.lineIndex() + ": " + converted.texts());
                }
            }
        }
        assertTrue(golden.size() >= 40, "only " + golden.size() + " golden scripts");
    }
}
