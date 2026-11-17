package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.CardFace;
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
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The keys a keyword's lines declare (findings 1.1, 1.4). A keyword that
 * prints one line declares its bare key there; one that prints several (a
 * saga's chapters, a dungeon's rooms, haunt) declares on each line the
 * {@code built} keys of the traits that line prints, and its bare key on
 * none. The kind-indexed key of
 * a trait a keyword built is declared nowhere, not even in
 * {@code dropped_keys}. Over every golden script, no key of any kind is
 * declared by two lines of a face, and no spell of an instant or sorcery is
 * left without a line (findings 1.2, 1.3).
 */
@ExtendWith(ForgeExtension.class)
class KeywordLineKeysTest {

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

    /** The text of the one line that starts with {@code prefix}. */
    private static String converted(ConvertedScript converted, String prefix) {
        List<String> texts = converted.texts().stream()
                .filter(text -> text.startsWith(prefix)).toList();
        assertEquals(1, texts.size(), prefix + " in " + converted.texts());
        return texts.get(0);
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
     * Ranger Class's level 3 states two sentences on two oracle lines, folded
     * into one converted line: a keyword printing one line, it declares its
     * bare key there, as level 2 does.
     */
    @Test
    void aClassLevelWithContinuationTextIsOneLineDeclaringItsBareKey() {
        ConvertedScript ranger = convert("r/ranger_class.txt");
        ProvenanceKey levelTwo = keyword("r/ranger_class.txt", 0);
        ProvenanceKey levelThree = keyword("r/ranger_class.txt", 1);

        assertEquals(List.of(levelTwo), keysOfLine(ranger, "level[2]: "));
        assertEquals(List.of(levelThree), keysOfLine(ranger, "level[3]: "));
        assertTrue(converted(ranger, "level[3]: ").endsWith(
                " you may cast creature spells from the top of your library."),
                ranger.texts().toString());
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
                for (ProvenanceKey key : line.provenance()) {
                    Integer earlier = declaringLine.putIfAbsent(key, line.lineIndex());
                    assertTrue(earlier == null, key.describe() + " is declared by lines "
                            + earlier + " and " + line.lineIndex() + ": " + converted.texts());
                }
            }
        }
        assertTrue(golden.size() >= 40, "only " + golden.size() + " golden scripts");
    }

    private static final Pattern INSTANT_OR_SORCERY = Pattern.compile("\\b(instant|sorcery)\\b");

    /**
     * Every spell ability an instant or sorcery face declares is on a line:
     * none sits in {@code dropped_keys}, where a record acting through it
     * would find no text. A keyword's spell (flashback) is its keyword's,
     * and a charm with no description has its mode lines.
     */
    @Test
    void everySpellAbilityOfAnInstantOrSorceryHasALine() throws IOException {
        int spellFaces = 0;
        for (ConvertedScript converted : everyGoldenScript()) {
            List<CardFace> faces = converted.parsed().card().faces();
            for (int face = 0; face < faces.size(); face++) {
                if (!INSTANT_OR_SORCERY.matcher(faces.get(face).types()).find()) continue;
                spellFaces++;
                assertEquals(List.of(), droppedSpellKeys(converted, face),
                        "spell abilities with no line: " + converted.texts());
            }
        }
        assertTrue(spellFaces >= 10, "only " + spellFaces + " instant or sorcery faces");
    }

    /**
     * The spell keys of {@code face} in {@code dropped_keys}, but for a
     * charm with no description, whose mode lines are its lines.
     */
    private static List<ProvenanceKey> droppedSpellKeys(ConvertedScript converted, int face) {
        return converted.sidecar().droppedKeys().stream()
                .filter(key -> key.face() == face)
                .filter(key -> key.traitKind() == TraitKind.SPELL)
                .filter(key -> !everyKey(converted).contains(key.withOption(0)))
                .toList();
    }
}
