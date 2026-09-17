package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.BatchConverter;
import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.MultiCard;
import com.pricepredictor.connector.RulesParser;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The provenance sidecar: the join between runtime traits and converted lines.
 *
 * <p>The byte-identity check is the important one. Every other feature here is
 * additive, and the whole design depends on the sidecar shipping without
 * reconverting the corpus — so a change that alters the rendered text is a
 * failure even if the sidecar itself is perfect.
 */
@ExtendWith(ForgeExtension.class)
class ProvenanceSidecarTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser parser = new RulesParser();

    @TempDir
    Path tempDir;

    private record Converted(MultiCard card, List<String> lines,
                             List<Ability> owners, List<Integer> faceOfLine,
                             ProvenanceSidecar sidecar) {
    }

    private Converted convert(String relativePath) {
        try {
            Path file = CARDS_FOLDER.resolve(relativePath);
            return convertSource(Files.readAllLines(file),
                    SourceTree.CARDSFOLDER + "/" + relativePath,
                    file.getFileName().toString());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Convert a script from its lines rather than from the card tree.
     *
     * <p>Some trait shapes the converter handles have no shipped card: a Class
     * level whose text is also registered as a trait of its own is one, and the
     * removal path it takes is only reachable from a script written for it.
     */
    private Converted convertSource(List<String> source, String scriptFile,
                                    String fileName) {
        RulesParser.ParsedCard parsed =
                parser.parseScript(source, fileName, scriptFile);
        List<Ability> owners = new ArrayList<>();
        List<Integer> faceOfLine = new ArrayList<>();
        List<String> lines = parsed.card().renderLines(owners, faceOfLine);
        ProvenanceSidecar sidecar = ProvenanceSidecar.build(
                parsed.card().faces().get(0).name(), scriptFile, parsed.card(),
                lines, owners, faceOfLine, parsed.recorders());
        return new Converted(parsed.card(), lines, owners, faceOfLine, sidecar);
    }

    // ── the text must not change ────────────────────────────────────────

    /**
     * A sample spanning single-face, multi-face, keyword-heavy, and modal
     * cards. Recording provenance renders the same bytes it always did.
     */
    private static final List<String> SAMPLE = List.of(
            "s/serra_angel.txt",
            "l/lightning_bolt.txt",
            "a/ajanis_pridemate.txt",
            "w/wall_of_omens.txt",
            "c/counterspell.txt"
    );

    @Test
    void convertedTextIsByteIdenticalToTheRecordingFreeRendering() {
        for (String relativePath : SAMPLE) {
            Path file = CARDS_FOLDER.resolve(relativePath);
            if (!Files.exists(file)) continue;
            List<String> source;
            try {
                source = Files.readAllLines(file);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            String withoutRecording = parser
                    .parseScript(source, file.getFileName().toString())
                    .formatText();
            String withRecording = String.join("\n", convert(relativePath).lines());
            assertEquals(withoutRecording, withRecording,
                    "recording provenance changed the converted text of "
                            + relativePath);
        }
    }

    @Test
    void renderedLinesJoinBackToTheFormattedText() {
        Converted converted = convert("s/serra_angel.txt");
        assertEquals(converted.card().formatText(),
                String.join("\n", converted.lines()));
    }

    // ── the join ────────────────────────────────────────────────────────

    @Test
    void everyAbilityLineIsIndexedByTheSidecar() {
        Converted converted = convert("s/serra_angel.txt");
        long abilityLines = converted.owners().stream().filter(o -> o != null).count();
        assertEquals(abilityLines, converted.sidecar().lines().size());
    }

    @Test
    void aLineIndexPointsAtTheLineItDescribes() {
        Converted converted = convert("a/ajanis_pridemate.txt");
        for (ProvenanceSidecar.Line line : converted.sidecar().lines()) {
            assertTrue(line.lineIndex() >= 0
                            && line.lineIndex() < converted.lines().size(),
                    "line_index out of range: " + line.lineIndex());
            assertTrue(converted.lines().get(line.lineIndex())
                            .startsWith(line.lineKind()),
                    "line_kind " + line.lineKind() + " does not match the line "
                            + converted.lines().get(line.lineIndex()));
        }
    }

    @Test
    void headerLinesBelongToNoAbilityAndAreNotIndexed() {
        Converted converted = convert("s/serra_angel.txt");
        assertTrue(converted.lines().get(0).startsWith("name: "));
        assertNotNull(converted.owners().get(0) == null ? "ok" : null,
                "the name line must belong to no ability");
        for (ProvenanceSidecar.Line line : converted.sidecar().lines()) {
            assertFalse(converted.lines().get(line.lineIndex()).startsWith("name: "));
        }
    }

    @Test
    void aKeywordLineKeysToItsKeyword() {
        Converted converted = convert("s/serra_angel.txt");
        boolean sawKeywordKey = converted.sidecar().lines().stream()
                .flatMap(line -> line.provenance().stream())
                .anyMatch(key -> ProvenanceKey.KIND_KEYWORD.equals(key.traitKind()));
        assertTrue(sawKeywordKey, "Serra Angel's flying and vigilance key to keywords");
    }

    @Test
    void everyKeyNamesTheCardsOwnScriptFile() {
        Converted converted = convert("a/ajanis_pridemate.txt");
        for (ProvenanceSidecar.Line line : converted.sidecar().lines()) {
            for (ProvenanceKey key : line.provenance()) {
                assertEquals("cardsfolder/a/ajanis_pridemate.txt", key.scriptFile());
            }
        }
    }

    @Test
    void aTraitThatProducedNoLineLandsInDroppedKeys() {
        // A trait with no text of its own — the implicit spell every permanent
        // has - produces no line. It is still live at runtime, so a record can
        // still name it, and the join must resolve it to "no line" rather than
        // failing. A trait whose text a line does carry is claimed by that
        // line and never lands here.
        List<ProvenanceKey> allDropped = new ArrayList<>();
        for (String relativePath : SAMPLE) {
            if (!Files.exists(CARDS_FOLDER.resolve(relativePath))) continue;
            allDropped.addAll(convert(relativePath).sidecar().droppedKeys());
        }
        for (ProvenanceKey key : allDropped) {
            assertNotNull(key.traitKind());
            assertTrue(key.indexWithinKind() >= 0);
        }
    }

    @Test
    void noKeyAppearsBothOnALineAndInDroppedKeys() {
        for (String relativePath : SAMPLE) {
            if (!Files.exists(CARDS_FOLDER.resolve(relativePath))) continue;
            Converted converted = convert(relativePath);
            List<ProvenanceKey> claimed = converted.sidecar().lines().stream()
                    .flatMap(line -> line.provenance().stream()).toList();
            for (ProvenanceKey dropped : converted.sidecar().droppedKeys()) {
                assertFalse(claimed.contains(dropped),
                        dropped + " is both rendered and dropped in " + relativePath);
            }
        }
    }

    // ── the synthetic land mana line claims its runtime abilities ────────

    @Test
    void aBasicLandsManaLineClaimsTheRuntimeManaAbility() {
        // Forge builds the mana ability from the land type at runtime; the
        // converter renders it as one canonical line. Before this claim the
        // line had no key and the runtime ability sat in dropped_keys, so
        // every mana activation record joined to nothing.
        Converted mountain = convert("m/mountain.txt");
        List<ProvenanceSidecar.Line> mana = mountain.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals("activated")).toList();
        assertEquals(1, mana.size(), mountain.lines().toString());
        List<ProvenanceKey> keys = mana.get(0).provenance();
        assertEquals(1, keys.size(), keys.toString());
        assertEquals(ProvenanceKey.KIND_SPELL, keys.get(0).traitKind());
        assertTrue(keys.get(0).indexWithinKind() >= 1, keys.toString());
        assertEquals(1, mountain.sidecar().droppedKeys().size(),
                mountain.sidecar().droppedKeys().toString());
        assertEquals(0, mountain.sidecar().droppedKeys().get(0).indexWithinKind());
    }

    @Test
    void aDualLandsManaLineClaimsBothRuntimeManaAbilities() {
        Converted tundra = convert("t/tundra.txt");
        List<ProvenanceSidecar.Line> mana = tundra.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals("activated")).toList();
        assertEquals(1, mana.size(), tundra.lines().toString());
        assertEquals(2, mana.get(0).provenance().size(), mana.get(0).provenance().toString());
    }

    // ── every trait whose text a line carries is claimed by that line ────

    private static List<ProvenanceKey> claimedBy(Converted converted, String kind, String needle) {
        return converted.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals(kind))
                .filter(line -> converted.lines().get(line.lineIndex()).contains(needle))
                .flatMap(line -> line.provenance().stream())
                .toList();
    }

    @Test
    void theSecondaryOfAnEntersOrDiesPairIsClaimedByTheRenderedLine() {
        // Forge registers "enters or dies" as two triggers with one
        // description; the converter renders one line. The second object is
        // still live and a record fired by it names its key.
        Converted marshal = convert("m/mogg_war_marshal.txt");
        List<ProvenanceKey> keys = claimedBy(marshal, "triggered", "enters or dies");
        assertTrue(keys.stream().anyMatch(k -> k.indexWithinKind() == 0), keys.toString());
        assertTrue(keys.stream().anyMatch(k -> k.indexWithinKind() == 1), keys.toString());
    }

    @Test
    void aKeywordDerivedTriggerIsClaimedByTheKeywordLine() {
        // Echo's upkeep trigger is a runtime Trigger object with a key of its
        // own; its text is the keyword line.
        Converted marshal = convert("m/mogg_war_marshal.txt");
        List<ProvenanceKey> keys = claimedBy(marshal, "triggered", "echo");
        assertTrue(keys.stream().anyMatch(k -> k.traitKind().equals(ProvenanceKey.KIND_KEYWORD)), keys.toString());
        assertTrue(keys.stream().anyMatch(k -> k.traitKind().equals(ProvenanceKey.KIND_TRIGGER)), keys.toString());
    }

    @Test
    void onlyTheImplicitPermanentSpellIsDroppedOnMoggWarMarshal() {
        Converted marshal = convert("m/mogg_war_marshal.txt");
        List<ProvenanceKey> dropped = marshal.sidecar().droppedKeys();
        assertEquals(1, dropped.size(), dropped.toString());
        assertEquals(ProvenanceKey.KIND_SPELL, dropped.get(0).traitKind());
        assertEquals(0, dropped.get(0).indexWithinKind());
    }

    @Test
    void anEntersOrAttacksPairIsClaimedByTheRenderedLine() {
        Converted tidalmage = convert("s/stadium_tidalmage.txt");
        List<ProvenanceKey> keys = claimedBy(tidalmage, "triggered", "enters or attacks");
        assertEquals(2, keys.stream().filter(k -> k.traitKind().equals(ProvenanceKey.KIND_TRIGGER)).count(), keys.toString());
    }

    @Test
    void aClassCardsLevelLinesCarryTheirTraitsKeys() {
        // Class post-processing rebuilds level lines as fresh objects; the
        // attribution has to move with them or every ability on the card is
        // textless to the model.
        Converted talent = convert("h/hunters_talent.txt");
        List<ProvenanceKey> levelKeys = talent.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals("level"))
                .flatMap(line -> line.provenance().stream())
                .toList();
        assertTrue(levelKeys.stream().anyMatch(k -> k.traitKind().equals(ProvenanceKey.KIND_TRIGGER)), levelKeys.toString());
        assertTrue(levelKeys.stream().anyMatch(k -> k.traitKind().equals(ProvenanceKey.KIND_STATIC)), levelKeys.toString());
        for (ProvenanceKey dropped : talent.sidecar().droppedKeys()) {
            assertFalse(dropped.traitKind().equals(ProvenanceKey.KIND_TRIGGER)
                    || dropped.traitKind().equals(ProvenanceKey.KIND_STATIC),
                    dropped + " has text on a level line and must not be dropped");
        }
    }

    @Test
    void aKeywordDerivedReplacementIsClaimedByTheLineItPrints() {
        // Bard Class prints its level-1 ability from an ETBReplacement keyword.
        // Forge also registers the replacement effect itself, and that object
        // is the one a record about the counter names.
        Converted bard = convert("b/bard_class.txt");
        List<ProvenanceKey> levelOne = claimedBy(bard, "level", "additional +1/+1 counter");
        assertTrue(levelOne.stream().anyMatch(
                        k -> k.traitKind().equals(ProvenanceKey.KIND_REPLACEMENT)),
                levelOne.toString());
        for (ProvenanceKey dropped : bard.sidecar().droppedKeys()) {
            assertFalse(dropped.traitKind().equals(ProvenanceKey.KIND_REPLACEMENT),
                    dropped + " prints the level 1 line and must not be dropped");
        }
    }

    @Test
    void aTraitRemovedAsAClassLevelDuplicateKeepsItsKeyOnTheLevelLine() {
        // The trait and the Class keyword that prints it both reach the parser,
        // and only the keyword's entry survives. The surviving level line has
        // to carry the removed trait's key: the trait is what fires at runtime.
        Converted card = convertSource(List.of(
                "Name:Test Class Duplicate",
                "ManaCost:G",
                "Types:Enchantment Class",
                "T:Mode$ Attacks | ValidCard$ Creature.YouCtrl | TriggerZones$ Battlefield"
                        + " | Execute$ TrigDraw | TriggerDescription$ Whenever a creature"
                        + " you control attacks, draw a card.",
                "SVar:TrigDraw:DB$ Draw | Defined$ You | NumCards$ 1",
                "K:Class:2:1 G:AddTrigger$ TriggerAttack",
                "SVar:TriggerAttack:Mode$ Attacks | ValidCard$ Creature.YouCtrl"
                        + " | TriggerZones$ Battlefield | Execute$ TrigDraw"
                        + " | TriggerDescription$ Whenever a creature you control attacks,"
                        + " draw a card."),
                SourceTree.CARDSFOLDER + "/t/test_class_duplicate.txt",
                "test_class_duplicate.txt");
        List<ProvenanceKey> keys = claimedBy(card, "level", "draw a card");
        assertTrue(keys.stream().anyMatch(
                        k -> k.traitKind().equals(ProvenanceKey.KIND_TRIGGER)),
                card.lines() + " " + keys);
        for (ProvenanceKey dropped : card.sidecar().droppedKeys()) {
            assertFalse(dropped.traitKind().equals(ProvenanceKey.KIND_TRIGGER),
                    dropped + " has its text on the level line and must not be dropped");
        }
    }

    // ── the script surface and role spans ───────────────────────────────

    @Test
    void anAbilityLineCarriesItsScriptApiTypeAndParameterKeys() {
        Converted converted = convert("l/lightning_bolt.txt");
        boolean sawScript = converted.sidecar().lines().stream()
                .anyMatch(line -> line.script().apiType() != null
                        && !line.script().paramKeys().isEmpty());
        assertTrue(sawScript, "a spell line should carry its API type and params");
    }

    @Test
    void roleSpansCoverTheProseOfEveryIndexedLine() {
        Converted converted = convert("a/ajanis_pridemate.txt");
        for (ProvenanceSidecar.Line line : converted.sidecar().lines()) {
            String prose = ProvenanceSidecar.prose(
                    converted.lines().get(line.lineIndex()));
            for (RoleSpans.Span span : line.roleSpans()) {
                assertTrue(span.start() >= 0 && span.end() <= prose.length(),
                        "span " + span + " outside the prose of: " + prose);
                assertTrue(span.start() <= span.end());
            }
        }
    }

    // ── the file on disk ────────────────────────────────────────────────

    @Test
    void aSidecarIsWrittenBesideEveryConvertedCard() throws IOException {
        new BatchConverter().convert(
                Path.of("src/test/resources/cardsfolder"), tempDir,
                SourceTree.CARDSFOLDER);
        assertTrue(Files.exists(tempDir.resolve("t/test_bear.provenance.json")));
        assertTrue(Files.exists(tempDir.resolve("t/test_flyer.provenance.json")));
    }

    @Test
    void aWrittenSidecarIsWellFormedJsonNamingItsCard() throws IOException {
        new BatchConverter().convert(
                Path.of("src/test/resources/cardsfolder"), tempDir,
                SourceTree.CARDSFOLDER);
        String json = Files.readString(tempDir.resolve("t/test_bear.provenance.json"));
        assertTrue(json.startsWith("{\"card\":"), json);
        assertTrue(json.contains("\"script_file\":\"cardsfolder/t/test_bear.txt\""),
                json);
        assertTrue(json.contains("\"lines\":["), json);
        assertTrue(json.contains("\"dropped_keys\":["), json);
        assertTrue(json.endsWith("}"), json);
    }

    /**
     * The converter and the resolver must compute one keyword ordinal.
     *
     * <p>This is the coupling guard for the keyword slot, and it compares the
     * two sides keyword by keyword rather than as a set — both rules produce
     * the ordinals 0..4 on a five-keyword card, so only the pairing shows a
     * disagreement.
     *
     * <p>The runtime resolver keys a flashback or cycling ability to
     * {@code keyword@N} through {@link ProvenanceKey#keywordIndex}. If the
     * converter numbers the same keywords by any other rule, every such record
     * names a key that is in neither {@code lines} nor {@code dropped_keys},
     * which is the sidecar contract's fail-loudly case — a whole collection run
     * of unjoinable keyword records. The two sides therefore have to come from
     * one function, the same discipline {@code faceIndex} already enforces for
     * faces.
     *
     * <p>Blast from the Past is the card that can tell them apart: with five
     * printed keywords, the hash order of {@code KeywordCollection} — a
     * {@code MultimapBuilder.hashKeys()} over an enum whose {@code hashCode()}
     * is the identity hash, and so not reproducible from one JVM to the next —
     * and the printed-text order genuinely disagree.
     */
    @Test
    void everyKeywordKeyAgreesWithTheSharedOrdinal() {
        Converted converted = convert("b/blast_from_the_past.txt");
        CardState state = TestCards.build("Blast from the Past").getCurrentState();

        Map<String, Integer> sharedOrdinal = new LinkedHashMap<>();
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            sharedOrdinal.put(keyword.getOriginal(),
                    ProvenanceKey.keywordIndex(state, keyword));
        }
        assertEquals(5, sharedOrdinal.size(),
                "Blast from the Past prints five keywords");

        int compared = 0;
        for (ProvenanceSidecar.Line line : converted.sidecar().lines()) {
            // A keyword line's only script surface is its original text, which
            // is what names the keyword the line's key is supposed to describe.
            if (!"Keyword".equals(line.script().apiType())) continue;
            Integer expected = sharedOrdinal.get(line.script().scriptText());
            if (expected == null) continue;
            for (ProvenanceKey key : line.provenance()) {
                if (!ProvenanceKey.KIND_KEYWORD.equals(key.traitKind())) continue;
                compared++;
                assertEquals(expected, key.indexWithinKind(),
                        "the sidecar numbered \"" + line.script().scriptText()
                                + "\" as keyword@" + key.indexWithinKind()
                                + " but ProvenanceKey.keywordIndex says keyword@"
                                + expected + ". RulesParser.parseFace must compute"
                                + " the ordinal with ProvenanceKey.keywordIndex("
                                + "card.getCurrentState(), ki) rather than with a"
                                + " running counter over card.getKeywords(), or"
                                + " every keyword record in the next corpus names"
                                + " a key the sidecar does not contain.");
            }
        }
        assertTrue(compared > 0, "no keyword line was compared");
    }

    @Test
    void theTokenTreePrefixesItsKeysDifferently() throws IOException {
        new BatchConverter().convert(
                Path.of("src/test/resources/cardsfolder"), tempDir,
                SourceTree.TOKENSCRIPTS);
        String json = Files.readString(tempDir.resolve("t/test_bear.provenance.json"));
        assertTrue(json.contains("\"script_file\":\"tokenscripts/t/test_bear.txt\""),
                json);
    }
}
