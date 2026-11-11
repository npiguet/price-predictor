package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A {@code continuous} record's board is the board without its static (base
 * record schema, "Rules"): every channel the static wrote comes back out, so
 * the permanent it acts on reads as it would with the static gone — type,
 * colour, keywords, set and boosted power and toughness, controller, and the
 * lines it added or took away (B1, L2).
 */
@ExtendWith(ForgeExtension.class)
class StaticSuppressionTest {

    @TempDir
    Path tempDir;

    /**
     * Every member of the affected permanent's entity in the aura's
     * continuous record equals the same permanent rendered after the aura
     * has left the battlefield: the board without the static is the board
     * without the aura, field by field.
     */
    @ParameterizedTest(name = "{0} on {1}")
    @CsvSource({
            "Imprisoned in the Moon, Serra Angel",
            "Imprisoned in the Moon, Prodigal Pyromancer",
            "Control Magic, Grizzly Bears",
            "Ensoul Artifact, Sol Ring",
            "Frogify, Serra Angel",
            "Frogify, Prodigal Pyromancer"})
    void theBoardWithoutAnAurasStaticIsTheBoardWithoutTheAura(String auraName, String targetName)
            throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card target = board.battlefield(targetName, board.opponent);
        Card aura = enchant(board, auraName, target);

        String record = continuousRecordOf(board, aura);
        Map<String, Object> suppressed = entity(record, Refs.of(target));

        board.game.getAction().moveToGraveyard(board.game.getCardState(aura), null, null);
        board.game.getAction().checkStaticAbilities();
        Map<String, Object> without = parse(EntityJson.render(
                board.game, board.game.getCardState(target), StaticSuppression.NONE));

        assertEquals(List.of(), differences(without, suppressed),
                auraName + "'s board still shows its own work on " + targetName);
    }

    /**
     * A static that only sets base power and toughness is a continuous
     * record: its contribution says what it sets, and its board shows the
     * base it set over (L2).
     */
    @Test
    void aStaticThatOnlySetsBasePowerAndToughnessIsRecorded() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card godhead = board.battlefield("Godhead of Awe", board.you);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        board.game.getAction().checkStaticAbilities();

        String record = continuousRecordOf(board, godhead);

        Map<String, Object> contribution = contributionTo(record, bears);
        assertEquals(List.of(1L, 1L), contribution.get("set_pt"), record);
        assertEquals(List.of(0L, 0L), contribution.get("pt_boost"), record);
        assertEquals(List.of(2L, 2L), entity(record, Refs.of(bears)).get("base_pt"), record);
    }

    /**
     * Frogify's contribution sets the 1/1 and boosts nothing, beside a board
     * that shows the creature's own base rather than the 1/1 already (L2).
     */
    @Test
    void aFrogifyContributionSetsTheBaseItsBoardDoesNotShowYet() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card angel = board.battlefield("Serra Angel", board.opponent);
        Card frogify = enchant(board, "Frogify", angel);

        String record = continuousRecordOf(board, frogify);

        Map<String, Object> contribution = contributionTo(record, angel);
        assertEquals(List.of(1L, 1L), contribution.get("set_pt"), record);
        assertEquals(List.of(0L, 0L), contribution.get("pt_boost"), record);
        assertEquals(List.of(4L, 4L), entity(record, Refs.of(angel)).get("base_pt"), record);
    }

    /**
     * A static that changes only who controls a permanent is a continuous
     * record naming the controller it gives, on a board where the permanent
     * is still its owner's.
     */
    @Test
    void aStaticThatGivesControlIsRecordedWithTheControllerItGives() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card magic = enchant(board, "Control Magic", bears);
        assertEquals(board.you, board.game.getCardState(bears).getController(),
                "the aura did not take control");

        String record = continuousRecordOf(board, magic);

        assertEquals(Refs.of(board.you), contributionTo(record, bears).get("controller"), record);
        assertEquals(Refs.of(board.opponent), entity(record, Refs.of(bears)).get("controller"),
                record);
    }

    /** A contribution that changes no controller and sets no base carries neither member. */
    @Test
    void anAnthemsContributionSetsNoBaseAndGivesNoControl() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card anthem = board.battlefield("Glorious Anthem", board.you);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.game.getAction().checkStaticAbilities();

        Map<String, Object> contribution =
                contributionTo(continuousRecordOf(board, anthem), bears);

        assertFalse(contribution.containsKey("set_pt"), String.valueOf(contribution));
        assertFalse(contribution.containsKey("controller"), String.valueOf(contribution));
        assertEquals(List.of(1L, 1L), contribution.get("pt_boost"));
    }

    /**
     * A static that takes away only some of a permanent's lines leaves the
     * rest keyed at their printed places: Blood Sun takes City of Shadows'
     * storage ability (printed first) and leaves its mana ability (printed
     * second), which keys to the sidecar's second spell line on the live
     * board, not to the first because it now stands first in the live list.
     * On the board without Blood Sun both lines are back, each at its own key.
     */
    @Test
    void aLineAStaticLeavesKeepsItsPrintedKey() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card city = board.battlefield("City of Shadows", board.you);
        Card bloodSun = board.battlefield("Blood Sun", board.you);
        board.game.getAction().checkStaticAbilities();
        Card printed = TestCards.build("City of Shadows");
        ProvenanceKey printedStorage = ProvenanceResolver.of(ability(printed, false));
        ProvenanceKey printedMana = ProvenanceResolver.of(ability(printed, true));
        // The land's play ability stands first in Forge's spell list.
        assertEquals(List.of(1, 2),
                List.of(printedStorage.indexWithinKind(), printedMana.indexWithinKind()),
                "the sidecar's spell lines: the storage ability, then the mana ability");
        assertTrue(city.getSpellAbilities().stream().noneMatch(
                        sa -> !sa.isManaAbility() && !sa.isLandAbility()),
                "Blood Sun left the storage ability in force");

        assertEquals(printedMana, ProvenanceResolver.of(ability(city, true)),
                "the mana ability Blood Sun leaves");
        assertEquals(List.of(printedMana.describe()), lineKeys(parse(EntityJson.render(
                board.game, city, StaticSuppression.NONE))), "the live board's lines");

        assumeSuppression();
        Map<String, Object> suppressed =
                entity(continuousRecordOf(board, bloodSun), Refs.of(city));
        assertEquals(List.of(printedStorage.describe(), printedMana.describe()),
                lineKeys(suppressed), "the board without Blood Sun's static");
    }

    /**
     * A static that takes one printed keyword away leaves the other keyed to
     * its printed line: Gravity Sphere takes Serra Angel's flying, and its
     * vigilance keys where the sidecar puts it, on the live board and on the
     * board without Gravity Sphere, beside flying there.
     */
    @Test
    void aKeywordAStaticLeavesKeepsItsPrintedKey() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card angel = board.battlefield("Serra Angel", board.you);
        Card sphere = board.battlefield("Gravity Sphere", board.you);
        board.game.getAction().checkStaticAbilities();
        Card printed = TestCards.build("Serra Angel");
        String flying = keywordKey(printed, "Flying");
        String vigilance = keywordKey(printed, "Vigilance");
        assertFalse(angel.hasKeyword(forge.game.keyword.Keyword.FLYING), "Gravity Sphere left flying");

        assertEquals(List.of(vigilance), lineKeys(parse(EntityJson.render(
                board.game, angel, StaticSuppression.NONE)), "keyword"), "the live board's lines");

        assumeSuppression();
        assertEquals(List.of(flying, vigilance).stream().sorted().toList(),
                lineKeys(entity(continuousRecordOf(board, sphere), Refs.of(angel)), "keyword")
                        .stream().sorted().toList(),
                "the board without Gravity Sphere's static");
    }

    /** The key of {@code card}'s printed keyword {@code name}, as {@link #lineKeys} spells it. */
    private static String keywordKey(Card card, String name) {
        for (forge.game.keyword.KeywordInterface keyword
                : card.getCurrentState().getIntrinsicKeywords()) {
            if (name.equals(keyword.getOriginal())) {
                ProvenanceKey key = ProvenanceResolver.keyOfKeyword(card.getCurrentState(), keyword);
                return "(" + key.scriptFile() + ") face" + key.face() + "/keyword["
                        + key.indexWithinKind() + "]";
            }
        }
        throw new AssertionError(card + " prints no " + name);
    }

    /** City of Shadows' mana ability, or its storage ability. */
    private static forge.game.spellability.SpellAbility ability(Card card, boolean mana) {
        return card.getSpellAbilities().stream()
                .filter(sa -> !sa.isLandAbility() && sa.isManaAbility() == mana)
                .findFirst()
                .orElseThrow(() -> new AssertionError(card + " has no such ability"));
    }

    /** Every spell line's keys an entity lists, as {@link ProvenanceKey#describe}, in order. */
    private static List<String> lineKeys(Map<String, Object> entity) {
        return lineKeys(entity, "spell");
    }

    /** Every {@code kind} line's keys an entity lists, as {@link ProvenanceKey#describe}, in order. */
    private static List<String> lineKeys(Map<String, Object> entity, String kind) {
        List<String> keys = new ArrayList<>();
        for (Object line : (List<?>) entity.get("lines")) {
            for (Object key : (List<?>) asMap(line).getOrDefault("keys", List.of())) {
                Map<String, Object> map = asMap(key);
                if (kind.equals(map.get("trait_kind"))) {
                    keys.add("(" + map.get("script_file") + ") face" + map.get("face")
                            + "/" + kind + "[" + map.get("index_within_kind") + "]");
                }
            }
        }
        return keys;
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private static void assumeSuppression() {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
    }

    /** Cast an aura of yours at {@code target} and resolve it; the aura on the battlefield. */
    private static Card enchant(CharacterizationBoard board, String auraName, Card target) {
        Card inHand = board.hand(auraName, board.you);
        board.cast(inHand, board.you, target);
        board.game.getStack().resolveStack();
        board.game.getAction().checkStaticAbilities();
        Card aura = board.game.getCardState(inHand);
        assertTrue(aura.isInPlay() && aura.getAttachedTo() != null,
                auraName + " did not resolve onto " + target);
        return aura;
    }

    /** The one continuous record a boundary writes for {@code host}'s static. */
    private String continuousRecordOf(CharacterizationBoard board, Card host) throws Throwable {
        List<String> records = ShardRun.records(tempDir, board.game,
                (bracket, collectors) -> collectors.continuous().collect());
        String script = ScriptLocator.scriptFileOf(host);
        List<String> found = new ArrayList<>();
        for (String record : records) {
            if ("continuous".equals(parse(record).get("kind"))
                    && script.equals(RecordLines.scriptOf(record))) {
                found.add(record);
            }
        }
        assertEquals(1, found.size(), "continuous records of " + script + ": " + records);
        return found.get(0);
    }

    private static Map<String, Object> entity(String record, String ref) {
        Map<String, Object> state = asMap(parse(record).get("state"));
        for (Object entity : (List<?>) state.get("entities")) {
            Map<String, Object> map = asMap(entity);
            if (ref.equals(map.get("id"))) {
                return map;
            }
        }
        throw new AssertionError("no " + ref + " in " + record);
    }

    private static Map<String, Object> contributionTo(String record, Card card) {
        for (Object each : (List<?>) RecordLines.payload(record).get("contributions")) {
            Map<String, Object> contribution = asMap(each);
            if (Refs.of(card).equals(contribution.get("entity"))) {
                return contribution;
            }
        }
        throw new AssertionError("no contribution to " + card + " in " + record);
    }

    /** Each member whose value differs, as {@code name: expected != actual}. */
    private static List<String> differences(Map<String, Object> expected, Map<String, Object> actual) {
        Set<String> names = new TreeSet<>(expected.keySet());
        names.addAll(actual.keySet());
        List<String> out = new ArrayList<>();
        for (String name : names) {
            if (!Objects.equals(expected.get(name), actual.get(name))) {
                out.add(name + ": " + expected.get(name) + " != " + actual.get(name));
            }
        }
        return out;
    }
}
