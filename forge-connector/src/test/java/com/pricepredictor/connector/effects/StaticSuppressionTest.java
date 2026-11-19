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

    /** An {@code untapped_production} with no mana, and with one green. */
    private static final String NO_MANA = "{\"W\":0,\"U\":0,\"B\":0,\"R\":0,\"G\":0,\"C\":0}";
    private static final String ONE_GREEN = "{\"W\":0,\"U\":0,\"B\":0,\"R\":0,\"G\":1,\"C\":0}";
    private static final String ONE_COLORLESS = "{\"W\":0,\"U\":0,\"B\":0,\"R\":0,\"G\":0,\"C\":1}";

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
            "Frogify, Prodigal Pyromancer",
            "Drake Umbra, Grizzly Bears",
            "Gryff's Boon, Grizzly Bears"})
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
     * An Equipment's static as an Aura's: the equipped creature's entity in
     * the Equipment's continuous record equals the creature rendered once the
     * Equipment has left the battlefield.
     */
    @Test
    void theBoardWithoutAnEquipmentsStaticIsTheBoardWithoutTheEquipment() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card target = board.battlefield("Grizzly Bears", board.you);
        Card sledge = board.battlefield("Behemoth Sledge", board.you);
        sledge.attachToEntity(target, null);
        board.game.getAction().checkStaticAbilities();
        assertTrue(target.hasKeyword(forge.game.keyword.Keyword.TRAMPLE), "the sledge is not on the bears");

        String record = continuousRecordOf(board, sledge);
        Map<String, Object> suppressed = entity(record, Refs.of(target));

        board.game.getAction().moveToGraveyard(board.game.getCardState(sledge), null, null);
        board.game.getAction().checkStaticAbilities();
        Map<String, Object> without = parse(EntityJson.render(
                board.game, board.game.getCardState(target), StaticSuppression.NONE));

        assertEquals(List.of(), differences(without, suppressed),
                "Behemoth Sledge's board still shows its own work on Grizzly Bears");
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

    /**
     * A player's {@code untapped_production} on a continuous record's board
     * counts no mana ability the static granted: on Chromatic Lantern's
     * record each land makes its own colour only, and the Lantern its own
     * printed any-colour mana, where on the live board every land also
     * makes any colour.
     */
    @Test
    void aPlayersProductionLeavesOutTheManaAbilitiesTheStaticGranted() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card lantern = board.battlefield("Chromatic Lantern", board.you);
        board.battlefield("Forest", board.you);
        board.battlefield("Island", board.you);
        board.game.getAction().checkStaticAbilities();

        assertProduction(board, lantern,
                "{\"W\":3,\"U\":4,\"B\":3,\"R\":3,\"G\":4,\"C\":0}",
                "{\"W\":1,\"U\":2,\"B\":1,\"R\":1,\"G\":2,\"C\":0}");
    }

    /**
     * A permanent's production counts for its controller on the board
     * without the static: on Annex's record the opponent's Forest makes the
     * opponent's green, not yours.
     */
    @Test
    void aPlayersProductionCountsWhatTheyControlWithoutTheStatic() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card forest = board.battlefield("Forest", board.opponent);
        Card annex = enchant(board, "Annex", forest);
        assertEquals(board.you, forest.getController(), "Annex did not take the Forest");

        assumeSuppression();
        String record = continuousRecordOf(board, annex);
        assertEquals(parse(NO_MANA), production(player(record, board.you)),
                "your production without Annex");
        assertEquals(parse(ONE_GREEN), production(player(record, board.opponent)),
                "the opponent's production without Annex");
    }

    /**
     * A creature that came under its controller's control this turn taps for
     * mana only with the static's haste: on Fires of Yavimaya's record
     * Llanowar Elves, summoning sick there, makes nothing.
     */
    @Test
    void aPlayersProductionLeavesOutWhatTheStaticsHasteAllowed() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card fires = board.battlefield("Fires of Yavimaya", board.you);
        Card elves = board.battlefield("Llanowar Elves", board.you);
        board.game.getAction().checkStaticAbilities();
        assertTrue(elves.isFirstTurnControlled() && !elves.isSick(), "the elves are not hasted");

        assertProduction(board, fires, ONE_GREEN, NO_MANA);
    }

    /**
     * A permanent the static makes a creature is summoning sick only with
     * the static: on March of the Machines' record a Sol Ring that entered
     * this turn, no creature there, taps for its colourless, where on the
     * live board it is a sick creature and makes nothing. The live card is
     * as sick after the snapshot as before.
     */
    @Test
    void aPlayersProductionCountsAHostOnlyTheStaticMadeSummoningSick() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card march = board.battlefield("March of the Machines", board.you);
        Card ring = board.battlefield("Sol Ring", board.you);
        board.game.getAction().checkStaticAbilities();
        assertTrue(ring.isCreature() && ring.isSick(), "March did not make the Sol Ring a sick creature");

        assertProduction(board, march, NO_MANA, ONE_COLORLESS);
        assertTrue(ring.isSick() && ring.getView().isFirstTurnControlled(),
                "the snapshot changed the Sol Ring's sickness");
    }

    /**
     * A mana ability whose colours depend on its activator's permanents is
     * read with the controller the board without the static gives it: on
     * Control Magic's record the opponent's Bloom Tender sees the opponent's
     * permanents, itself and a Hill Giant (green and red), not yours, the
     * Grizzly Bears and Control Magic (green and blue). The live card keeps
     * its controller and its ability's activator.
     */
    @Test
    void aPlayersProductionReadsItsColoursWithTheControllerWithoutTheStatic() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Hill Giant", board.opponent);
        Card magic = enchant(board, "Control Magic", board.battlefield("Bloom Tender", board.opponent));
        Card tender = magic.getAttachedTo();
        tender.setSickness(false); // Control Magic resolved on an earlier turn
        forge.game.spellability.SpellAbility mana = tender.getManaAbilities().get(0);
        forge.game.player.Player activator = mana.getActivatingPlayer();

        assertEquals(parse(greenAnd("U")), production(parse(PlayerJson.render(
                board.game, board.you, StaticSuppression.NONE))), "the live board's production");
        assumeSuppression();
        String record = continuousRecordOf(board, magic);
        assertEquals(parse(greenAnd("R")), production(player(record, board.opponent)),
                "the opponent's production without Control Magic");
        assertEquals(parse(NO_MANA), production(player(record, board.you)),
                "your production without Control Magic");
        assertEquals(board.you, tender.getController(), "the snapshot changed the Tender's controller");
        assertEquals(board.you.getView(), tender.getView().getController(), "nor its view's");
        assertEquals(activator, mana.getActivatingPlayer(), "the snapshot changed its activator");
    }

    /** One green and one of {@code color}, as an {@code untapped_production}. */
    private static String greenAnd(String color) {
        return NO_MANA.replace("\"G\":0", "\"G\":1")
                .replace("\"" + color + "\":0", "\"" + color + "\":1");
    }

    /**
     * Your production on the live board is {@code live}, and on the board of
     * {@code host}'s continuous record {@code without}.
     */
    private void assertProduction(CharacterizationBoard board, Card host, String live, String without)
            throws Throwable {
        assertEquals(parse(live), production(parse(PlayerJson.render(
                board.game, board.you, StaticSuppression.NONE))), "the live board's production");
        assumeSuppression();
        assertEquals(parse(without), production(player(continuousRecordOf(board, host), board.you)),
                "the board without " + host.getName() + "'s static");
    }

    private static Map<String, Object> production(Map<String, Object> player) {
        return asMap(player.get("untapped_production"));
    }

    private static Map<String, Object> player(String record, forge.game.player.Player player) {
        Map<String, Object> state = asMap(parse(record).get("state"));
        for (Object each : (List<?>) state.get("players")) {
            Map<String, Object> map = asMap(each);
            if (Refs.of(player).equals(map.get("id"))) {
                return map;
            }
        }
        throw new AssertionError("no " + Refs.of(player) + " in " + record);
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
