package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.spellability.OptionalCost;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The snapshot properties that a running game cannot check for you.
 *
 * <p>The tier vector is the one a corpus cannot show to be wrong: chosen per
 * call site, it would make {@code state.tiers} a perfect proxy for a
 * collection flag the schema bars from being a model input. Beside it, the
 * state's block order and the
 * resolution-time refs reading that replaces the cast-time block and nothing
 * else.
 *
 * <p>None of those needs a game: the tier vector is a constant, and the
 * blocks are assembled from already-rendered parts. The referenced tier,
 * pinned last, does: a target is a card in some zone.
 */
class SnapshotBuilderTest {

    // ── refs ────────────────────────────────────────────────────────────

    /**
     * A record with no acting spell ability still renders the whole refs shape.
     *
     * <p>Trigger, rewrite, playability and combat records pass no ability, and a
     * reader that had to tell "no refs block" from "an empty one" would be
     * reading collection wiring rather than the board.
     */
    @Test
    void refsRenderTheirFullShapeWithNoActingLine() {
        assertEquals(
                "{\"targets\":[],\"source\":null,\"modes\":[],\"x\":null,"
                        + "\"choices\":{}}",
                RefsJson.render(null));
    }

    // ── the state's blocks ──────────────────────────────────────────────

    /**
     * The blocks in schema order, ending in all four tiers: every snapshot
     * declares them, so an empty hand or stack reads as empty rather than as
     * uncollected. The literal is the contract: Python's {@code InclusionTier}
     * and {@code record_io} mirror it.
     */
    @Test
    void theStateBlocksKeepTheOrderTheSchemaFixes() {
        assertEquals(
                "{\"global\":G,\"players\":P,\"entities\":E,\"refs\":R,"
                        + "\"pending_event\":V,\"tiers\":[1,2,3,4]}",
                new Snapshot("G", "P", "E", "R", "V").render());
    }

    /**
     * A resolution-time refs reading replaces the cast-time block and leaves
     * the board it was captured with alone: re-reading the board at resolution
     * would describe what the ability had already changed.
     */
    @Test
    void laterRefsReplaceOnlyTheRefsBlock() {
        Snapshot castTime = new Snapshot("G", "P", "E", "{\"choices\":{}}", "null");
        String resolved = "{\"choices\":{\"chosen_color\":\"U\"}}";

        assertEquals(
                new Snapshot("G", "P", "E", resolved, "null"),
                castTime.withRefs(resolved));
        assertEquals("{\"choices\":{}}", castTime.refs(),
                "the cast-time snapshot is a value; a later reading makes a new one");
    }

    // ── the referenced tier ─────────────────────────────────────────────

    /**
     * What an ability's snapshot carries whatever zone it sits in: its host
     * and the cards it targets.
     *
     * <p>One list for the bracket collector and the fork collector alike: a
     * target this list leaves out and no other tier sweeps, such as an exiled
     * card, is in no tier at all, and the record names a target its state
     * does not hold.
     */
    @Nested
    @ExtendWith(ForgeExtension.class)
    class ReferencedTier {

        @Test
        void anAbilitysTargetsAreReferencedWhereverTheySit() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card host = board.hand("Pull from Eternity", board.you);
            SpellAbility pull = board.spell(host, board.you);
            Card exiled = board.game.getAction().moveTo(
                    ZoneType.Exile, board.card("Grizzly Bears", board.opponent), null, null);
            pull.getTargets().add(exiled);
            pull.getTargets().add(board.opponent);

            String state = new SnapshotBuilder(board.game).ofResolving(pull).render();

            assertTrue(state.contains("{\"id\":\"" + Refs.of(exiled)
                    + "\",\"name\":\"Grizzly Bears\",\"zone\":\"exile\""), state);
        }

        @Test
        void noAbilityReferencesNothing() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            SnapshotBuilder snapshots = new SnapshotBuilder(board.game);

            assertEquals(snapshots.ofBoard().render(), snapshots.ofResolving(null).render());
        }
    }

    // ── a card's lines (FR-021 to FR-023) ───────────────────────────────

    /**
     * Story 2's acceptance 1 to 5 and the face-down case: what a card slot's
     * {@code lines} carries, read off the rendered entity.
     */
    @Nested
    @ExtendWith(ForgeExtension.class)
    class Lines {

        /**
         * Acceptance 1: Guile's static, trigger and replacement are each a
         * line keyed into its own script, and the static carries
         * {@code active}, which holds on the battlefield.
         */
        @Test
        void aCreaturesStaticTriggerAndReplacementAreLinesWithTheirKeys() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guile = board.battlefield("Guile", board.you);

            List<Map<String, Object>> lines = linesOf(board, guile);

            assertEquals(List.of("trigger", "static", "replacement"), kinds(lines));
            for (Map<String, Object> line : lines) {
                Map<String, Object> key = onlyKey(line);
                assertEquals("cardsfolder/g/guile.txt", key.get("script_file"), line.toString());
                assertEquals(line.get("kind"), key.get("trait_kind"), line.toString());
                assertEquals(0L, key.get("index_within_kind"), line.toString());
                assertEquals(false, line.get("granted"), line.toString());
            }
            assertEquals(true, line(lines, "static").get("active"));
            assertFalse(line(lines, "trigger").containsKey("active"),
                    "active is a static's: " + lines);
        }

        /** A static whose condition does not hold says so: no threshold, no flying. */
        @Test
        void aStaticWhoseConditionFailsIsInactive() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card enforcer = board.battlefield("Mystic Enforcer", board.you);

            assertEquals(false, line(linesOf(board, enforcer), "static").get("active"));
        }

        /**
         * Acceptance 2: under Humility a creature has lost all its abilities,
         * and the live lists Forge plays from carry none; neither does its
         * plain cast, which is no line of its own.
         */
        @Test
        void aCreatureThatLostAllAbilitiesListsNone() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guile = board.battlefield("Guile", board.you);
            board.battlefield("Humility", board.opponent);
            board.game.getAction().checkStaticAbilities();

            assertEquals(List.of(), linesOf(board, guile));
        }

        /**
         * Acceptance 3: flying from Levitation is a keyword line on the
         * creature, granted, keyed to the static that grants it.
         */
        @Test
        void aKeywordAnotherPermanentsStaticGrantsIsListedGranted() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            board.battlefield("Levitation", board.you);
            board.game.getAction().checkStaticAbilities();

            Map<String, Object> flying = line(linesOf(board, bears), "keyword");
            assertEquals("Flying", flying.get("fragment"));
            assertEquals(true, flying.get("granted"));
            Map<String, Object> key = onlyKey(flying);
            assertEquals("cardsfolder/l/levitation.txt", key.get("script_file"));
            assertEquals("static", key.get("trait_kind"));
        }

        /**
         * Acceptance 4: ward {2} and annihilator 2 are keyword lines carrying
         * their instance's fragment, value included, and keyed to the keyword
         * the card prints. Annihilator's trigger is that line, not one more.
         */
        @Test
        void wardAndAnnihilatorAreKeywordLinesWithTheirValues() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card spawn = board.battlefield("Aboleth Spawn", board.you);
            Card crusher = board.battlefield("Ulamog's Crusher", board.you);

            Map<String, Object> ward = keywordLine(linesOf(board, spawn), "Ward:2");
            assertEquals(false, ward.get("granted"));
            assertEquals("keyword", onlyKey(ward).get("trait_kind"));
            List<Map<String, Object>> crusherLines = linesOf(board, crusher);
            Map<String, Object> annihilator = keywordLine(crusherLines, "Annihilator:2");
            assertEquals("cardsfolder/u/ulamogs_crusher.txt",
                    onlyKey(annihilator).get("script_file"));
            assertEquals(List.of("static", "keyword"), kinds(crusherLines),
                    "the keyword's own trigger is its line: " + crusherLines);
        }

        /** Acceptance 5: the protection line is the instance's, naming its quality. */
        @Test
        void protectionFromGoblinsNamesGoblins() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guide = board.battlefield("Foothill Guide", board.you);

            keywordLine(linesOf(board, guide), "Protection:Goblin");
        }

        /**
         * A face-down permanent's lines are its face-down state's, and its real
         * face rides beside them, every line flagged inactive.
         */
        @Test
        void aFaceDownPermanentCarriesItsRealFaceAsInactiveLines() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guide = board.battlefield("Foothill Guide", board.you);
            guide.turnFaceDown(true);

            List<Map<String, Object>> lines = linesOf(board, guide);

            Map<String, Object> protection = keywordLine(lines, "Protection:Goblin");
            assertEquals(false, protection.get("active"), lines.toString());
            Map<String, Object> morph = keywordLine(lines, "Morph:W");
            assertEquals(false, morph.get("active"), lines.toString());
            assertEquals("cardsfolder/f/foothill_guide.txt", onlyKey(morph).get("script_file"));
        }

        /** A spell on the stack cast with its kicker says so on the kicker's line. */
        @Test
        void aKickedSpellOnTheStackMarksItsKickerLine() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card onStack = board.game.getAction().moveToStack(
                    board.hand("Burst Lightning", board.you), null);
            SpellAbility burst = board.spell(onStack, board.you);
            burst.addOptionalCost(OptionalCost.Kicker1);
            burst.resetTargets();
            burst.getTargets().add(board.opponent);
            board.game.getStack().add(burst);

            List<Map<String, Object>> lines = linesOf(board, onStack);

            assertEquals(1L, keywordLine(lines, "Kicker:4").get("cast_with"), lines.toString());
            assertFalse(line(lines, "spell").containsKey("cast_with"), lines.toString());
        }

        /** A charm on the stack names the modes it chose on its own line. */
        @Test
        void aCharmOnTheStackNamesItsChosenModes() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card onStack = board.game.getAction().moveToStack(
                    board.hand("Boros Charm", board.you), null);
            SpellAbility charm = TestBoard.charmOf(onStack, board.you);
            // Mode 1, permanents you control gain indestructible: no target to choose.
            TestBoard.chain(charm, 1);
            board.game.getStack().add(charm);

            List<Map<String, Object>> lines = linesOf(board, onStack);

            assertEquals(List.of(1L), line(lines, "spell").get("chosen"), lines.toString());
        }

        /** Face up, the same lines are in force and carry no flag. */
        @Test
        void aFaceUpPermanentsLinesAreNotFlaggedInactive() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guide = board.battlefield("Foothill Guide", board.you);

            for (Map<String, Object> line : linesOf(board, guide)) {
                assertFalse(line.containsKey("active"), line.toString());
            }
        }
    }

    // ── base power and toughness (FR-025) ───────────────────────────────

    @Nested
    @ExtendWith(ForgeExtension.class)
    class PowerAndToughness {

        /**
         * Ensoul Artifact sets the base (layer 7b): Ornithopter's base is
         * 5/5, not its printed 0/2, and nothing is a boost.
         */
        @Test
        void ensoulArtifactSetsTheBase() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card thopter = board.battlefield("Ornithopter", board.you);
            board.battlefield("Ensoul Artifact", board.you).attachToEntity(thopter, null);
            board.game.getAction().checkStaticAbilities();

            Map<String, Object> entity = entityOf(board, thopter);
            assertEquals(List.of(5L, 5L), entity.get("base_pt"));
            assertEquals(Map.of("temporary", List.of(0L, 0L), "counters", List.of(0L, 0L)),
                    entity.get("pt_boosts"));
        }

        /**
         * Boosts apart from the base: an anthem's +1/+1 is temporary, and
         * every counter type adds what Forge counts for it.
         */
        @Test
        void boostsAndEveryCountersContributionAreApartFromTheBase() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            board.battlefield("Glorious Anthem", board.you);
            bears.addCounterInternal(CounterEnumType.P1P1, 1, board.you, false, null, null);
            bears.addCounterInternal(CounterEnumType.P1P2, 1, board.you, false, null, null);
            board.game.getAction().checkStaticAbilities();

            Map<String, Object> entity = entityOf(board, bears);
            assertEquals(List.of(2L, 2L), entity.get("base_pt"));
            assertEquals(Map.of("temporary", List.of(1L, 1L), "counters", List.of(2L, 3L)),
                    entity.get("pt_boosts"));
            assertEquals(2, RecordLines.asMap(entity.get("counters")).size(),
                    "both counter types, each by name: " + entity.get("counters"));
        }

        @Test
        void aNoncreatureHasNoPowerOrToughness() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Map<String, Object> entity = entityOf(board, board.battlefield("Forest", board.you));

            assertNull(entity.get("base_pt"));
            assertNull(entity.get("pt_boosts"));
        }
    }

    // ── reading an entity ───────────────────────────────────────────────

    private static Map<String, Object> entityOf(CharacterizationBoard board, Card card) {
        return RecordLines.parse(EntityJson.render(board.game, card, null));
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> linesOf(CharacterizationBoard board, Card card) {
        return (List<Map<String, Object>>) entityOf(board, card).get("lines");
    }

    private static List<Object> kinds(List<Map<String, Object>> lines) {
        return lines.stream().map(line -> line.get("kind")).toList();
    }

    /** The one line of this kind. */
    private static Map<String, Object> line(List<Map<String, Object>> lines, String kind) {
        List<Map<String, Object>> found = lines.stream()
                .filter(line -> kind.equals(line.get("kind"))).toList();
        assertEquals(1, found.size(), lines.toString());
        return found.get(0);
    }

    /** The one keyword line written with this fragment. */
    private static Map<String, Object> keywordLine(
            List<Map<String, Object>> lines, String fragment) {
        List<Map<String, Object>> found = lines.stream()
                .filter(line -> "keyword".equals(line.get("kind"))
                        && fragment.equals(line.get("fragment"))).toList();
        assertEquals(1, found.size(), "one " + fragment + " line in " + lines);
        return found.get(0);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> onlyKey(Map<String, Object> line) {
        List<Object> keys = (List<Object>) line.get("keys");
        assertEquals(1, keys.size(), line.toString());
        return RecordLines.asMap(keys.get(0));
    }
}
