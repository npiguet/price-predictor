package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.reflect.Constructor;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The two snapshot properties that a running game cannot check for you.
 *
 * <p>Both defects these pin were invisible in a fourteen-million-record corpus:
 * a tier vector chosen per call site, which made {@code state.tiers} a perfect
 * proxy for a collection flag the schema bars from being a model input; and the
 * key ordering a resolution-time refs splice depends on, which degrades to the
 * un-spliced state rather than to an error when it stops holding.
 *
 * <p>Neither needs a game: the tier vector is a constant of the builder, and
 * the state's block order is decided by string assembly. The referenced tier,
 * pinned last, does: a target is a card in some zone.
 */
class SnapshotBuilderTest {

    // ── one depth for every record ──────────────────────────────────────

    /**
     * The regression guard for the tier-4-means-interventional leak: a depth a
     * caller could choose is how {@code state.tiers} became a proxy for which
     * collector wrote a record, so the builder takes no depth at all.
     */
    @Test
    void noConstructionChoosesATierDepth() {
        Constructor<?>[] constructors = SnapshotBuilder.class.getConstructors();
        assertEquals(1, constructors.length,
                "a second constructor would let a call site pick its own depth");
        assertArrayEquals(
                new Class<?>[]{Game.class}, constructors[0].getParameterTypes());
    }

    /**
     * Every snapshot declares all four tiers, so an empty hand or stack reads as
     * empty rather than as uncollected. The literal is the contract:
     * Python's {@code InclusionTier} and {@code record_io} mirror it.
     */
    @Test
    void everySnapshotDeclaresAllFourTiers() {
        assertEquals("[1,2,3,4]", SnapshotBuilder.TIERS_JSON);
    }

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
                new SnapshotBuilder(null).refsJson(null));
    }

    // ── the block order a resolution-time splice depends on ─────────────

    @Test
    void theStateBlocksKeepTheOrderTheSchemaFixes() {
        assertEquals(
                "{\"global\":G,\"players\":P,\"entities\":E,\"refs\":R,"
                        + "\"pending_event\":V,\"tiers\":T}",
                SnapshotBuilder.stateJson("G", "P", "E", "R", "V", "T"));
    }

    /**
     * The splice is one method, not one per collector.
     *
     * <p>Two collectors doing the same substring arithmetic by hand is how the
     * two readings drift, and this one degrades silently rather than failing --
     * so the arithmetic is stated once and asserted here.
     */
    @Test
    void spliceRefsReplacesOnlyTheRefsBlock() {
        String castTime = SnapshotBuilder.stateJson(
                "G", "P", "E", "{\"choices\":{}}", "null", "[1,2,3]");
        String resolved = "{\"choices\":{\"chosen_color\":\"U\"}}";

        assertEquals(
                SnapshotBuilder.stateJson("G", "P", "E", resolved, "null", "[1,2,3]"),
                SnapshotBuilder.spliceRefs(castTime, resolved));
    }

    /**
     * A state without both markers comes back untouched.
     *
     * <p>The splice widens a record; a caller left with the cast-time refs has
     * less than it wanted, while one handed a mangled state has a record that
     * cannot be read at all.
     */
    @Test
    void aStateWithoutBothMarkersIsReturnedUnchanged() {
        assertEquals("{\"global\":G}",
                SnapshotBuilder.spliceRefs("{\"global\":G}", "{}"));
        assertEquals("{\"refs\":{}}",
                SnapshotBuilder.spliceRefs("{\"refs\":{}}", "{}"));
        assertEquals(null, SnapshotBuilder.spliceRefs(null, "{}"));
        assertEquals("s", SnapshotBuilder.spliceRefs("s", null));
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

            List<Card> referenced = SnapshotBuilder.referencedOf(pull);

            assertEquals(List.of(host, exiled), referenced);
            String state = new SnapshotBuilder(board.game).toJson(pull, referenced);
            assertTrue(state.contains("{\"id\":\"" + SnapshotBuilder.entityId(exiled)
                    + "\",\"name\":\"Grizzly Bears\",\"zone\":\"exile\""), state);
        }

        @Test
        void noAbilityReferencesNothing() {
            assertEquals(List.of(), SnapshotBuilder.referencedOf(null));
        }
    }
}
