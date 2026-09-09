package com.pricepredictor.connector.effects;

import forge.game.Game;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
 * <p>Neither needs a game: the tier vector is decided in the constructor, and
 * the state's block order is decided by string assembly.
 */
class SnapshotBuilderTest {

    private static final int[] STAGE_TWO = {1, 2, 3};

    // ── the tier vector is a run-level value ────────────────────────────

    /**
     * The regression guard for the tier-4-means-interventional leak: a
     * convenience constructor is how a run-level property became a per-call-site
     * one, so its absence is the fix and this test is what states it.
     */
    @Test
    void everyConstructionStatesItsTierDepth() {
        Constructor<?>[] constructors = SnapshotBuilder.class.getConstructors();
        assertEquals(1, constructors.length,
                "a second constructor would let a call site pick its own depth");
        assertEquals(2, constructors[0].getParameterCount());
        assertEquals(Game.class, constructors[0].getParameterTypes()[0]);
        assertEquals(int[].class, constructors[0].getParameterTypes()[1]);
    }

    @Test
    void theRunsOwnTierVectorsAreAccepted() {
        new SnapshotBuilder(null, new int[]{1, 2});
        new SnapshotBuilder(null, new int[]{1, 2, 3});
        new SnapshotBuilder(null, new int[]{1, 2, 3, 4});
    }

    @Test
    void aTierVectorWithAHoleInItIsRefused() {
        // Tiers are cumulative, so [1,3] describes no collection anyone can run
        // — but it would be written into state.tiers as though it did.
        assertThrows(IllegalArgumentException.class,
                () -> new SnapshotBuilder(null, new int[]{1, 3}));
        assertThrows(IllegalArgumentException.class,
                () -> new SnapshotBuilder(null, new int[]{2, 3}));
    }

    @Test
    void aTierVectorThatOmitsTheBoardIsRefused() {
        // The battlefield and the command zone go in unconditionally, so a
        // vector without tier 2 would tell a reader the board was uncollected
        // while the board sits in the entity list.
        assertThrows(IllegalArgumentException.class,
                () -> new SnapshotBuilder(null, new int[]{1}));
        assertThrows(IllegalArgumentException.class,
                () -> new SnapshotBuilder(null, new int[]{}));
    }

    @Test
    void aTierVectorPastTheFourthIsRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> new SnapshotBuilder(null, new int[]{1, 2, 3, 4, 5}));
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
                new SnapshotBuilder(null, STAGE_TWO).refsJson(null));
    }

    /**
     * The refs block is re-rendered at resolution by a collector in another
     * class, so it has to be reachable from there.
     */
    @Test
    void refsCanBeReRenderedFromOutsideTheBuilder() throws Exception {
        Method refs = SnapshotBuilder.class.getMethod(
                "refsJson", forge.game.spellability.SpellAbility.class);
        assertTrue(Modifier.isPublic(refs.getModifiers()));
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
     * Refs sit immediately before the pending event, with nothing between them.
     *
     * <p>This is what lets a collector re-read refs at resolution — when the
     * chosen colour and the named card exist — and splice them into a state
     * captured before the ability resolved. The splice falls back to the
     * un-spliced state when it cannot find both markers, so a reorder would show
     * up as refs quietly staying empty rather than as a failure. Performing the
     * splice here is the only way that stays loud.
     */
    @Test
    void refsCanBeSplicedWithoutTouchingAnyOtherBlock() {
        String castTime = SnapshotBuilder.stateJson(
                "G", "P", "E", "{\"choices\":{}}", "null", "[1,2,3]");
        String resolved = "{\"choices\":{\"chosen_color\":\"U\"}}";

        int open = castTime.indexOf(SnapshotBuilder.REFS_KEY);
        int close = castTime.indexOf(SnapshotBuilder.AFTER_REFS_KEY);
        assertTrue(open >= 0 && close > open, "both splice markers present");

        String spliced = castTime.substring(0, open)
                + SnapshotBuilder.REFS_KEY + resolved
                + castTime.substring(close);

        assertEquals(
                SnapshotBuilder.stateJson("G", "P", "E", resolved, "null",
                        "[1,2,3]"),
                spliced);
    }

    /**
     * And the splice itself is one method, not one per collector.
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
}
