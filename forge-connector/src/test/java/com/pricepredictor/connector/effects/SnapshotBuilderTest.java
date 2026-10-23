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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
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

    // ── one depth for every record ──────────────────────────────────────

    /**
     * A depth a caller could choose would make {@code state.tiers} a proxy for
     * which collector wrote a record, so the builder takes no depth at all.
     */
    @Test
    void noConstructionChoosesATierDepth() {
        Constructor<?>[] constructors = SnapshotBuilder.class.getDeclaredConstructors();
        assertEquals(1, constructors.length,
                "a second constructor would let a call site pick its own depth");
        assertArrayEquals(
                new Class<?>[]{Game.class}, constructors[0].getParameterTypes());
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
}
