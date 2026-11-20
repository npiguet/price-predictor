package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which repeated events {@link EventBuffer} files once and which it files every time. */
class EventBufferTest {

    /**
     * The same transition told twice in a row for its subject is filed once.
     *
     * <p>A card is on the battlefield or it is not, so a second identical
     * {@code zone_change} with no other transition of the card between
     * carries nothing the first did not.
     */
    @Test
    void aStateTransitionToldTwiceInARowIsFiledOnce() {
        EventBuffer into = new EventBuffer();

        assertTrue(into.file(zoneChange()));
        assertTrue(into.file(new EffectEvent(EventType.TAPPED).subject("E8")));
        assertFalse(into.file(zoneChange()),
                "another subject's transition between is no second occurrence");

        assertEquals(2, into.events().size());
    }

    /**
     * The same transition after another of the same subject is a second
     * occurrence: a creature flickered twice by one resolution left and
     * returned twice, a permanent tapped, untapped and tapped again was
     * tapped twice.
     */
    @Test
    void aTransitionAfterAnotherOfItsSubjectIsFiledAgain() {
        EventBuffer into = new EventBuffer();
        EffectEvent leaves = move("E7", "battlefield", "exile");
        EffectEvent returns = move("E7", "exile", "battlefield");

        for (int flicker = 0; flicker < 2; flicker++) {
            assertTrue(into.file(leaves), "flicker " + flicker);
            assertTrue(into.file(returns), "flicker " + flicker);
        }
        for (int tap = 0; tap < 2; tap++) {
            assertTrue(into.file(new EffectEvent(EventType.TAPPED).subject("E9")));
            assertTrue(into.file(new EffectEvent(EventType.UNTAPPED).subject("E9")));
        }

        assertEquals(8, into.events().size());
    }

    /** A buffer copied from events refuses a repeat of their last transition. */
    @Test
    void aCopyRefusesARepeatOfTheLastTransition() {
        EventBuffer copy = EventBuffer.copyOf(java.util.List.of(zoneChange()));

        assertFalse(copy.file(zoneChange()));
        assertEquals(1, copy.events().size());
    }

    private static EffectEvent move(String card, String from, String to) {
        return new EffectEvent(EventType.ZONE_CHANGE)
                .subject(card).param(EventParam.FROM_ZONE, from).param(EventParam.TO_ZONE, to);
    }

    /**
     * A repeated quantity is not, because there the repeat is the information.
     *
     * <p>Two creatures each dealing one damage to the same blocker render two
     * identical events, and collapsing them would turn two damage into one.
     */
    @Test
    void aRepeatedQuantityIsFiledEveryTime() {
        EventBuffer into = new EventBuffer();

        assertTrue(into.file(oneDamage()));
        assertTrue(into.file(oneDamage()));

        assertEquals(2, into.events().size());
    }

    /**
     * A shuffle of a library takes the position off the moves into it before
     * it: the card was shuffled in, and sits nowhere in particular. A move
     * into another player's library keeps its position.
     */
    @Test
    void aShuffleTakesThePositionOffTheMovesIntoThatLibrary() {
        EventBuffer into = new EventBuffer();
        EffectEvent shuffledIn = intoLibrary("E7", 0).intoLibraryOf("P0");
        EffectEvent elsewhere = intoLibrary("E8", -1).intoLibraryOf("P1");
        into.file(shuffledIn);
        into.file(elsewhere);

        into.file(new EffectEvent(EventType.LIBRARY_SHUFFLED).subject("P0"));

        assertFalse(shuffledIn.hasParam(EventParam.LIBRARY_POSITION));
        assertEquals(-1, elsewhere.param(EventParam.LIBRARY_POSITION));
    }

    /** A move after the shuffle is where it was put. */
    @Test
    void aMoveAfterTheShuffleKeepsItsPosition() {
        EventBuffer into = new EventBuffer();
        into.file(new EffectEvent(EventType.LIBRARY_SHUFFLED).subject("P0"));
        EffectEvent onTop = intoLibrary("E7", 0).intoLibraryOf("P0");

        into.file(onTop);

        assertEquals(0, onTop.param(EventParam.LIBRARY_POSITION));
    }

    private static EffectEvent intoLibrary(String card, int position) {
        return new EffectEvent(EventType.ZONE_CHANGE)
                .subject(card).param(EventParam.FROM_ZONE, "hand")
                .param(EventParam.TO_ZONE, "library").param(EventParam.LIBRARY_POSITION, position);
    }

    private static EffectEvent zoneChange() {
        return new EffectEvent(EventType.ZONE_CHANGE)
                .subject("E7").param(EventParam.FROM_ZONE, "hand")
                .param(EventParam.TO_ZONE, "battlefield");
    }

    private static EffectEvent oneDamage() {
        return new EffectEvent(EventType.DAMAGE_DEALT)
                .subject("E7").param(EventParam.AMOUNT, 1).param(EventParam.COMBAT, true);
    }
}
