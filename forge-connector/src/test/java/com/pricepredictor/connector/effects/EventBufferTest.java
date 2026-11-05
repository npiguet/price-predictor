package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which repeated events {@link EventBuffer} files once and which it files every time. */
class EventBufferTest {

    /**
     * The same transition told twice in one record is filed once.
     *
     * <p>A card is on the battlefield or it is not, so a second identical
     * {@code zone_change} carries nothing the first did not.
     */
    @Test
    void aRepeatedStateTransitionIsFiledOnce() {
        EventBuffer into = new EventBuffer();

        assertTrue(into.file(zoneChange()));
        assertFalse(into.file(zoneChange()));

        assertEquals(1, into.events().size());
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

        assertFalse(shuffledIn.params().containsKey("library_position"));
        assertEquals(-1, elsewhere.params().get("library_position"));
    }

    /** A move after the shuffle is where it was put. */
    @Test
    void aMoveAfterTheShuffleKeepsItsPosition() {
        EventBuffer into = new EventBuffer();
        into.file(new EffectEvent(EventType.LIBRARY_SHUFFLED).subject("P0"));
        EffectEvent onTop = intoLibrary("E7", 0).intoLibraryOf("P0");

        into.file(onTop);

        assertEquals(0, onTop.params().get("library_position"));
    }

    private static EffectEvent intoLibrary(String card, int position) {
        return new EffectEvent(EventType.ZONE_CHANGE)
                .subject(card).param("from_zone", "hand")
                .param("to_zone", "library").param("library_position", position);
    }

    private static EffectEvent zoneChange() {
        return new EffectEvent(EventType.ZONE_CHANGE)
                .subject("E7").param("from_zone", "hand")
                .param("to_zone", "battlefield");
    }

    private static EffectEvent oneDamage() {
        return new EffectEvent(EventType.DAMAGE_DEALT)
                .subject("E7").param("amount", 1).param("combat", true);
    }
}
