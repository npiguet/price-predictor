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

    private static EffectEvent zoneChange() {
        return new EffectEvent(EffectEvent.ZONE_CHANGE)
                .subject("E7").param("from_zone", "hand")
                .param("to_zone", "battlefield");
    }

    private static EffectEvent oneDamage() {
        return new EffectEvent(EffectEvent.DAMAGE_DEALT)
                .subject("E7").param("amount", 1).param("combat", true);
    }
}
