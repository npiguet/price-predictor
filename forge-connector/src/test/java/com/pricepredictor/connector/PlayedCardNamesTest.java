package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A played card is recorded under the name its deck lists, whichever face was
 * cast: the cards-played columns are sets of the deck's own card names.
 */
@ExtendWith(ForgeExtension.class)
class PlayedCardNamesTest {

    @Test
    void aRoomDoorIsRecordedUnderTheFullRoomName() {
        String full = "Painter's Studio // Defaced Gallery";
        assertEquals(full, PlayedCardCollector.canonicalName("Painter's Studio"));
        assertEquals(full, PlayedCardCollector.canonicalName("Defaced Gallery"));
        assertEquals(full, PlayedCardCollector.canonicalName(full));
    }

    @Test
    void aSplitCardHalfIsRecordedUnderTheFullName() {
        assertEquals("Fire // Ice", PlayedCardCollector.canonicalName("Fire"));
        assertEquals("Fire // Ice", PlayedCardCollector.canonicalName("Ice"));
    }

    @Test
    void anAdventureIsRecordedUnderItsCreature() {
        assertEquals("Bonecrusher Giant", PlayedCardCollector.canonicalName("Stomp"));
    }

    @Test
    void aBackFaceIsRecordedUnderTheDeckListedName() {
        // Transform and modal cards: the deck lists the front face.
        assertEquals("Delver of Secrets", PlayedCardCollector.canonicalName("Insectile Aberration"));
        assertEquals("Valki, God of Lies", PlayedCardCollector.canonicalName("Tibalt, Cosmic Impostor"));
    }

    @Test
    void anOrdinaryCardAndAnUnknownNameAreUnchanged() {
        assertEquals("Lightning Bolt", PlayedCardCollector.canonicalName("Lightning Bolt"));
        assertEquals("Not A Card Name", PlayedCardCollector.canonicalName("Not A Card Name"));
    }
}
