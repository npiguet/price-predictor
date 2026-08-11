package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Unit tests for the seat table's parsing, pod grouping and pair drawing. */
class SeatTableIndexTest {

    private static SeatTableIndex.SeatEntry seat(String pod, String label) {
        return new SeatTableIndex.SeatEntry(
                pod, "BLB", label, SeatTableIndex.KIND_DECK, List.of("Forest", "Island"));
    }

    // ── Line parsing ────────────────────────────────────────────────

    @Test
    void parsesTheFiveFields() {
        SeatTableIndex.SeatEntry e = SeatTableIndex.parseLine("d1;BLB;gen4;deck;A|B|C");

        assertEquals("d1", e.draftId());
        assertEquals("BLB", e.setCode());
        assertEquals("gen4", e.label());
        assertEquals(SeatTableIndex.KIND_DECK, e.kind());
        assertEquals(List.of("A", "B", "C"), e.cardNames());
    }

    @Test
    void cardNamesMayContainCommasAndApostrophes() {
        SeatTableIndex.SeatEntry e =
                SeatTableIndex.parseLine("d1;BLB;gen4;deck;Ach! Hans, Run!|Yawgmoth's Will");

        assertEquals(List.of("Ach! Hans, Run!", "Yawgmoth's Will"), e.cardNames());
    }

    @Test
    void poolKindIsRecognised() {
        SeatTableIndex.SeatEntry e =
                SeatTableIndex.parseLine("d1;BLB;forge-native;pool;A|B");

        assertTrue(e.isPool());
    }

    @Test
    void deckKindIsNotAPool() {
        assertFalse(SeatTableIndex.parseLine("d1;BLB;gen4;deck;A").isPool());
    }

    @Test
    void aShortLineIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> SeatTableIndex.parseLine("d1;BLB;gen4;deck"));
    }

    // ── Pod grouping ────────────────────────────────────────────────

    @Test
    void seatsAreGroupedByDraftId() {
        SeatTableIndex index = new SeatTableIndex(
                List.of(
                        seat("d1", "gen4"), seat("d2", "gen4"),
                        seat("d1", "gen1"), seat("d2", "forge-full")),
                false);

        assertEquals(2, index.eligiblePodCount());
        assertEquals(4, index.seatCount());
    }

    @Test
    void aSingleSeatPodCannotYieldAPairing() {
        SeatTableIndex index = new SeatTableIndex(List.of(seat("d1", "gen4")), false);
        assertEquals(0, index.eligiblePodCount());
    }

    @Test
    void aSingleLabelPodIsExcludedWhenMirrorsAre() {
        SeatTableIndex index = new SeatTableIndex(
                List.of(seat("d1", "gen4"), seat("d1", "gen4")), false);

        assertEquals(0, index.eligiblePodCount());
        assertNull(index.randomPairing(false));
    }

    @Test
    void aSingleLabelPodIsUsableWhenMirrorsAreIncluded() {
        SeatTableIndex index = new SeatTableIndex(
                List.of(seat("d1", "gen4"), seat("d1", "gen4")), true);

        assertEquals(1, index.eligiblePodCount());
        assertNotNull(index.randomPairing(true));
    }

    // ── Pair drawing ────────────────────────────────────────────────

    @Test
    void pairsNeverSpanPods() {
        SeatTableIndex index = new SeatTableIndex(
                List.of(
                        seat("d1", "gen4"), seat("d1", "gen1"),
                        seat("d2", "gen4"), seat("d2", "forge-full")),
                false,
                new Random(7));

        for (int i = 0; i < 200; i++) {
            SeatTableIndex.Pairing p = index.randomPairing(false);
            assertEquals(p.a().draftId(), p.b().draftId());
        }
    }

    @Test
    void bothSeatsAreDistinct() {
        SeatTableIndex index = new SeatTableIndex(
                List.of(seat("d1", "gen4"), seat("d1", "gen1"), seat("d1", "forge-full")),
                false,
                new Random(11));

        for (int i = 0; i < 200; i++) {
            SeatTableIndex.Pairing p = index.randomPairing(false);
            assertFalse(p.a() == p.b());
        }
    }

    @Test
    void mirrorPairsAreRejectedWhenExcluded() {
        // Two gen4 seats and one gen1: a mirror is drawable but must never be returned.
        SeatTableIndex index = new SeatTableIndex(
                List.of(seat("d1", "gen4"), seat("d1", "gen4"), seat("d1", "gen1")),
                false,
                new Random(3));

        for (int i = 0; i < 300; i++) {
            SeatTableIndex.Pairing p = index.randomPairing(false);
            assertFalse(p.a().label().equals(p.b().label()));
        }
    }

    @Test
    void mirrorPairsAppearWhenIncluded() {
        SeatTableIndex index = new SeatTableIndex(
                List.of(seat("d1", "gen4"), seat("d1", "gen4"), seat("d1", "gen1")),
                true,
                new Random(5));

        boolean sawMirror = false;
        for (int i = 0; i < 300 && !sawMirror; i++) {
            SeatTableIndex.Pairing p = index.randomPairing(true);
            sawMirror = p.a().label().equals(p.b().label());
        }
        assertTrue(sawMirror, "a mirror pair should be reachable when included");
    }

    @Test
    void anEmptyTableYieldsNoPairing() {
        assertNull(new SeatTableIndex(List.of(), false).randomPairing(false));
    }
}
