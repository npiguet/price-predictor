package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Unit tests for the seat table's parsing, pod grouping and pair drawing. */
class SeatTableIndexTest {

    private static SeatTableIndex.SeatEntry seat(String pod, String label) {
        return new SeatTableIndex.SeatEntry(
                pod, "BLB", label, SeatTableIndex.SeatKind.DECK, List.of("Forest", "Island"));
    }

    // ── Line parsing ────────────────────────────────────────────────

    @TempDir
    Path tempDir;

    /** A seat table file holding {@code lines}. */
    private Path table(String... lines) throws IOException {
        Path table = tempDir.resolve("seats.txt");
        Files.writeString(table, String.join("\n", lines) + "\n");
        return table;
    }

    /**
     * The seat {@code line} loads as, read the way the worker reads it: from a
     * table, beside a second seat of the same pod labelled apart from it, so
     * that the pod's one pairing holds it.
     */
    private SeatTableIndex.SeatEntry loaded(String line) throws IOException {
        SeatTableIndex.Pairing pairing = SeatTableIndex.load(
                        table(line, "d1;BLB;other;deck;Forest"), false)
                .randomPairing().orElseThrow();
        return pairing.a().label().equals("other") ? pairing.b() : pairing.a();
    }

    @Test
    void parsesTheFiveFields() throws IOException {
        SeatTableIndex.SeatEntry e = loaded("d1;BLB;gen4;deck;A|B|C");

        assertEquals("d1", e.draftId());
        assertEquals("BLB", e.setCode());
        assertEquals("gen4", e.label());
        assertEquals(SeatTableIndex.SeatKind.DECK, e.kind());
        assertEquals(List.of("A", "B", "C"), e.cardNames());
    }

    @Test
    void cardNamesMayContainCommasAndApostrophes() throws IOException {
        SeatTableIndex.SeatEntry e = loaded("d1;BLB;gen4;deck;Ach! Hans, Run!|Yawgmoth's Will");

        assertEquals(List.of("Ach! Hans, Run!", "Yawgmoth's Will"), e.cardNames());
    }

    @Test
    void poolKindIsRecognised() throws IOException {
        assertTrue(loaded("d1;BLB;forge-native;pool;A|B").isPool());
    }

    @Test
    void deckKindIsNotAPool() throws IOException {
        assertFalse(loaded("d1;BLB;gen4;deck;A").isPool());
    }

    @Test
    void anUnknownKindIsRejectedRatherThanPlayedAsADeck() throws IOException {
        Path table = table("d1;BLB;gen4;dek;A|B", "d1;BLB;other;deck;Forest");
        assertThrows(IllegalArgumentException.class, () -> SeatTableIndex.load(table, false));
    }

    @Test
    void aShortLineIsRejected() throws IOException {
        Path table = table("d1;BLB;gen4;deck");
        assertThrows(IllegalArgumentException.class, () -> SeatTableIndex.load(table, false));
    }

    @Test
    void blankLinesAreNotSeats() throws IOException {
        SeatTableIndex index = SeatTableIndex.load(
                table("d1;BLB;gen4;deck;A", "", "   ", "d1;BLB;gen1;deck;B"), false);

        assertEquals(2, index.seatCount());
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
        assertTrue(index.randomPairing().isEmpty());
    }

    @Test
    void aSingleLabelPodIsUsableWhenMirrorsAreIncluded() {
        SeatTableIndex index = new SeatTableIndex(
                List.of(seat("d1", "gen4"), seat("d1", "gen4")), true);

        assertEquals(1, index.eligiblePodCount());
        assertTrue(index.randomPairing().isPresent());
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
            SeatTableIndex.Pairing p = index.randomPairing().orElseThrow();
            assertEquals(p.a().draftId(), p.b().draftId());
        }
    }

    /**
     * Mirrors allowed, so the label check cannot reject a seat drawn twice:
     * only the distinct-index draw keeps a seat from playing itself.
     */
    @Test
    void bothSeatsAreDistinct() {
        SeatTableIndex index = new SeatTableIndex(
                List.of(seat("d1", "gen4"), seat("d1", "gen1"), seat("d1", "forge-full")),
                true,
                new Random(11));

        for (int i = 0; i < 200; i++) {
            SeatTableIndex.Pairing p = index.randomPairing().orElseThrow();
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
            SeatTableIndex.Pairing p = index.randomPairing().orElseThrow();
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
            SeatTableIndex.Pairing p = index.randomPairing().orElseThrow();
            sawMirror = p.a().label().equals(p.b().label());
        }
        assertTrue(sawMirror, "a mirror pair should be reachable when included");
    }

    @Test
    void anEmptyTableYieldsNoPairing() {
        assertTrue(new SeatTableIndex(List.of(), false).randomPairing().isEmpty());
    }
}
