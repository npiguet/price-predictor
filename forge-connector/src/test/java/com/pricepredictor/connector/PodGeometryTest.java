package com.pricepredictor.connector;

import forge.gamemodes.limited.DraftPack;
import forge.item.PaperCard;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link PodGeometry} against the rules {@code src/draft/domain/draft_geometry.py}
 * reads a transcript by: the two must agree on every booster and every pick,
 * or the reader attributes picks to the wrong seats.
 */
class PodGeometryTest {

    /** {@code pack_number(k) = k // pod_size + 1}, {@code opening_seat(k) = k % pod_size}. */
    @Test
    void eachBoosterIndexIsOnePackAndOpeningSeat() {
        Set<Integer> seen = new HashSet<>();
        for (int pack = 1; pack <= PodGeometry.PACKS; pack++) {
            for (int seat = 0; seat < PodGeometry.POD_SIZE; seat++) {
                int k = PodGeometry.boosterIndex(pack, seat);
                assertEquals(pack, k / PodGeometry.POD_SIZE + 1);
                assertEquals(seat, k % PodGeometry.POD_SIZE);
                seen.add(k);
            }
        }
        assertEquals(PodGeometry.BOOSTERS, seen.size());
        assertEquals(0, Collections.min(seen));
        assertEquals(PodGeometry.BOOSTERS - 1, Collections.max(seen));
    }

    /** {@code direction(p)}: +1 for packs 1 and 3, -1 for pack 2. */
    @Test
    void oddPacksPassLeftAndEvenPacksRight() {
        assertEquals(1, PodGeometry.passDirection(1));
        assertEquals(-1, PodGeometry.passDirection(2));
        assertEquals(1, PodGeometry.passDirection(3));
    }

    @Test
    void passingWrapsAroundThePod() {
        assertEquals(0, PodGeometry.nextSeat(PodGeometry.POD_SIZE - 1, 1));
        assertEquals(PodGeometry.POD_SIZE - 1, PodGeometry.nextSeat(0, 2));
    }

    /**
     * {@code seat_of_pick(k, j) = (opening_seat + j * dir) mod pod_size}: the
     * seat a booster reaches after {@code j} passes is the one the reader
     * credits with its {@code j}-th pick.
     */
    @Test
    void theSeatHoldingABoosterAfterJPassesIsTheReadersAuthorOfPickJ() {
        int packSize = 15;
        for (int pack = 1; pack <= PodGeometry.PACKS; pack++) {
            int dir = PodGeometry.passDirection(pack);
            for (int opening = 0; opening < PodGeometry.POD_SIZE; opening++) {
                int seat = opening;
                for (int j = 0; j < packSize; j++) {
                    assertEquals(Math.floorMod(opening + j * dir, PodGeometry.POD_SIZE), seat,
                            "pack " + pack + " opened by " + opening + ", pick offset " + j);
                    seat = PodGeometry.nextSeat(seat, pack);
                }
            }
        }
    }

    // ── DraftWorkerMain.boosterGeometryProblem ────────────────────────────────

    private static DraftPack[] boosters(int... sizes) {
        DraftPack[] packs = new DraftPack[sizes.length];
        for (int k = 0; k < sizes.length; k++) {
            packs[k] = new DraftPack(
                    new ArrayList<>(Collections.<PaperCard>nCopies(sizes[k], null)), k);
        }
        return packs;
    }

    @Test
    void boostersOfOneSizeHaveNoProblem() {
        assertEquals(Optional.empty(), DraftWorkerMain.boosterGeometryProblem(boosters(14, 14, 14)));
    }

    @Test
    void boostersOfTwoSizesAreAProblemNamingBoth() {
        String problem = DraftWorkerMain.boosterGeometryProblem(boosters(14, 13, 14)).orElseThrow();
        assertTrue(problem.contains("13") && problem.contains("14"), problem);
    }

    @Test
    void emptyBoostersAreAProblem() {
        assertEquals(Optional.of("empty boosters"),
                DraftWorkerMain.boosterGeometryProblem(boosters(0, 0)));
    }
}
