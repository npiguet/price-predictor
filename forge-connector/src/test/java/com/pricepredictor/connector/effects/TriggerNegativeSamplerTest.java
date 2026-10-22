package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which trigger evaluations are offered to the writer, and how many one line may write. */
class TriggerNegativeSamplerTest {

    /** A game's sampler, drawing from the stream a game seeded with 42 would. */
    private static TriggerNegativeSampler negatives() {
        return new TriggerNegativeSampler(new Random(GameRecording.scramble(42L)));
    }

    // ── the trigger negative sample and the per-line cap ───────────────

    /**
     * Drive one mode's evaluations and count what the writer was offered.
     *
     * @param burst how many non-fired evaluations follow each firing
     */
    private static int[] driveTrigger(
            TriggerNegativeSampler negatives, String mode, int burst, int firings) {
        int[] kept = new int[2];
        for (int firing = 0; firing < firings; firing++) {
            if (negatives.offer(mode, true)) {
                negatives.keep(mode, true);
                kept[0]++;
            }
            for (int i = 0; i < burst; i++) {
                if (negatives.offer(mode, false)) {
                    negatives.keep(mode, false);
                    kept[1]++;
                }
            }
        }
        return kept;
    }

    /**
     * The kept negatives track the kept positives, whatever the population is.
     *
     * <p>The first corpus used a fixed 2% rate and landed at 74:26 rather than
     * 1:1, because a fixed rate cannot hold a ratio: the true population was
     * 1:17.7 and it varies by mode, board and turn. Two modes with very
     * different burst lengths are driven here, and both have to come out level
     * — a fixed rate fails this by construction, at 1:0.8 for the short burst
     * and 1:0.06 for the long one.
     */
    @Test
    void negativesAreKeptAtAboutOneToOneWhateverTheBurstLength() {
        TriggerNegativeSampler negatives = negatives();
        int[] rare = driveTrigger(negatives, "ChangesZone", 40, 200);
        int[] common = driveTrigger(negatives, "TapsForMana", 3, 200);

        for (int[] kept : List.of(rare, common)) {
            assertTrue(kept[0] > 0 && kept[1] > 0, "nothing was kept");
            double ratio = (double) kept[1] / kept[0];
            assertTrue(Math.abs(ratio - 1.0) < 0.1,
                    "kept " + kept[0] + " fired against " + kept[1]
                            + " not fired, a ratio of " + ratio);
        }
    }

    /**
     * The ratio holds over what reaches the shard, not over what was offered.
     *
     * <p>This is the 58:42 the corpus measured while the unit test above read
     * 1:1. The difference is here: an offered negative is refused downstream by
     * the duplicate check or the per-line flood cap far more often than a
     * positive is, and under the old rate the deficit that leaves standing was
     * repaid only at {@code deficit / estimated run length} — a lag a game's
     * handful of firings per mode never works off. Three refusals in four is
     * harsher than the corpus and the kept sets still come out level.
     */
    @Test
    void negativesStayLevelEvenWhenMostOfThemAreRefusedDownstream() {
        TriggerNegativeSampler negatives = negatives();
        java.util.Random refusals = new java.util.Random(7);
        int[] kept = new int[2];
        for (int firing = 0; firing < 400; firing++) {
            if (negatives.offer("ChangesZone", true)) {
                negatives.keep("ChangesZone", true);
                kept[0]++;
            }
            for (int i = 0; i < 20; i++) {
                if (!negatives.offer("ChangesZone", false)) {
                    continue;
                }
                // The record was built and then refused: nothing is kept, and
                // the deficit it was drawn against is still owed.
                if (refusals.nextInt(4) != 0) {
                    continue;
                }
                negatives.keep("ChangesZone", false);
                kept[1]++;
            }
        }
        double ratio = (double) kept[1] / kept[0];
        assertTrue(Math.abs(ratio - 1.0) < 0.1,
                "kept " + kept[0] + " fired against " + kept[1]
                        + " not fired, a ratio of " + ratio);
    }

    /**
     * A deficit that cannot be filled costs a few offers, not all of them.
     *
     * <p>The price of offering on the deficit alone: a mode whose every acting
     * line has hit its per-line cap owes a negative for ever, and each offer
     * builds a snapshot on the hook that fires most often in the collector.
     * The backoff is what keeps that bounded.
     */
    @Test
    void aDeficitThatCanNeverBeFilledStopsSpendingWorkOnItself() {
        TriggerNegativeSampler negatives = negatives();
        negatives.offer("ChangesZone", true);
        negatives.keep("ChangesZone", true);

        int offered = 0;
        for (int i = 0; i < 20_000; i++) {
            if (negatives.offer("ChangesZone", false)) {
                offered++;   // built, refused downstream, never kept
            }
        }
        assertTrue(offered > 0, "the standing deficit is never abandoned");
        assertTrue(offered < 100,
                "20,000 unfillable evaluations built " + offered + " records");
    }

    /** A mode's balance is its own, so a busy mode cannot starve a quiet one. */
    @Test
    void oneModeDoesNotSpendAnothersBudget() {
        TriggerNegativeSampler negatives = negatives();
        driveTrigger(negatives, "ChangesZone", 40, 200);
        int[] quiet = driveTrigger(negatives, "Untaps", 2, 5);

        assertTrue(quiet[1] > 0,
                "a quiet mode kept no negatives after a busy one ran");
    }

    /** A fired evaluation is always offered; only negatives are sampled. */
    @Test
    void everyFiredEvaluationIsOffered() {
        TriggerNegativeSampler negatives = negatives();
        for (int i = 0; i < 50; i++) {
            assertTrue(negatives.offer("ChangesZone", true));
        }
    }

    /** And a mode that never fires contributes no negatives, by construction. */
    @Test
    void aModeThatNeverFiresKeepsNoNegatives() {
        TriggerNegativeSampler negatives = negatives();
        for (int i = 0; i < 500; i++) {
            assertFalse(negatives.offer("Untaps", false));
        }
    }

    /** One acting line is worth a couple of dozen looks in a game, not hundreds. */
    @Test
    void oneTriggerLineIsCappedPerGameAndPerVerdict() {
        TriggerNegativeSampler negatives = negatives();
        int written = 0;
        for (int i = 0; i < 100; i++) {
            if (negatives.allowLine("[key]", true)) {
                written++;
            }
        }
        assertEquals(24, written);
        // The other verdict has its own budget: "fired" and "did not fire" are
        // different observations of the same line.
        assertTrue(negatives.allowLine("[key]", false));
    }
}
