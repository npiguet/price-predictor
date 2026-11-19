package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.SeedStreams.Consumer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The independent streams a game's seed yields: no two consumers of one game
 * share draws (a match's first game is seeded with the match's start time,
 * which also seeded the seat draw), no two games' first forks replay one
 * another, and the same seed always yields the same streams.
 */
class SeedStreamsTest {

    private static final long FIRST_GAME = 1_763_500_000_123L;
    private static final long NEXT_GAME = FIRST_GAME + 1;

    private static List<Long> firstDraws(Random random, int count) {
        List<Long> draws = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            draws.add(random.nextLong());
        }
        return draws;
    }

    @Test
    void theConsumersOfAMatchsFirstGameDrawDifferentSequences() {
        List<List<Long>> sequences = new ArrayList<>();
        for (Consumer consumer : Consumer.values()) {
            sequences.add(firstDraws(SeedStreams.random(FIRST_GAME, consumer), 4));
        }
        for (int i = 0; i < sequences.size(); i++) {
            for (int j = i + 1; j < sequences.size(); j++) {
                assertNotEquals(sequences.get(i).get(0), sequences.get(j).get(0),
                        Consumer.values()[i] + " and " + Consumer.values()[j] + " draw alike first");
                assertNotEquals(sequences.get(i), sequences.get(j));
            }
        }
    }

    @Test
    void twoWorkersMatchesStartingInTheSameMillisecondDrawDifferentStreams() {
        long a = SeedStreams.matchSeed(FIRST_GAME, "3-k9x2abcd");
        long b = SeedStreams.matchSeed(FIRST_GAME, "4-k9x2wxyz");
        assertNotEquals(a, b);
        for (Consumer consumer : Consumer.values()) {
            assertNotEquals(SeedStreams.random(a, consumer).nextLong(),
                    SeedStreams.random(b, consumer).nextLong(), consumer.name());
        }
        SeedStreams.GameStreams first = new SeedStreams.GameStreams(a);
        SeedStreams.GameStreams second = new SeedStreams.GameStreams(b);
        assertNotEquals(first.nextFork().nextLong(), second.nextFork().nextLong());
        assertEquals(a, SeedStreams.matchSeed(FIRST_GAME, "3-k9x2abcd"), "a token seeds one way");
    }

    @Test
    void theSameSeedYieldsTheSameStreams() {
        for (Consumer consumer : Consumer.values()) {
            assertEquals(
                    firstDraws(SeedStreams.random(FIRST_GAME, consumer), 8),
                    firstDraws(SeedStreams.random(FIRST_GAME, consumer), 8), consumer.name());
        }
        SeedStreams.GameStreams again = new SeedStreams.GameStreams(FIRST_GAME);
        SeedStreams.GameStreams once = new SeedStreams.GameStreams(FIRST_GAME);
        assertEquals(firstDraws(once.nextFork(), 8), firstDraws(again.nextFork(), 8));
        assertEquals(firstDraws(once.nextFork(), 8), firstDraws(again.nextFork(), 8));
    }

    @Test
    void aConsumersStreamDoesNotDependOnWhichOthersWereBuiltFirst() {
        long[] forwards = new long[Consumer.values().length];
        for (Consumer consumer : Consumer.values()) {
            forwards[consumer.ordinal()] = SeedStreams.seed(FIRST_GAME, consumer);
        }
        Consumer[] reversed = Consumer.values().clone();
        Collections.reverse(Arrays.asList(reversed));
        for (Consumer consumer : reversed) {
            assertEquals(forwards[consumer.ordinal()], SeedStreams.seed(FIRST_GAME, consumer));
        }
    }

    @Test
    void aConsumerDrawsDifferentlyInTwoGames() {
        for (Consumer consumer : Consumer.values()) {
            assertNotEquals(
                    firstDraws(SeedStreams.random(FIRST_GAME, consumer), 4),
                    firstDraws(SeedStreams.random(NEXT_GAME, consumer), 4), consumer.name());
        }
    }

    @Test
    void theFirstForkOfTwoGamesDrawsDifferentSequences() {
        SeedStreams.GameStreams first = new SeedStreams.GameStreams(FIRST_GAME);
        SeedStreams.GameStreams next = new SeedStreams.GameStreams(NEXT_GAME);
        assertNotEquals(firstDraws(first.nextFork(), 4), firstDraws(next.nextFork(), 4));
    }

    @Test
    void aGamesForksDrawDifferentSequencesFromOneAnother() {
        SeedStreams.GameStreams game = new SeedStreams.GameStreams(FIRST_GAME);
        List<List<Long>> forks = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            forks.add(firstDraws(game.nextFork(), 4));
        }
        assertEquals(forks.size(), forks.stream().distinct().count());
    }

    @Test
    void theNthForkOfTwoWorkerLifetimesDiffers() {
        // A lifetime's games are seeded from its own start time, so its n-th fork
        // of whichever game is drawn from that game's seed and its own count.
        long lifetimeA = FIRST_GAME;
        long lifetimeB = FIRST_GAME + 86_400_000L;
        for (int n = 0; n < 5; n++) {
            SeedStreams.GameStreams a = new SeedStreams.GameStreams(lifetimeA);
            SeedStreams.GameStreams b = new SeedStreams.GameStreams(lifetimeB);
            Random nthOfA = null;
            Random nthOfB = null;
            for (int i = 0; i <= n; i++) {
                nthOfA = a.nextFork();
                nthOfB = b.nextFork();
            }
            assertTrue(nthOfA.nextLong() != nthOfB.nextLong(), "fork " + n);
        }
    }
}
