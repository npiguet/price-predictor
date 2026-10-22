package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which continuous records coalesce: per static and stable board, and per rendering. */
class ContinuousChannelTest {

    /** A channel over no game, enough for what it decides before anything is written. */
    private static ContinuousChannel channel() {
        return channel(new RecordDedup());
    }

    private static ContinuousChannel channel(RecordDedup dedup) {
        return new ContinuousChannel(
                new LiveGame(null), TestCollectors.context(null, "game-1"), dedup);
    }

    // ── continuous coalescing ───────────────────────────────────────────

    /**
     * A static's record is keyed by the static and the board: one repeating
     * both is refused, and a change in either is a record of its own.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "a static is recorded once per stable board, anthem, board-1, false",
            "a changed board gets a new record, anthem, board-2, true",
            "two statics on one board each get a record, crusade, board-1, true"})
    void aContinuousRecordIsKeyedByItsStaticAndBoard(
            String why, String staticKey, String board, boolean allowed) {
        ContinuousChannel collector = channel();
        assertTrue(collector.allowContinuousRecord("anthem", "board-1"));
        assertEquals(allowed, collector.allowContinuousRecord(staticKey, board), why);
    }

    @Test
    void continuousRecordsTakeNoNumericCap() {
        // Coalescing per stable board is itself the cap (FR-029).
        ContinuousChannel collector = channel();
        for (int board = 0; board < 100; board++) {
            assertTrue(collector.allowContinuousRecord("anthem", "board-" + board));
        }
    }

    /**
     * Two copies of one anthem render one record.
     *
     * <p>The per-board coalescer cannot see this: the two statics have different
     * ids, so it treats them as different questions, while their key, their
     * contributions and their suppressed board are all the same.
     */
    @Test
    void twoStaticsThatRenderIdenticallyAreWrittenOnce() {
        RecordDedup dedup = new RecordDedup();
        ContinuousChannel collector = channel(dedup);
        assertTrue(collector.allowContinuousRecord("11", "board-1"));
        assertTrue(collector.allowContinuousRecord("12", "board-1"));
        assertTrue(dedup.allowDistinct("continuous", "{anthem}", "{board}"));
        assertFalse(dedup.allowDistinct("continuous", "{anthem}", "{board}"));
    }
}
