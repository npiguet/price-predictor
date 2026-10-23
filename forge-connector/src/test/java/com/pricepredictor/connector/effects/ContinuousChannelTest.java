package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which continuous records coalesce: per static and stable board, and per rendering. */
@ExtendWith(ForgeExtension.class)
class ContinuousChannelTest {

    @TempDir
    Path tempDir;

    /** A channel over no game, enough for what it decides before anything is written. */
    private static ContinuousChannel channel() {
        return new ContinuousChannel(
                new LiveGame(null), TestCollectors.context(null, "game-1"), new RecordDedup());
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

    /**
     * Two copies of one anthem render one record.
     *
     * <p>The per-board coalescer cannot see this: the two statics have different
     * ids, so it treats them as different questions, while their key, their
     * contributions and their suppressed board are all the same.
     */
    @Test
    void twoStaticsThatRenderIdenticallyAreWrittenOnce() throws Throwable {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Glorious Anthem", board.you);
        board.battlefield("Glorious Anthem", board.you);
        board.battlefield("Grizzly Bears", board.you);
        board.game.getAction().checkStaticAbilities();

        List<String> records = ShardRun.records(tempDir, board.game,
                (bracket, collectors) -> collectors.continuous().collect());

        assertEquals(1, records.size(), String.valueOf(records));
    }
}
