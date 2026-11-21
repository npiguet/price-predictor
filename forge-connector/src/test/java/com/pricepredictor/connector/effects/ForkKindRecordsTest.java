package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A record says which kind of fork wrote it ({@code fork_kind}), which a
 * live record does not carry; and a fork's record holding the game's outcome is the fork's
 * outcome, not the game's, so it does not pass the per-game cap.
 *
 * <p>The protection and directed kinds are checked where their forks'
 * records are read ({@code ProtectionForkRecordsTest},
 * {@code ForkPlayForwardTest}).
 */
@ExtendWith(ForgeExtension.class)
class ForkKindRecordsTest {

    private static final String PLAYER_LOST = "\"type\":\"player_lost\"";

    @TempDir
    Path tempDir;

    /**
     * A probe branch whose fork ends the game: Child of Night's lifelink
     * stripped, its controller's life gain no longer offsets the unblocked
     * Colossal Dreadmaw, and the branch holds {@code player_lost} while the
     * real combat record, where the opponent lives on at 2, holds none. The
     * branch says {@code probe}, the real record carries no {@code fork_kind}.
     */
    @Test
    void aProbeBranchSaysProbeAndTheLiveRecordNothing() throws Throwable {
        RecordShardWriter writer = ShardRun.writer(tempDir);
        lifelinkCombat(writer);
        List<String> records = ShardRun.readShard(writer.path());

        assertEquals(2, records.size(), records::toString);
        Map<String, Object> live = parse(records.get(0));
        Map<String, Object> branch = parse(records.get(1));
        assertEquals(Boolean.FALSE, live.get("fork"), records.get(0));
        assertFalse(live.containsKey("fork_kind"), records.get(0));
        assertFalse(records.get(0).contains(PLAYER_LOST), records.get(0));
        assertEquals("probe", branch.get("fork_kind"), records.get(1));
        assertTrue(records.get(1).contains(PLAYER_LOST), records.get(1));
    }

    /**
     * The same combat in a game already at the cap: the branch holding
     * {@code player_lost} is withheld with the real record it mirrors, since
     * only a live record holding the outcome passes the cap.
     */
    @Test
    void aCappedGamesProbeBranchHoldingTheOutcomeIsWithheld() throws Throwable {
        RecordShardWriter writer = new RecordShardWriter(tempDir, new ShardId("run", 0, "l1"), 0);
        int withheld = lifelinkCombat(writer);

        assertEquals(2, withheld, "the real record and its branch are withheld");
        assertFalse(Files.exists(writer.path()), () -> "a record reached the shard: " + readable(writer));
    }

    /** A forced resolution's effect half says {@code intervention}. */
    @Test
    void aForcedResolutionSaysIntervention() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Mountain", board.you);
        board.battlefield("Grizzly Bears", board.opponent);
        SpellAbility pyroclasm = board.spell(board.hand("Pyroclasm", board.you), board.you);
        CollectionCaps caps = TestCaps.caps().everyDecision().interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forked(tempDir, board.game, caps, 1L, pyroclasm);

        assertTrue(fork.wrote(), "the fork observed nothing");
        for (String record : fork.records()) {
            assertEquals("intervention", parse(record).get("fork_kind"), record);
        }
    }

    /**
     * Hill Giant blocked by Child of Night and Colossal Dreadmaw unblocked,
     * the opponent at 6, the regular damage step dealt live with a lifelink
     * probe taken ahead of it; the records the game had withheld.
     */
    private static int lifelinkCombat(RecordShardWriter writer) throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DAMAGE);
        Card giant = board.battlefield("Hill Giant", board.you);
        Card dreadmaw = board.battlefield("Colossal Dreadmaw", board.you);
        Card child = board.battlefield("Child of Night", board.opponent);
        board.opponent.setLife(6, null);
        Combat combat = TestBoard.blockedCombat(new TestBoard.CombatSides(board.game, board.you, board.opponent), giant, child, dreadmaw);
        CollectionCaps caps = TestCaps.caps().everyDecision()
                .interventions(0).probes(1).probeKeywords("lifelink").build();
        try {
            ProbeScheduler probing = new ProbeScheduler(board.game, ShardRun.GAME_ID, caps,
                    new DamageStepProbe(board.game, TestCollectors.context(writer, ShardRun.GAME_ID),
                            new ForkBudget(caps)));
            board.game.subscribeToEvents(new BusBracketCollector(
                    board.game, TestCollectors.context(writer, ShardRun.GAME_ID), probing::writeHeldProbes));
            probing.collect();
            assertTrue(combat.assignCombatDamage(false), "the step assigned no damage");
            combat.dealAssignedDamage();
            board.game.getAction().checkStateEffects(false);
            TestBoard.enterStep(board.game, PhaseType.COMBAT_END);
            assertEquals(2, board.opponent.getLife(), "the opponent lives on live");
            return writer.withheld(ShardRun.GAME_ID);
        } finally {
            writer.close();
        }
    }

    /** What the writer's shard holds, for a failure message. */
    private static String readable(RecordShardWriter writer) {
        try {
            return ShardRun.readShard(writer.path()).toString();
        } catch (java.io.IOException e) {
            return e.toString();
        }
    }
}
