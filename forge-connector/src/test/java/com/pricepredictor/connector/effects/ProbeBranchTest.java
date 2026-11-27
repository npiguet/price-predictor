package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a damage-step probe's branch holds, against the real combat record it
 * mirrors: a started game played through its damage steps by the real phase
 * handler, recorded by a {@link GameRecording} with every hook installed.
 */
@ExtendWith(ForgeExtension.class)
class ProbeBranchTest {

    @TempDir
    Path tempDir;

    /** The combat as the declare-blockers step leaves it: {@code attacker} blocked by {@code blocker}. */
    private record Board(Game game, Card attacker, Card blocker, Player defending) {

        static Board of(String attacker, String blocker) {
            Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
            Player you = game.getPlayers().get(0);
            Player opponent = game.getPlayers().get(1);
            Card attacking = TestBoard.inPlay(game, you, attacker);
            Card blocking = TestBoard.inPlay(game, opponent, blocker);
            TestBoard.blockedCombat(new TestBoard.CombatSides(game, you, opponent), attacking, blocking);
            return new Board(game, attacking, blocking, opponent);
        }
    }

    /** The real combat record and the one branch mirroring it, of a combat whose steps were probed. */
    private record Pair(String real, String branch) {
    }

    /**
     * Play {@code board}'s combat to its end, probing {@code keyword}, and
     * read the one real combat record that has a branch, and the branch.
     */
    private Pair probed(Board board, String keyword) throws Throwable {
        CollectionCaps caps = TestCaps.caps().everyDecision().interventions(0).probes(4)
                .probeKeywords(keyword).build();
        RecordShardWriter writer = ShardRun.recordedWithClauseHook(tempDir, new ShardRun.Hooked(board.game(), caps, "to probe a damage step"), () -> board.game().getPhaseHandler().devAdvanceToPhase(PhaseType.COMBAT_END));
        List<String> combats = ShardRun.readShard(writer.path()).stream()
                .filter(line -> line.contains("\"kind\":\"combat\"")).toList();
        List<String> branches = combats.stream()
                .filter(line -> Boolean.TRUE.equals(RecordLines.parse(line).get("fork"))).toList();
        assertEquals(1, branches.size(), combats.toString());
        Object mirrored = RecordLines.parse(branches.get(0)).get("mirror_of");
        String real = combats.stream()
                .filter(line -> mirrored.equals(RecordLines.parse(line).get("record_id")))
                .findFirst().orElseThrow();
        return new Pair(real, branches.get(0));
    }

    private static boolean holds(String record, EventType type, Card subject) {
        return RecordEvents.eventsNaming(record, "E" + subject.getId()).stream()
                .anyMatch(event -> type.wireName().equals(event.get("type")));
    }

    /**
     * A lord dying in combat changes the other creatures it pumped, in the
     * branch as in the real record: the live step's state-based check
     * publishes the stats change, and a branch that did not would read the
     * change as something the stripped keyword did.
     */
    @Test
    void aLordDyingInCombatChangesTheOtherCreaturesInTheBranchAsInTheRealRecord() throws Throwable {
        Board board = Board.of("Colossal Dreadmaw", "Benalish Marshal");
        Card pumped = TestBoard.inPlay(board.game(), board.defending(), "Grizzly Bears");

        Pair pair = probed(board, "trample");

        assertTrue(holds(pair.real(), EventType.PT_CHANGE, pumped), pair.real());
        assertTrue(holds(pair.branch(), EventType.PT_CHANGE, pumped), pair.branch());
    }

    /**
     * A first striker stripped of first strike at the first-strike step
     * deals its damage in the regular step, and the branch plays that step:
     * Youthful Knight and the Grizzly Bears blocking it trade there, where
     * the real first-strike step has the Knight kill the Bears unharmed.
     */
    @Test
    void aStrippedFirstStrikersBranchHoldsItsRegularStepDamage() throws Throwable {
        Board board = Board.of("Youthful Knight", "Grizzly Bears");

        Pair pair = probed(board, "first_strike");

        assertEquals(DamageStep.FIRST_STRIKE.wireName(), combatSubstep(pair.real()), pair.real());
        String knight = "E" + board.attacker().getId();
        assertTrue(RecordEvents.eventsNaming(pair.branch(), "E" + board.blocker().getId()).stream()
                        .anyMatch(event -> EventType.DAMAGE_DEALT.wireName().equals(event.get("type"))
                                && knight.equals(RecordLines.params(event).get("source"))),
                pair.branch());
        assertTrue(holds(pair.branch(), EventType.DAMAGE_DEALT, board.attacker()), pair.branch());
    }

    private static Object combatSubstep(String record) {
        Map<String, Object> state = RecordLines.asMap(RecordLines.parse(record).get("state"));
        return RecordLines.asMap(state.get("global")).get("combat_substep");
    }
}
