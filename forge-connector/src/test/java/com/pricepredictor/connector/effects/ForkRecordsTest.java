package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.ForkTestCaps.forkCaps;
import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordPicks.records;
import static com.pricepredictor.connector.effects.RecordText.payloadOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The records of the turn-conditioned forced lines and of the protection forks.
 */
@ExtendWith(ForgeExtension.class)
class ForkRecordsTest {

    @TempDir
    Path tempDir;

    // ── a turn-conditioned line forced on the other player's turn (FR-057) ──

    /**
     * Zealous Display untaps its caster's creatures only when it is not the
     * caster's turn. Held by the player whose turn it is not, the scheduler
     * forces it at this boundary of the other player's turn, for its holder,
     * and the record shows the branch: the holder's tapped bears untap.
     */
    @Test
    void aTurnConditionedLineIsForcedForItsHolderOnTheOtherPlayersTurn() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, holder, "Grizzly Bears");
        bears.tap(true, null, null);
        TestBoard.inHand(game, holder, "Zealous Display");

        List<String> forced = scheduled(game, forkCaps().interventions(2).probes(0).build(), 1);

        assertEquals(1, forced.size(), forced.toString());
        assertEquals(Refs.of(holder), parse(forced.get(0)).get("actor_player"), forced.get(0));
        assertEquals(Boolean.TRUE, parse(forced.get(0)).get("interventional"), forced.get(0));
        assertTrue(eventTypes(forced.get(0)).contains(EventType.UNTAPPED.wireName()),
                payloadOf(forced.get(0)));
    }

    /**
     * The same line is forced again once its holder's own turn comes, where
     * it takes the other branch: the off-turn forcing and the on-turn one are
     * kept apart, so one game can show both.
     */
    @Test
    void theSameLineIsForcedAgainOnItsHoldersOwnTurnTakingTheOtherBranch() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, holder, "Grizzly Bears");
        bears.tap(true, null, null);
        TestBoard.inHand(game, holder, "Zealous Display");

        List<String> forced = scheduled(game, forkCaps().interventions(2).probes(0).build(), 1,
                () -> game.getPhaseHandler().devModeSet(PhaseType.MAIN1, holder));

        assertEquals(2, forced.size(), forced.toString());
        assertTrue(eventTypes(forced.get(0)).contains(EventType.UNTAPPED.wireName()),
                payloadOf(forced.get(0)));
        assertFalse(eventTypes(forced.get(1)).contains(EventType.UNTAPPED.wireName()),
                payloadOf(forced.get(1)));
    }

    /** A line whose text does not turn on whose turn it is is not forced off its holder's turn. */
    @Test
    void aLineWithNoTurnConditionIsNotForcedOnTheOtherPlayersTurn() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        TestBoard.inHand(game, holder, "Lightning Bolt");

        assertTrue(scheduled(game, forkCaps().interventions(2).probes(0).build(), 1).isEmpty());
    }

    /** Which lines turn on whose turn it is: a clause of the chain naming either parameter. */
    @Test
    void aTurnConditionAnywhereInTheChainMarksTheLine() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        assertTrue(TurnConditions.branchesOnWhoseTurn(
                TestBoard.inHand(game, holder, "Fated Conflagration").getFirstSpellAbility()));
        assertTrue(TurnConditions.branchesOnWhoseTurn(
                TestBoard.inHand(game, holder, "Zealous Display").getFirstSpellAbility()));
        assertFalse(TurnConditions.branchesOnWhoseTurn(
                TestBoard.inHand(game, holder, "Lightning Bolt").getFirstSpellAbility()));
    }

    /**
     * The scheduler's records over {@code boundaries} boundaries of {@code
     * game}, {@code between} run after each but the last, read from the shard.
     */
    private List<String> scheduled(Game game, CollectionCaps caps, int boundaries, Runnable... between)
            throws IOException {
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording live = new GameRecording(
                GameRecordingRequest.of(game, shard, ShardRun.GAME_ID, caps)
                        .seeded(1L))) {
            InterventionScheduler scheduler = new InterventionScheduler(game, new InterventionFork(
                    game, live, new ForkBudget(caps), 1L));
            scheduler.collect();
            for (Runnable next : between) {
                next.run();
                scheduler.collect();
            }
        } finally {
            shard.close();
        }
        return Files.exists(shard.path()) ? ShardRun.readShard(shard.path()) : List.of();
    }

    // ── protection forks (FR-058) ───────────────────────────────────────

    /**
     * A protection fork grants a creature a rare protection through the
     * {@code keyword} operation and plays the copy to the end of the turn:
     * every record it writes is an interventional fork joined to the live
     * game, the board shows the granted line, and the grant is no event of
     * the turn. The fork's decisions are recorded too, as only a fork played
     * on has any.
     */
    @Test
    void aProtectionForkGrantsTheQualityAndPlaysTheTurnOn() throws Throwable {
        Game game = ForkPlayForwardTest.board();
        Player active = game.getPlayers().get(0);
        Card bears = game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        TestBoard.inPlay(game, game.getPlayers().get(1), "Goblin Piker");
        CollectionCaps caps = forkCaps().interventions(0).probes(0).everyDecision().build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        boolean kept;
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, shard, ShardRun.GAME_ID, caps)
                        .seeded(7L))) {
            recording.open();
            kept = new ProtectionFork(game, new ForkBudget(caps), recording.playForward(), 7L)
                    .take(new ProtectionFork.Grant(bears, "Protection:Goblin"));
        } finally {
            shard.close();
        }
        List<String> forked = ShardRun.readShard(shard.path());

        assertTrue(kept, "the fork was kept");
        assertEquals(PhaseType.MAIN1, game.getPhaseHandler().getPhase(), "the live game did not move");
        assertTrue(forked.stream().anyMatch(record -> "playability".equals(parse(record).get("kind"))),
                "the fork played on, its decisions recorded");
        assertTrue(forked.stream().anyMatch(record -> record.contains("\"fragment\":\"Protection:Goblin\"")),
                "a fork record shows the granted protection");
        String grantedToBears = "{\"type\":\"keyword_change\",\"subjects\":[\"" + Refs.of(bears) + "\"]";
        for (String record : forked) {
            Map<String, Object> parsed = parse(record);
            Object id = parsed.get("record_id");
            assertEquals(Boolean.TRUE, parsed.get("fork"), () -> "fork on " + id);
            assertEquals(Boolean.TRUE, parsed.get("interventional"), () -> "interventional on " + id);
            assertNull(parsed.get("directed"), () -> "directed on " + id);
            assertEquals(ShardRun.GAME_ID, parsed.get("game_id"), () -> "game_id on " + id);
            assertFalse(payloadOf(record).contains(grantedToBears), () -> "the grant is an event of " + id);
        }
        assertTrue(active.getCardsIn(ZoneType.Battlefield).contains(bears));
        assertTrue(bears.getKeywords().stream().noneMatch(k -> "Protection:Goblin".equals(k.getOriginal())),
                "the live creature gained nothing");
    }

    /**
     * The qualities a grant draws from are the rare ones of what the
     * creature can meet: the opponent's non-land permanents and hand, each
     * quality once, spelled as rule_outcomes.ProtectionQuality.fragment spells
     * them, then protection from everything. No colour, and nothing of the
     * opponent's lands or of the creature's own side.
     */
    @Test
    void aGrantDrawsFromTheRareQualitiesOfWhatTheCreatureCanMeet() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player own = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, own, "Grizzly Bears");
        TestBoard.inPlay(game, own, "Serra Angel");
        TestBoard.inPlay(game, opponent, "Goblin Piker");
        TestBoard.inPlay(game, opponent, "Forest");
        TestBoard.inHand(game, opponent, "Lightning Bolt");

        assertEquals(List.of(
                "Protection from each color",
                "Protection:Card.MonoColor:monocolored",
                "Protection:Card.cmcLE3:mana value 3 or less",
                "Protection:Creature",
                "Protection:Goblin",
                "Protection:Instant",
                "Protection:Warrior",
                "Protection from everything"), ProtectionFork.qualitiesFor(bears));
    }

    /**
     * One protection fork a game, only on a run taking interventions, and
     * only as a first main phase begins; a boundary that takes none spends
     * nothing.
     */
    @Test
    void aProtectionForkIsTakenOnceAGameOnARunTakingInterventions() {
        assertFalse(new ForkBudget(forkCaps().interventions(0).probes(2).build()).mayProtect());
        ForkBudget budget = new ForkBudget(forkCaps().interventions(2).probes(0).build());
        assertTrue(budget.mayProtect());

        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        assertFalse(new ProtectionFork(game, budget, null, 1L).collect(PhaseType.UPKEEP));
        assertTrue(budget.mayProtect(), "a boundary that took no fork spent nothing");

        budget.spendProtection();
        assertFalse(budget.mayProtect());
    }
}
