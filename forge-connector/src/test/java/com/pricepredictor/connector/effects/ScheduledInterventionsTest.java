package com.pricepredictor.connector.effects;

import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

/**
 * The intervention scheduler as a worker runs it: a started game recorded by
 * a {@link GameRecording} with every hook installed, its phase boundaries
 * published on the game's bus, and real forks through {@code GameCopier},
 * each boundary drawing against the worker's gate.
 */
@ExtendWith(ForgeExtension.class)
class ScheduledInterventionsTest {

    /**
     * Boundaries published per stretch, each of a game turn of its own: the
     * worker's gate draws once a turn and spends what is left of the budget
     * one turn at a time past a typical game's length, so a line that can
     * be forced is.
     */
    private static final int BOUNDARIES = 40;

    @TempDir
    Path tempDir;

    /**
     * Publish {@code count} boundaries of the step the game is in, the same
     * player's, each in the next game turn: the step's phase event ({@code
     * devModeSet} publishes it), then its first priority, with the stack
     * empty. The turn number put back after with no priority, so a board
     * read before and after differs only by what a fork did to it.
     */
    private static void boundaries(Game game, int count) {
        steps(game, count, () -> TestPriority.settle(game));
    }

    /** The same boundaries, each a phase event then {@code priority}. */
    private static void steps(Game game, int count, Runnable priority) {
        int turn = game.getPhaseHandler().getTurn();
        for (int i = 0; i < count; i++) {
            game.getPhaseHandler().devModeSet(game.getPhaseHandler().getPhase(),
                    game.getPhaseHandler().getPlayerTurn(), game.getPhaseHandler().getTurn() + 1);
            priority.run();
        }
        forge.game.phase.PhaseHandler phases = game.getPhaseHandler();
        phases.devModeSet(phases.getPhase(), phases.getPlayerTurn(), turn);
    }

    /**
     * The forced resolutions' records the recording wrote: interventional
     * resolution halves with no partner, which leaves out a protection fork's.
     */
    private List<String> forced(Game game, int interventions, ShardRun.Played body) throws Throwable {
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(interventions).probes(0).build();
        RecordShardWriter writer = ShardRun.recordedWithClauseHook(
                tempDir, game, caps, "to force a line as a worker does", body);
        if (!Files.exists(writer.path())) {
            return List.of();
        }
        return ShardRun.readShard(writer.path()).stream()
                .filter(line -> Boolean.TRUE.equals(parse(line).get("interventional"))
                        && "resolution".equals(parse(line).get("kind"))
                        && parse(line).get("link_id") == null)
                .toList();
    }

    /**
     * A removal spell held over an empty board has nothing to resolve
     * against: it spends no fork and is not marked forced, so once a creature
     * arrives it is forced under the budget of one it would otherwise have
     * spent. The live board is the same before and after the fork.
     */
    @Test
    void aSpellWithNoTargetIsForcedOnceATargetArrives() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inHand(game, you, "Doom Blade");
        Map<String, Object>[] boards = new Map[2];

        List<String> records = forced(game, 1, () -> {
            boundaries(game, BOUNDARIES);
            TestBoard.inPlay(game, opponent, "Grizzly Bears");
            game.getTriggerHandler().resetActiveTriggers();
            boards[0] = BoardStates.of(game);
            boundaries(game, BOUNDARIES);
            boards[1] = BoardStates.of(game);
        });

        assertEquals(1, records.size(), records.toString());
        assertTrue(eventTypes(records.get(0)).contains(EventType.DESTROYED.wireName())
                        || eventTypes(records.get(0)).contains(EventType.ZONE_CHANGE.wireName()),
                records.get(0));
        assertEquals(boards[0], boards[1], "the fork reached the live board");
    }

    /**
     * Two lines of one card with the same effect API (Chainflinger's two
     * damage abilities) are two lines, each forced once: its threshold line
     * with seven cards in its holder's graveyard, where it can be activated.
     */
    @Test
    void twoLinesOfOneCardWithOneApiAreForcedApart() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Chainflinger");
        for (int card = 0; card < 7; card++) {
            TestBoard.inGraveyard(game, game.getPlayers().get(0), "Mountain");
        }

        List<String> records = forced(game, 4, () -> boundaries(game, 2 * BOUNDARIES));

        assertEquals(2, records.size(), records.toString());
        assertNotEquals(parse(records.get(0)).get("ability"), parse(records.get(1)).get("ability"),
                records.toString());
    }

    /**
     * The fork's copy of a forced line is that line: Chainflinger's second
     * damage ability, forced, is written under its own key and deals its own
     * two damage, not the first's one (C3 #5: the copier's mapper checks the
     * counterpart is alike, its API and script line, since WP1).
     */
    @Test
    void aForcedLineIsWrittenUnderItsOwnKeyAndDoesWhatItSays() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        forge.game.card.Card flinger = TestBoard.inPlay(game, game.getPlayers().get(0), "Chainflinger");
        forge.game.spellability.SpellAbility second = flinger.getSpellAbilities().stream()
                .filter(ability -> "2".equals(ability.getParam("NumDmg"))).findFirst().orElseThrow();
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, game, caps, 1L, second, "to force a line through the hooks");

        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        assertEquals(parse("{\"ability\":"
                        + ProvenanceKey.toJsonArray(ProvenanceResolver.resolve(second).keys()) + "}").get("ability"),
                parse(records.get(0)).get("ability"), records.get(0));
        List<Map<String, Object>> damage = RecordEvents.events(records.get(0)).stream()
                .filter(event -> EventType.DAMAGE_DEALT.wireName().equals(event.get("type"))).toList();
        assertEquals(1, damage.size(), records.get(0));
        assertEquals(2, ((Number) ((Map<?, ?>) damage.get(0).get("params")).get("amount")).intValue());
    }

    /**
     * A line is forced at a step's first priority, never at its phase
     * event, which comes before the step's turn-based action and also for a
     * step the turn then skips: steps that publish their phase event and
     * give no priority force nothing, and the line is forced at the next
     * step that gives priority, on the board as it stands there.
     */
    @Test
    void aLineIsForcedAtAStepsFirstPriorityNeverAtItsPhaseEvent() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inHand(game, game.getPlayers().get(0), "Lightning Bolt");

        List<String> records = forced(game, 1, () -> {
            steps(game, BOUNDARIES, () -> { });
            // What tells the step the fork was taken at.
            game.getPlayers().get(1).setLife(7, null);
            boundaries(game, BOUNDARIES);
        });

        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("{\"id\":\"P1\",\"life\":7,"),
                "forked at a phase event: " + records.get(0));
    }

    /**
     * The draw step gives priority once its player has drawn, so a line
     * forced there has the post-draw library: the intervention, drawn and
     * carried over upkeeps with nothing to force, is spent at the draw step
     * the engine runs, on the board that has the card drawn.
     */
    @Test
    void aLineForcedAtTheDrawStepHasThePostDrawLibrary() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        Player you = game.getPlayers().get(0);
        game.getPhaseHandler().devModeSet(PhaseType.UPKEEP, you, 2);
        for (int card = 0; card < 3; card++) {
            game.getAction().moveTo(ZoneType.Library, TestBoard.card(game, you, "Mountain"), null, null);
        }

        List<String> records = forced(game, 1, () -> {
            boundaries(game, BOUNDARIES);
            // Past every turn the gate drew for, so the carried intervention opens at once.
            game.getPhaseHandler().devModeSet(PhaseType.UPKEEP, you, 4 * BOUNDARIES);
            TestBoard.inHand(game, you, "Lightning Bolt");
            game.getPhaseHandler().devAdvanceToPhase(PhaseType.DRAW);
            TestPriority.settle(game);
        });

        assertEquals(1, records.size(), records.toString());
        Map<String, Object> state = RecordLines.asMap(parse(records.get(0)).get("state"));
        assertEquals("draw", RecordLines.asMap(state.get("global")).get("phase"), records.get(0));
        Map<String, Object> drawer = RecordLines.asMap(((List<?>) state.get("players")).get(0));
        assertEquals(2L, drawer.get("library"), records.get(0));
        assertEquals(1L, RecordLines.asMap(drawer.get("turn_counts")).get("cards_drawn"), records.get(0));
    }

    /**
     * The beginning-of-combat step (507) gives priority and Forge has built
     * the turn's combat by then, so its score check passes: a line is forced
     * and written at its first priority, on a board with creatures.
     */
    @Test
    void aLineIsForcedAtTheBeginningOfCombatsFirstPriority() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, you, "Grizzly Bears");
        TestBoard.inPlay(game, opponent, "Hill Giant");
        TestBoard.inHand(game, you, "Lightning Bolt");

        List<String> records = forced(game, 1, () -> {
            // Past every turn the gate drew for, so the carried intervention opens at once.
            game.getPhaseHandler().devModeSet(PhaseType.MAIN1, you, 4 * BOUNDARIES);
            game.getPhaseHandler().devAdvanceToPhase(PhaseType.COMBAT_BEGIN);
            TestPriority.settle(game);
        });

        assertEquals(1, records.size(), records.toString());
        Map<String, Object> global = RecordLines.asMap(
                RecordLines.asMap(parse(records.get(0)).get("state")).get("global"));
        assertEquals("combat_begin", global.get("phase"), records.get(0));
    }

    /**
     * A declared attack that is blocked: a line forced at the declare-blockers
     * step's and at the combat-damage step's first priority is written, each
     * record's board holding the attacker. {@code InterventionSpreadTest}'s
     * seeded games force at no combat step that needs an attacker, so this
     * is the case that checks one.
     */
    @Test
    void aLineForcedInACombatStepIsWrittenWithTheAttackerOnTheBoard() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        forge.game.card.Card attacker = TestBoard.inPlay(game, you, "Grizzly Bears");
        forge.game.card.Card blocker = TestBoard.inPlay(game, opponent, "Hill Giant");
        TestBoard.inHand(game, you, "Lightning Bolt");
        TestBoard.inHand(game, you, "Shock");

        List<String> records = forced(game, 2, () -> {
            game.getPhaseHandler().devModeSet(PhaseType.COMBAT_DECLARE_BLOCKERS, you, 4 * BOUNDARIES);
            TestBoard.blockedCombat(new TestBoard.CombatSides(game, you, opponent), attacker, blocker);
            TestPriority.settle(game);
            game.getPhaseHandler().devModeSet(PhaseType.COMBAT_DAMAGE, you, 4 * BOUNDARIES);
            TestBoard.blockedCombat(new TestBoard.CombatSides(game, you, opponent), attacker, blocker);
            TestPriority.settle(game);
        });

        assertEquals(List.of("combat_declare_blockers", "combat_damage"), records.stream()
                .map(record -> RecordLines.asMap(RecordLines.asMap(parse(record).get("state")).get("global"))
                        .get("phase"))
                .toList(), records.toString());
        for (String record : records) {
            assertTrue(RecordLines.asMap(parse(record).get("state")).get("entities") instanceof List<?> entities
                            && entities.stream().map(RecordLines::asMap).anyMatch(entity ->
                                    entity.get("combat") != null
                                            && RecordLines.asMap(entity.get("combat")).get("attacking") != null),
                    "a combat step's forced record with no attacker: " + record);
        }
    }

    /**
     * Spirit en-Dal's forecast is activated only during its holder's upkeep
     * and once a turn: held through its holder's end step and another
     * player's upkeep it is not forced, and its forced record's board is in
     * its holder's upkeep.
     */
    @Test
    void aForecastLineIsForcedOnlyInItsHoldersUpkeep() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.END_OF_TURN);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inHand(game, you, "Spirit en-Dal");
        TestBoard.inPlay(game, opponent, "Grizzly Bears");

        List<String> records = forced(game, 1, () -> {
            boundaries(game, BOUNDARIES);
            game.getPhaseHandler().devModeSet(PhaseType.UPKEEP, opponent);
            boundaries(game, BOUNDARIES);
            game.getPhaseHandler().devModeSet(PhaseType.UPKEEP, you);
            boundaries(game, BOUNDARIES);
        });

        assertEquals(1, records.size(), records.toString());
        Map<String, Object> global = RecordLines.asMap(
                RecordLines.asMap(parse(records.get(0)).get("state")).get("global"));
        assertEquals("upkeep", global.get("phase"), records.get(0));
        assertEquals(parse(records.get(0)).get("actor_player"), global.get("active"), records.get(0));
    }
}
