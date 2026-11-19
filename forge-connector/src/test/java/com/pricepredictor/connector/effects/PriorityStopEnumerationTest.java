package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.event.GameEvent;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * The collector's own enumeration at a priority stop is invisible to the
 * channels (finding 8.5): the rules queries {@link PriorityCandidates} makes
 * of both players' candidates run Forge's static check on the live game,
 * which evaluates every state trigger and runs the ones whose condition
 * holds, and none of that is the game's.
 */
@ExtendWith(ForgeExtension.class)
class PriorityStopEnumerationTest {

    @TempDir
    Path tempDir;

    /** Every game event published while it listens. */
    public static final class Published {
        final List<GameEvent> events = new ArrayList<>();

        @Subscribe
        public void on(GameEvent event) {
            events.add(event);
        }
    }

    /**
     * A's main phase, A holding priority: A's Kukemssa Serpent with no
     * Island beside it, whose state trigger ("when you control no Islands,
     * sacrifice it") holds and waits nowhere, as a state trigger that cannot
     * stay on the stack leaves it at every stop; B's Pirate Ship beside an
     * Island, whose same trigger does not hold; and in A's hand Exalted
     * Angel, whose face-down cast is an alternative cost Forge reads on a
     * face-down copy, re-applying every static to the live game and running
     * its state triggers ({@code GameActionUtil.getAlternativeCosts}).
     */
    private static Game aStateTriggerAndAMorphInHand() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        TestBoard.inPlay(game, a, "Kukemssa Serpent");
        TestBoard.inHand(game, a, "Exalted Angel");
        TestBoard.inPlay(game, b, "Island");
        TestBoard.inPlay(game, b, "Pirate Ship");
        // What the board's set-up queued is the set-up's, not the stop's;
        // then the triggers the state-based check before a stop registers.
        game.getStack().clearSimultaneousStack();
        game.getTriggerHandler().resetActiveTriggers();
        return game;
    }

    /** What {@code body} wrote with every hook installed; none when it wrote nothing. */
    private List<String> recorded(Game game, ShardRun.Played body) throws Throwable {
        CollectionCaps noForks = TestCaps.caps().interventions(0).probes(0).build();
        Path shard = ShardRun.recordedWithClauseHook(tempDir, game, noForks,
                "the trigger hook reports the enumeration's evaluations", body).path();
        return Files.exists(shard) ? ShardRun.readShard(shard) : List.of();
    }

    /**
     * The stop leaves the triggers waiting for the stack as it found them,
     * hands the trigger channel nothing to write, and publishes nothing.
     */
    @Test
    void theEnumerationQueuesNoTriggerAndOffersTheSamplerNothing() throws Throwable {
        Game game = aStateTriggerAndAMorphInHand();
        Published published = new Published();
        List<List<SpellAbility>> queued = new ArrayList<>();

        List<String> records = recorded(game, () -> {
            queued.add(game.getStack().getSimultaneousStackEntries());
            game.subscribeToEvents(published);
            TestPriority.settle(game);
            queued.add(game.getStack().getSimultaneousStackEntries());
        });

        assertEquals(List.of(), records.stream()
                .filter(r -> r.contains("\"kind\":\"trigger\"")).toList(),
                "the enumeration's evaluations reached the trigger channel");
        assertEquals(List.of(), published.events, "the enumeration published game events");
        assertEquals(queued.get(0), queued.get(1),
                "the stop left the triggers waiting for the stack as it found them");
    }

    /**
     * Only what the stop queued is taken off: a trigger the game had queued
     * before it keeps its place. The game's own static check queues the
     * Serpent's trigger; B's Island then leaves its zone with no static
     * check after it (taken off the zone itself: a move's own check would
     * queue the Pirate Ship's trigger), so the Pirate Ship's trigger holds
     * and waits nowhere, and the stop's static check would queue it.
     */
    @Test
    void aTriggerTheGameHadQueuedKeepsItsPlace() throws Throwable {
        Game game = aStateTriggerAndAMorphInHand();
        Player b = game.getPlayers().get(1);
        game.getAction().checkStaticAbilities(false);
        List<SpellAbility> before = game.getStack().getSimultaneousStackEntries();
        assertEquals(1, before.size(), "the game's own check queued the Serpent's trigger");
        b.getZone(ZoneType.Battlefield).remove(b.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Island")).findFirst().orElseThrow());
        List<List<SpellAbility>> after = new ArrayList<>();

        recorded(game, () -> {
            assertEquals(before, game.getStack().getSimultaneousStackEntries());
            TestPriority.settle(game);
            after.add(game.getStack().getSimultaneousStackEntries());
        });

        assertEquals(1, after.get(0).size(), after.toString());
        assertSame(before.get(0), after.get(0).get(0));
    }
}
