package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RandomSeatController;
import com.pricepredictor.connector.RandomSeatLobbyPlayer;
import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The random seat aiming a play at a spell on the stack (findings 3.1, 3.2),
 * end to end on the real engine: the seat draws the counterspell, draws its
 * target, casts it, and the stack resolves.
 *
 * <p>The seat's only playable spell is the counterspell and {@code P} is one,
 * so the play and its target are the random draw's, never the AI's.
 */
@ExtendWith(ForgeExtension.class)
class RandomSeatStackTargetsTest {

    private static final String WHY = "to record the countering clause's effect half";
    private static final String NEGATE = "cardsfolder/n/negate.txt";

    @TempDir
    Path tempDir;

    /** Seat A random with {@code P} one, in A's second main phase, two Islands untapped. */
    private static Game gameWithRandomSeatA(long seed) {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN2,
                new RandomSeatLobbyPlayer("a", 1.0, new Random(seed)),
                new LobbyPlayerAi("b", null));
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Island");
        TestBoard.inPlay(game, a, "Island");
        return game;
    }

    private static RandomSeatController seatOf(Player player) {
        RandomSeatController seat = (RandomSeatController) player.getController();
        seat.attach(player.getGame(), null, null);
        return seat;
    }

    /** The seat's one play, drawn and cast, then the stack resolved to empty. */
    private static void playAndResolve(Game game, RandomSeatController seat, String card) {
        List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();
        assertEquals(card, chosen.get(0).getHostCard().getName());
        assertTrue(seat.playChosenSpellAbility(chosen.get(0)), "the draw finds the spell");
        while (!game.getStack().isEmpty()) {
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
        }
    }

    private static boolean holds(Player player, ZoneType zone, String name) {
        return player.getCardsIn(zone).stream().anyMatch(c -> c.getName().equals(name));
    }

    /**
     * Negate the seat casts at a Lightning Bolt aimed at it: its effect half
     * holds the Bolt's {@code spell_countered}, and the Bolt never deals its
     * damage. With the Bolt's card drawn as the target, Negate resolved
     * doing nothing and the Bolt then resolved.
     */
    @Test
    void negateTheSeatDrewCountersTheInstant() throws Throwable {
        Game game = gameWithRandomSeatA(1L);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        TestBoard.inHand(game, a, "Negate");
        RandomSeatController seat = seatOf(a);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, TestBoard.inHand(game, b, "Lightning Bolt"), b, a);
            playAndResolve(game, seat, "Negate");
            TestBoard.nextStep(game);
        });

        assertEquals(20, a.getLife(), "the Bolt did not resolve");
        assertTrue(holds(b, ZoneType.Graveyard, "Lightning Bolt"), "the Bolt was countered");
        String countering = only(all, "resolution", NEGATE);
        assertTrue(eventTypes(countering).contains("spell_countered"), countering);
    }

    /**
     * Essence Scatter at a creature spell can be drawn, and counters it: no
     * card of the stack zone was ever its candidate, so every draw failed and
     * the seat could never counter a permanent spell.
     */
    @Test
    void essenceScatterTheSeatDrewCountersTheCreatureSpell() {
        for (long seed = 1; seed <= 3; seed++) {
            Game game = gameWithRandomSeatA(seed);
            Player a = game.getPlayers().get(0);
            Player b = game.getPlayers().get(1);
            TestBoard.inHand(game, a, "Essence Scatter");
            SpellAbility bears = TestBoard.castUntargeted(game, b, "Grizzly Bears");
            Card bearsCard = bears.getHostCard();
            RandomSeatController seat = seatOf(a);

            playAndResolve(game, seat, "Essence Scatter");

            assertTrue(holds(b, ZoneType.Graveyard, bearsCard.getName()), "seed " + seed + ": countered");
            assertTrue(b.getCardsIn(ZoneType.Battlefield).isEmpty(), "seed " + seed + ": never resolved");
        }
    }
}
