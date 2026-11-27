package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.TestBoard;
import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A random cost choice never strands a play.
 *
 * <p>The sacrifice and tap choices are drawn as the cast pays, after the card
 * has moved to the stack. The engine draws every part's decision before it
 * pays any part, and pays mana first, so a draw is made on the untapped board
 * and a play either fails before any payment or is made whole: the boards
 * below pin that the card is never left in the stack zone and the game goes
 * on.
 */
@ExtendWith(ForgeExtension.class)
class RandomSeatCostChoiceTest {

    private static RandomSeatController seatOf(Game game) {
        RandomSeatController seat = GamePlayer.randomSeatOf(game).orElseThrow();
        seat.attach(game, null, null);
        return seat;
    }

    private static Game newGame() {
        return TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new RandomSeatLobbyPlayer("a", 1.0, new Random(1L)), new LobbyPlayerAi("b", null));
    }

    /** Plays the seat's one play and returns where things stand, as text, for a failure message. */
    private static String playOnce(RandomSeatController seat, Game game, Card spellCard, boolean[] played) {
        List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();
        assertNotNull(chosen, "the board has a play");
        played[0] = seat.playChosenSpellAbility(chosen.get(0));
        return "played=" + played[0] + " card in " + game.getCardState(spellCard).getZone()
                + " stack=" + game.getStack().size();
    }

    /**
     * Life's Legacy ({1}{G}, sacrifice a creature) with Llanowar Elves the only
     * creature and a Forest: made whole, the Elves sacrificed.
     */
    @Test
    void aSacrificeCostPlayIsMadeWhole() {
        Game game = newGame();
        Player a = game.getPlayers().get(0);
        Card elves = TestBoard.inPlay(game, a, "Llanowar Elves");
        TestBoard.inPlay(game, a, "Forest");
        Card legacy = TestBoard.inHand(game, a, "Life's Legacy");
        RandomSeatController seat = seatOf(game);

        boolean[] played = new boolean[1];
        String where = playOnce(seat, game, legacy, played);

        assertTrue(played[0], where);
        assertEquals(ZoneType.Stack, game.getCardState(legacy).getZone().getZoneType(), where);
        assertEquals(1, game.getStack().size(), where);
        assertEquals(ZoneType.Graveyard, game.getCardState(elves).getZone().getZoneType(), where);
        game.getStack().resolveStack();
        seat.chooseSpellAbilityToPlay();
    }

    /**
     * Revelsong Horn ({1}, {T}, tap an untapped creature you control) with
     * Llanowar Elves the only creature and the only mana: the Elves cannot
     * make the mana and be tapped as the creature (601.2g), so the play is
     * never made, and nothing is tapped or left in the stack zone.
     */
    @Test
    void aTapCostPlayWithTheOnlyCreatureMakingTheManaIsNeverMade() {
        Game game = newGame();
        Player a = game.getPlayers().get(0);
        Card elves = TestBoard.inPlay(game, a, "Llanowar Elves");
        Card horn = TestBoard.inPlay(game, a, "Revelsong Horn");
        RandomSeatController seat = seatOf(game);

        List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();

        assertTrue(chosen == null || chosen.stream().noneMatch(sa -> sa.getHostCard().equals(horn)),
                "the Horn is not offered: " + chosen);
        assertTrue(!game.getCardState(elves).isTapped() && !game.getCardState(horn).isTapped());
        assertEquals(0, game.getStack().size());
    }

    /**
     * The same Horn with a second creature: made, whatever the seed draws
     * first, the other creature tapped for the cost and the Elves for the mana.
     */
    @Test
    void aSecondCreatureIsTheOneTapped() {
        for (long seed = 1; seed <= 12; seed++) {
            Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                    new RandomSeatLobbyPlayer("a", 1.0, new Random(seed)), new LobbyPlayerAi("b", null));
            Player a = game.getPlayers().get(0);
            Card elves = TestBoard.inPlay(game, a, "Llanowar Elves");
            Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
            Card horn = TestBoard.inPlay(game, a, "Revelsong Horn");
            RandomSeatController seat = seatOf(game);

            List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();
            assertNotNull(chosen, "seed " + seed);
            SpellAbility play = chosen.stream().filter(sa -> sa.getHostCard().equals(horn))
                    .findFirst().orElse(chosen.get(0));
            if (!play.getHostCard().equals(horn)) {
                continue; // the seat drew another play (the Elves are no candidate; the Bears neither)
            }
            boolean made = seat.playChosenSpellAbility(play);
            String where = "seed " + seed + " made=" + made + " stack=" + game.getStack().size();
            assertTrue(made, where);
            assertTrue(game.getCardState(bears).isTapped(), where);
            assertTrue(game.getCardState(elves).isTapped(), where);
            assertEquals(ZoneType.Battlefield, game.getCardState(horn).getZone().getZoneType(), where);
            assertTrue(!game.getCardState(elves).isUsedToPay() && !game.getCardState(bears).isUsedToPay(), where);
        }
    }
}
