package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.PlayabilityVerdict;
import com.pricepredictor.connector.effects.PriorityCandidates;
import com.pricepredictor.connector.effects.TestBoard;
import forge.ai.LobbyPlayerAi;
import java.util.Random;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A creature is counted once across a "tap an untapped creature you control"
 * part and the mana part: 601.2g makes the mana
 * first, so a creature tapped for mana is no longer untapped to pay the tap
 * cost. Revelsong Horn is {1}, {T}, tap an untapped creature you control.
 */
@ExtendWith(ForgeExtension.class)
class TapCreatureCostTest {

    @Test
    void theOnlyCreatureCannotPayBothTheManaAndTheTap() {
        Game game = newGame();
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Llanowar Elves");
        SpellAbility horn = hornOf(game, a);

        assertFalse(PlayabilityVerdict.of(horn, a).affordable());
    }

    @Test
    void aSecondCreatureTapsWhileTheElvesMakeTheMana() {
        Game game = newGame();
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Llanowar Elves");
        TestBoard.inPlay(game, a, "Grizzly Bears");
        SpellAbility horn = hornOf(game, a);

        assertTrue(PlayabilityVerdict.of(horn, a).affordable(), PlayabilityVerdict.of(horn, a).toString());
    }

    @Test
    void aForestMakesTheManaSoTheElvesMayTap() {
        Game game = newGame();
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Llanowar Elves");
        TestBoard.inPlay(game, a, "Forest");
        SpellAbility horn = hornOf(game, a);

        assertTrue(PlayabilityVerdict.of(horn, a).affordable());
    }

    /** The Horn's activation as the seat's priority stop lists it. */
    private static Game newGame() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new RandomSeatLobbyPlayer("a", 1.0, new Random(1L)), new LobbyPlayerAi("b", null));
        GamePlayer.randomSeatOf(game).orElseThrow().attach(game, null, null);
        return game;
    }

    private static SpellAbility hornOf(Game game, Player a) {
        TestBoard.inPlay(game, a, "Revelsong Horn");
        return PriorityCandidates.candidatesOf(game, a).stream()
                .map(PriorityCandidates.Candidate::ability)
                .filter(sa -> sa.isActivatedAbility() && sa.getHostCard().getName().equals("Revelsong Horn"))
                .findFirst().orElseThrow();
    }
}
