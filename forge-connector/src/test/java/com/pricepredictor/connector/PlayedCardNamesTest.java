package com.pricepredictor.connector;

import forge.StaticData;
import forge.ai.LobbyPlayerAi;
import forge.card.CardStateName;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.card.CardView;
import forge.game.event.GameEventCardChangeZone;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A played card is recorded under the name its deck lists, whichever face was
 * cast: the cards-played columns are sets of the deck's own card names.
 */
@ExtendWith(ForgeExtension.class)
class PlayedCardNamesTest {

    private static Game twoPlayerGame() {
        Deck deck = new Deck();
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("p1", null)),
                new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("p2", null)));
        GameRules rules = new GameRules(GameType.Constructed);
        return new Game(players, rules, new Match(rules, players, "PlayedCardNamesTest"));
    }

    /** {@code cardName} with {@code face} showing, entering the battlefield under p1. */
    private static List<String> playedAs(String cardName, CardStateName face) {
        Game game = twoPlayerGame();
        Player owner = game.getPlayers().get(0);
        Card card = CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard(cardName), owner, game);
        game.getAction().moveToHand(card, null);
        if (face != null) {
            card.setState(face, true);
        }
        PlayedCardCollector collector = new PlayedCardCollector(game);
        collector.visit(new GameEventCardChangeZone(
                CardView.get(card), null,
                owner.getZone(ZoneType.Battlefield).getView()));
        return collector.getCards(Side.A);
    }

    @Test
    void aCollectorThatSawNothingHasNoCards() {
        PlayedCardCollector collector = new PlayedCardCollector(twoPlayerGame());
        assertTrue(collector.getCards(Side.A).isEmpty());
        assertTrue(collector.getCards(Side.B).isEmpty());
    }

    @Test
    void anOrdinaryCardIsRecordedUnderItsName() {
        assertEquals(List.of("Lightning Bolt"), playedAs("Lightning Bolt", null));
    }

    @Test
    void aRoomDoorIsRecordedUnderTheFullRoomName() {
        String full = "Painter's Studio // Defaced Gallery";
        Game game = twoPlayerGame();
        Card room = CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard(full),
                game.getPlayers().get(0), game);
        room.setState(CardStateName.LeftSplit, true);
        // The test only guards the wiring if a door's own name is on show.
        assertEquals("Painter's Studio", CardView.get(room).getCurrentState().getOracleName());

        assertEquals(List.of(full), playedAs(full, CardStateName.LeftSplit));
        assertEquals(List.of(full), playedAs(full, CardStateName.RightSplit));
    }

    @Test
    void aSplitCardHalfIsRecordedUnderTheFullName() {
        assertEquals(List.of("Fire // Ice"), playedAs("Fire // Ice", CardStateName.RightSplit));
    }

    @Test
    void aBackFaceIsRecordedUnderTheDeckListedName() {
        assertEquals(List.of("Delver of Secrets"),
                playedAs("Delver of Secrets", CardStateName.Backside));
    }
}
