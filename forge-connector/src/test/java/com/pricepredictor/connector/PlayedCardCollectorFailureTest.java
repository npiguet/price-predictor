package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.TestBoard;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventCardChangeZone;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A card owned by a lobby player neither side registers cannot be credited to
 * a deck. Forge's event bus swallows a subscriber's exception, so the
 * collector keeps the failure for the caller to read at the end of the game.
 */
@ExtendWith(ForgeExtension.class)
class PlayedCardCollectorFailureTest {

    private static PlayedCardCollector collectorAfterABoltEnteredFor(String ownerLobbyName) {
        Game game = TestBoard.twoPlayerGame(ownerLobbyName, "p2");
        Player owner = game.getPlayers().get(0);
        Card card = TestBoard.card(game, owner, "Lightning Bolt");
        game.getAction().moveToHand(card, null);
        PlayedCardCollector collector = new PlayedCardCollector(game);
        assertDoesNotThrow(() -> collector.visit(new GameEventCardChangeZone(
                CardView.get(card), null, owner.getZone(ZoneType.Battlefield).getView())));
        return collector;
    }

    @Test
    void aCardOfAnUnregisteredPlayerRecordsAFailureInsteadOfThrowing() {
        PlayedCardCollector collector = collectorAfterABoltEnteredFor("nobody");

        assertNotNull(collector.failure(), "the failure must be kept for the end of the game");
        assertTrue(collector.failure().contains("nobody"), collector.failure());
        assertTrue(collector.getCards(Side.A).isEmpty());
        assertTrue(collector.getCards(Side.B).isEmpty());
    }

    @Test
    void aRegisteredPlayersCardLeavesNoFailure() {
        assertNull(collectorAfterABoltEnteredFor("p1").failure());
    }
}
