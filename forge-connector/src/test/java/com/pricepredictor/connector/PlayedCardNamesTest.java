package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.TestBoard;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventCardChangeZone;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A played card is recorded under the name its deck lists, whichever face was
 * cast: the cards-played columns are sets of the deck's own card names.
 */
@ExtendWith(ForgeExtension.class)
class PlayedCardNamesTest {

    /** {@code cardName} with {@code face} showing, entering the battlefield under p1. */
    private static List<String> playedAs(String cardName, CardStateName face) {
        Game game = TestBoard.twoPlayerGame("p1", "p2");
        Player owner = game.getPlayers().get(0);
        Card card = TestBoard.card(game, owner, cardName);
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
        PlayedCardCollector collector =
                new PlayedCardCollector(TestBoard.twoPlayerGame("p1", "p2"));
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
        Game game = TestBoard.twoPlayerGame("p1", "p2");
        Card room = TestBoard.card(game, game.getPlayers().get(0), full);
        room.setState(CardStateName.LeftSplit, true);
        // The test only guards the wiring if a door's own name is on show.
        assertEquals("Painter's Studio", CardView.get(room).getCurrentState().getOracleName());

        assertEquals(List.of(full), playedAs(full, CardStateName.LeftSplit));
        assertEquals(List.of(full), playedAs(full, CardStateName.RightSplit));
    }

    /**
     * A card the game does not hold has no printing to read, so it is
     * recorded under the oracle name its view shows: for a Room shown by one
     * door, the door's name rather than the deck's, which is what the
     * printing route above exists to avoid.
     */
    @Test
    void aCardTheGameDoesNotHoldIsRecordedUnderTheNameItsViewShows() {
        Game elsewhere = TestBoard.twoPlayerGame("p1", "p2");
        Player owner = elsewhere.getPlayers().get(0);
        Card room = TestBoard.card(elsewhere, owner, "Painter's Studio // Defaced Gallery");
        elsewhere.getAction().moveToHand(room, null);
        room.setState(CardStateName.LeftSplit, true);
        Game watched = TestBoard.twoPlayerGame("p1", "p2");
        assertNull(watched.findById(room.getId()), "the watched game must not hold the card");

        PlayedCardCollector collector = new PlayedCardCollector(watched);
        collector.visit(new GameEventCardChangeZone(
                CardView.get(room), null,
                watched.getPlayers().get(0).getZone(ZoneType.Battlefield).getView()));

        assertEquals(List.of("Painter's Studio"), collector.getCards(Side.A));
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
