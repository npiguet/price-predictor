package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The inverted explore (701.44b), discover (701.57b) and connive (701.50f)
 * rules bind a card that explores, discovers or connives as it is cast or
 * enters, never one whose trigger waits for an attack, a death or a second
 * spell, and a watcher that triggers for the active player, never one that
 * only triggers for its own controller when that is the opponent (finding
 * 6.2). Each board is a real {@code TestBoard} game.
 */
@ExtendWith(ForgeExtension.class)
class ActorAndWatcherTest {

    private final Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    private final Player you = game.getPlayers().get(0);
    private final Player opponent = game.getPlayers().get(1);

    private Card permanent(Player owner, String name) {
        Card card = TestBoard.inPlay(game, owner, name);
        card.setGameTimestamp(game.getNextTimestamp());
        return card;
    }

    private void permanents(Player owner, String name, int count) {
        for (int i = 0; i < count; i++) {
            permanent(owner, name);
        }
    }

    private void library(String... names) {
        for (String name : names) {
            game.getAction().moveTo(ZoneType.Library, TestBoard.card(game, you, name), null, null);
        }
    }

    private void ledgerShredderBoard(String creature) {
        permanents(you, "Island", 4);
        permanent(you, "Glorious Purpose");
        TestBoard.inHand(game, you, creature);
        library("Forest", "Island");
        you.drawCard();
    }

    private void caparoctiBoard(String creature) {
        permanents(you, "Mountain", 2);
        permanents(you, "Plains", 2);
        permanent(you, "Val, Marooned Surveyor");
        TestBoard.inHand(game, you, creature);
        library("Forest", "Grizzly Bears");
    }

    @Test
    void aCreatureThatConnivesOnASecondSpellDoesNotBind() {
        ledgerShredderBoard("Ledger Shredder");
        assertTrue(ConniveWithoutDraw.bind(game).isEmpty());
    }

    @Test
    void aCreatureThatConnivesAsItEntersBinds() {
        ledgerShredderBoard("Echo Inspector");
        assertTrue(ConniveWithoutDraw.bind(game).isPresent());
    }

    @Test
    void aCreatureThatDiscoversWhenItAttacksDoesNotBind() {
        caparoctiBoard("Caparocti Sunborn");
        assertTrue(DiscoverNothingFound.bind(game).isEmpty());
    }

    @Test
    void aCreatureThatDiscoversAsItEntersBinds() {
        permanents(you, "Mountain", 4);
        permanent(you, "Val, Marooned Surveyor");
        TestBoard.inHand(game, you, "Geological Appraiser");
        library("Forest", "Grizzly Bears");
        assertTrue(DiscoverNothingFound.bind(game).isPresent());
    }

    @Test
    void anOpponentsWatcherAloneDoesNotBindAnExplorer() {
        permanents(you, "Forest", 2);
        permanent(opponent, "Wildgrowth Walker");
        TestBoard.inHand(game, you, "Merfolk Branchwalker");
        library("Forest", "Grizzly Bears");
        assertTrue(ExploreEmptyLibrary.bind(game).isEmpty());
    }

    @Test
    void anOpponentsGloriousPurposeAloneDoesNotBindAConniver() {
        permanents(you, "Island", 4);
        permanent(opponent, "Glorious Purpose");
        TestBoard.inHand(game, you, "Echo Inspector");
        library("Forest", "Island");
        you.drawCard();
        assertTrue(ConniveWithoutDraw.bind(game).isEmpty());
    }

    @Test
    void anOpponentsValAloneDoesNotBindADiscoverer() {
        permanents(you, "Mountain", 4);
        permanent(opponent, "Val, Marooned Surveyor");
        TestBoard.inHand(game, you, "Geological Appraiser");
        library("Forest", "Grizzly Bears");
        assertTrue(DiscoverNothingFound.bind(game).isEmpty());
    }

    @Test
    void theActivePlayersOwnWatcherBindsTheExplorer() {
        permanents(you, "Forest", 2);
        permanent(you, "Wildgrowth Walker");
        Card branchwalker = TestBoard.inHand(game, you, "Merfolk Branchwalker");
        library("Forest", "Grizzly Bears");
        assertEquals(branchwalker, ExploreEmptyLibrary.bind(game).orElseThrow().card("card"));
    }
}
