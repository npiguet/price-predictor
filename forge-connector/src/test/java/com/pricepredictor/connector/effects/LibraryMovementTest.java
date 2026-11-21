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

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordEvents.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.TestBoard.castAndResolve;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.priorityWithTriggers;
import static com.pricepredictor.connector.effects.TestBoard.spellOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Draws and mills told apart from other moves out of a library, on real
 * resolutions recorded as a worker records them.
 *
 * <p>A card leaving a library for its owner's hand is a draw only when Forge
 * counted one (its per-player draw counter moved); a tutor or an explore
 * that reveals a land moves a card the same way and draws nothing. A surveil
 * mills each card it bins once: the card's move to the graveyard is the
 * mill, the surveil itself adds none.
 */
@ExtendWith(ForgeExtension.class)
class LibraryMovementTest {

    private static final String WHY = "to record a resolution's library movements";

    @TempDir
    Path tempDir;

    /** The sum of the {@code count} of every {@code type} event in {@code all}. */
    private static long total(List<String> all, String type) {
        long sum = 0;
        for (String record : all) {
            for (Map<String, Object> event : events(record)) {
                if (type.equals(event.get("type"))) {
                    sum += ((Number) params(event).get("count")).longValue();
                }
            }
        }
        return sum;
    }

    private static void library(Game game, Player owner, String name, int count) {
        for (int i = 0; i < count; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, owner, name), null, null);
        }
    }

    /** Resolve {@code name} cast by {@code caster} from its hand, then move to the next step. */
    private List<String> resolved(Game game, Player caster, String name) throws Throwable {
        Card card = TestBoard.inHand(game, caster, name);
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            Card onStack = game.getAction().moveToStack(card, null);
            castAndResolve(game, spellOf(onStack, caster));
            nextStep(game);
        });
    }

    /** The first player of a fresh main-phase game, with {@code hillGiants} Hill Giants in the library. */
    private static Player casterWithLibrary(int hillGiants) {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        library(game, caster, "Hill Giant", hillGiants);
        return caster;
    }

    /** A card that a resolution must leave in a zone of its caster, and what to say if it does not. */
    private record Reached(ZoneType zone, String card, String message) {
    }

    /** The {@code from_zone>to_zone} of each {@code zone_change} event in {@code all}, in order. */
    private static List<String> zoneChanges(List<String> all) {
        List<String> moves = new ArrayList<>();
        for (String record : all) {
            for (Map<String, Object> event : events(record)) {
                if ("zone_change".equals(event.get("type"))) {
                    moves.add(params(event).get("from_zone") + ">" + params(event).get("to_zone"));
                }
            }
        }
        return moves;
    }

    /** Resolve {@code name}, check the caster ended with the card {@code reached} names, return the records. */
    private List<String> resolvedReaching(Player caster, String name, Reached reached)
            throws Throwable {
        List<String> all = resolved(caster.getGame(), caster, name);
        assertTrue(caster.getCardsIn(reached.zone()).stream()
                .anyMatch(card -> card.getName().equals(reached.card())), reached.message());
        return all;
    }

    /**
     * Notion Rain surveils 2 into the graveyard, then draws two: two cards
     * milled and two drawn, each told once.
     */
    @Test
    void surveilTwoIntoTheGraveyardMillsTwo() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        // Non-lands with no land in play or in hand: the AI bins both.
        library(game, caster, "Hill Giant", 12);

        List<String> all = resolved(game, caster, "Notion Rain");

        assertEquals(2, caster.getCardsIn(ZoneType.Graveyard).stream()
                .filter(card -> card.getName().equals("Hill Giant")).count(),
                "the AI binned both surveilled cards");
        assertEquals(2, total(all, "card_milled"), all.toString());
        assertEquals(2, total(all, "card_drawn"), all.toString());
    }

    /** Lay of the Land puts a basic land from the library into the hand: nothing drawn. */
    @Test
    void aTutorToHandDrawsNothing() throws Throwable {
        Player caster = casterWithLibrary(4);
        library(caster.getGame(), caster, "Forest", 1);

        List<String> all = resolvedReaching(caster, "Lay of the Land",
                new Reached(ZoneType.Hand, "Forest", "the Forest was fetched"));

        assertEquals(0, total(all, "card_drawn"), all.toString());
    }

    /** Merfolk Branchwalker explores and reveals a land, which goes to the hand: nothing drawn. */
    @Test
    void anExploreRevealingALandDrawsNothing() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        library(game, caster, "Hill Giant", 4);
        library(game, caster, "Forest", 1); // on top

        Card walker = TestBoard.inHand(game, caster, "Merfolk Branchwalker");
        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            Card onStack = game.getAction().moveToStack(walker, null);
            castAndResolve(game, spellOf(onStack, caster));
            priorityWithTriggers(game);
            while (!game.getStack().isEmpty()) {
                game.getStack().resolveStack();
                priorityWithTriggers(game);
            }
            nextStep(game);
        });

        assertTrue(caster.getCardsIn(ZoneType.Hand).stream()
                .anyMatch(card -> card.getName().equals("Forest")), "the explore took the Forest");
        assertTrue(all.stream().anyMatch(r -> r.contains("merfolk_branchwalker")),
                "the explore's resolution was recorded: " + all);
        assertEquals(0, total(all, "card_drawn"), all.toString());
    }

    /** Opt scries 1 and draws one: one card drawn. */
    @Test
    void aRealDrawDrawsOne() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        library(game, caster, "Hill Giant", 6);

        List<String> all = resolved(game, caster, "Opt");

        assertEquals(1, total(all, "card_drawn"), all.toString());
    }

    /**
     * Fact or Fiction puts the cards of the pile the caster does not take
     * into the graveyard: moves out of the library that Forge never counts as
     * a mill, so none is written.
     */
    @Test
    void aDigsRestInTheGraveyardMillsNothing() throws Throwable {
        Player caster = casterWithLibrary(12);

        List<String> all = resolvedReaching(caster, "Fact or Fiction",
                new Reached(ZoneType.Graveyard, "Hill Giant", "a pile reached the graveyard"));

        assertEquals(0, total(all, "card_milled"), all.toString());
    }

    /**
     * Grapple with the Past mills three, then returns the one creature among
     * them to the hand: the returned card was milled all the same, so three
     * cards are milled and the return follows.
     */
    @Test
    void aMilledCardTheSameResolutionReturnsWasStillMilled() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        // Top of the library last: Divination, Divination, then the Bears third.
        library(game, caster, "Grizzly Bears", 1);
        library(game, caster, "Divination", 2);

        List<String> all = resolvedReaching(caster, "Grapple with the Past",
                new Reached(ZoneType.Hand, "Grizzly Bears", "the Bears were returned"));

        assertEquals(3, total(all, "card_milled"), all.toString());
        List<String> moves = zoneChanges(all);
        assertEquals(3, moves.stream().filter("library>graveyard"::equals).count(), moves.toString());
        assertTrue(moves.contains("graveyard>hand"), "the Bears' return follows: " + moves);
    }

    /** Entomb puts a chosen card from the library into the graveyard: a move, no mill. */
    @Test
    void anEntombMillsNothing() throws Throwable {
        Player caster = casterWithLibrary(4);

        List<String> all = resolvedReaching(caster, "Entomb",
                new Reached(ZoneType.Graveyard, "Hill Giant", "a card was entombed"));

        assertEquals(0, total(all, "card_milled"), all.toString());
    }

    /**
     * A discard that madness sends to exile instead of the graveyard is still
     * a discard (702.35a: the card is discarded, then exiled by the madness
     * replacement).
     */
    @Test
    void aDiscardReplacedByMadnessIsStillADiscard() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player victim = game.getPlayers().get(1);
        TestBoard.inHand(game, victim, "Fiery Temper");
        Card rot = TestBoard.inHand(game, caster, "Mind Rot");

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.castAndResolve(game, TestBoard.cast(game, rot, caster, victim));
            nextStep(game);
        });

        assertEquals(0, victim.getCardsIn(ZoneType.Graveyard).stream()
                .filter(card -> card.getName().equals("Fiery Temper")).count(),
                "madness took the card away from the graveyard");
        assertEquals(1, total(all, "card_discarded"), all.toString());
    }

    /** An ordinary discard still reads as one. */
    @Test
    void anOrdinaryDiscardIsADiscard() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player victim = game.getPlayers().get(1);
        TestBoard.inHand(game, victim, "Hill Giant");
        Card rot = TestBoard.inHand(game, caster, "Mind Rot");

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.castAndResolve(game, TestBoard.cast(game, rot, caster, victim));
            nextStep(game);
        });

        assertEquals(1, total(all, "card_discarded"), all.toString());
    }
}
