package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardRegenerated;
import forge.game.event.GameEventDayTimeChanged;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventPlayerCounters;
import forge.game.event.GameEventPlayerRadiation;
import forge.game.event.GameEventShuffle;
import forge.game.event.GameEventSpeedChanged;
import forge.game.player.Player;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Bus events Forge already fires, read the way the schema names them.
 *
 * <p>Energy is the subtle one: it is a player counter, so it arrives on
 * {@code GameEventPlayerCounters} rather than an event of its own, and
 * without a subscriber there an Aether Hub resolution reads as an ability
 * that does nothing. Radiation, speed and day/night are each on a dedicated
 * event: declared in the vocabulary, reachable on the bus, and read here.
 */
@ExtendWith(ForgeExtension.class)
class BusEventsTest {

    private final Player player = new Player("counters", TestCards.game(), 1);

    /**
     * Energy is a player counter, so it arrives on the counters event and not on
     * one of its own. Unread, Aether Hub reads as an ability that does nothing.
     */
    @Test
    void anEnergyCounterBecomesAnEnergyChangeEvent() {
        EffectEvent event = BusEvents.playerCounter(
                new GameEventPlayerCounters(
                        player, CounterEnumType.ENERGY, 0, 2));

        assertNotNull(event);
        assertEquals(EventType.ENERGY_CHANGE, event.type());
        assertEquals(2, event.params().get("delta"));
        assertEquals(List.of("P" + player.getId()), event.subjects());
    }

    /**
     * The delta is {@code amount - oldValue}, not {@code amount} alone.
     *
     * <p>Despite its name, {@code GameEventPlayerCounters.amount()} is fired
     * with the counter's new total rather than the change — {@code
     * Player.setCounters} computes the real delta as {@code num - old} for the
     * sibling poison and radiation events two lines below the same fire site,
     * from the same two values. A player who already has 3 energy and taps
     * Aether Hub again fires {@code oldValue=3, amount=5}; reading {@code
     * amount} alone would record a delta of 5 for a tap that only added 2.
     */
    @Test
    void anEnergyCounterDeltaIsTheChangeNotTheNewTotal() {
        EffectEvent event = BusEvents.playerCounter(
                new GameEventPlayerCounters(
                        player, CounterEnumType.ENERGY, 3, 5));

        assertEquals(2, event.params().get("delta"));
    }

    /** A counter with no vocabulary entry is not guessed at. */
    @Test
    void aPlayerCounterWithNoEventTypeIsSkipped() {
        assertNull(BusEvents.playerCounter(
                new GameEventPlayerCounters(
                        player, CounterEnumType.EXPERIENCE, 0, 1)));
    }

    /**
     * Forge also fires this event with no counter type at all, for a bulk
     * reset — {@code Player.clearCounters} and the whole-{@code Multiset}
     * overload {@code GameCopier} and {@code GameSnapshot} use to copy a
     * player's counters onto a fresh game both name no single counter. That is
     * even less of a reading than an unmapped type, not more: skipped the same
     * way rather than risking a {@code NullPointerException} on {@code
     * event.type().getName()}.
     */
    @Test
    void aPlayerCounterEventWithNoTypeAtAllIsSkipped() {
        assertNull(BusEvents.playerCounter(
                new GameEventPlayerCounters(player, null, 0, 0)));
    }

    @Test
    void daytimeBecomesADayNightEvent() {
        EffectEvent event = BusEvents.dayTime(new GameEventDayTimeChanged(false));

        assertEquals(EventType.DAY_NIGHT_CHANGED, event.type());
        assertEquals("night", event.params().get("to"));
    }

    @Test
    void speedAndRadiationCarryTheirDelta() {
        assertEquals(2, BusEvents.speed(
                new GameEventSpeedChanged(player, 1, 3)).params().get("delta"));
        assertEquals(3, BusEvents.radiation(
                new GameEventPlayerRadiation(player, player, 3)).params().get("delta"));
    }

    /**
     * Regeneration and shuffle, two types the run-parameter mode table cannot
     * reach — no Forge trigger or replacement mode is the regeneration
     * itself, and reading the bus instead is what makes the type reachable at
     * all.
     */
    @Test
    void aRegeneratedCardNamesTheCard() {
        Card card = TestCards.build("Runeclaw Bear");

        EffectEvent event = BusEvents.regenerated(new GameEventCardRegenerated(card));

        assertEquals(EventType.REGENERATED, event.type());
        assertEquals(List.of("E" + card.getId()), event.subjects());
    }

    @Test
    void aShuffleNamesThePlayer() {
        EffectEvent event = BusEvents.libraryShuffled(new GameEventShuffle(player));

        assertEquals(EventType.LIBRARY_SHUFFLED, event.type());
        assertEquals(List.of("P" + player.getId()), event.subjects());
    }

    // ── player_won: name resolved back to the real Player, not carried as-is ──

    /**
     * {@code computeWinningPlayerName} returns null when {@code
     * GameOutcome.getWinningLobbyPlayer()} does -- a draw -- and a {@code
     * player_won} with no winner would be a false record. Read against a game
     * with real seats, because a game with none finds no winner whatever the
     * event names.
     */
    @Test
    void aDrawNamesNoWinner() {
        GameEventGameOutcome draw = new GameEventGameOutcome(4, List.of("Draw"), null, "");

        assertNull(BusEvents.gameOutcome(draw, TestBoard.twoPlayerGame("p1", "p2")));
    }

    /**
     * The winner is a NAME on the event, and every other subject on this
     * branch is a ref -- resolved back to the real {@link Player} by matching
     * {@code getLobbyPlayer().getName()}; a name left unresolved would break
     * the ref convention the way an unresolved {@code damage_prevented.source}
     * would.
     */
    @Test
    void theWinnerResolvesToTheMatchingPlayersRef() {
        Game game = TestBoard.twoPlayerGame("p1", "p2");
        Player p1 = game.getPlayers().stream()
                .filter(p -> "p1".equals(p.getLobbyPlayer().getName()))
                .findFirst().orElseThrow();
        GameEventGameOutcome event = new GameEventGameOutcome(
                4, List.of("p1 has won"), "p1", "p1: 1 p2: 0 ");

        EffectEvent built = BusEvents.gameOutcome(event, game);

        assertEquals(EventType.PLAYER_WON, built.type());
        assertEquals(List.of(Refs.of(p1)), built.subjects());
    }

    /** A winning name this game cannot resolve to any seat produces nothing. */
    @Test
    void aWinningNameNoSeatMatchesProducesNoEvent() {
        Game game = TestBoard.twoPlayerGame("p1", "p2");
        GameEventGameOutcome event = new GameEventGameOutcome(
                4, List.of("nobody has won"), "a name nobody at this table has", "");

        assertNull(BusEvents.gameOutcome(event, game));
    }

    /**
     * Two players sharing a lobby name is not a shape this connector's own
     * worker setup produces, but the resolution has to do something rather
     * than throw -- the first match in {@code game.getPlayers()}'s own order.
     */
    @Test
    void aSharedLobbyNameResolvesToTheFirstMatchingSeat() {
        Game game = TestBoard.twoPlayerGame("same", "same");
        Player first = game.getPlayers().get(0);
        GameEventGameOutcome event = new GameEventGameOutcome(
                4, List.of("same has won"), "same", "");

        EffectEvent built = BusEvents.gameOutcome(event, game);

        assertEquals(List.of(Refs.of(first)), built.subjects());
    }

    /**
     * A card put onto the stack is a cast, not an outcome of one.
     *
     * <p>It has a record already -- the activation half, with what was paid. As a
     * zone_change it lands in whichever bracket is open, so an opponent's
     * response reads as something the resolving ability did.
     */
    @Test
    void aMoveOntoTheStackIsNotRecordedAsAnOutcome() {
        assertNull(BusEvents.cardMoved(new GameEventCardChangeZone(
                TestCards.build("Lightning Bolt"),
                new Zone(ZoneType.Hand, TestCards.game()),
                new Zone(ZoneType.Stack, TestCards.game()))));
    }

    // ── a completed move ────────────────────────────────────────────────

    private static Zone zone(ZoneType type) {
        return new Zone(type, TestCards.game());
    }

    /**
     * A completed move says where the card came from as well as where it went.
     *
     * <p>{@code from_zone} is a documented parameter of the event type. The
     * per-zone notification names one zone and not a move, so it is the
     * completed move that carries both ends.
     */
    @Test
    void aMoveNamesBothOfItsEnds() {
        Card card = TestCards.build("Mountain");

        EffectEvent moved = BusEvents.cardMoved(new GameEventCardChangeZone(
                card, zone(ZoneType.Hand), zone(ZoneType.Battlefield)));

        assertEquals(EventType.ZONE_CHANGE, moved.type());
        String json = moved.toJson();
        assertTrue(json.contains("\"from_zone\":\"hand\""), json);
        assertTrue(json.contains("\"to_zone\":\"battlefield\""), json);
        assertTrue(json.contains("\"E" + card.getId() + "\""), json);
    }

    /** A card entering the game from nowhere still names where it landed. */
    @Test
    void aCardArrivingFromNowhereStillNamesWhereItLanded() {
        EffectEvent moved = BusEvents.cardMoved(new GameEventCardChangeZone(
                TestCards.build("Mountain"), null, zone(ZoneType.Battlefield)));

        String json = moved.toJson();
        assertTrue(json.contains("\"to_zone\":\"battlefield\""), json);
        assertFalse(json.contains("\"from_zone\""), json);
    }

    /** And a move with neither end is not a move this collector can describe. */
    @Test
    void aMoveWithNeitherEndIsNotRecorded() {
        assertNull(BusEvents.cardMoved(
                new GameEventCardChangeZone(TestCards.build("Mountain"), null, null)));
    }
}
