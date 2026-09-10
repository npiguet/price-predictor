package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.event.GameEventCardRegenerated;
import forge.game.event.GameEventDayTimeChanged;
import forge.game.event.GameEventPlayerCounters;
import forge.game.event.GameEventPlayerRadiation;
import forge.game.event.GameEventShuffle;
import forge.game.event.GameEventSpeedChanged;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Four bus events Forge already fires, read the way the schema names them.
 *
 * <p>Energy is the one that mattered: it is a player counter, so it arrives on
 * {@code GameEventPlayerCounters} rather than an event of its own, and nothing
 * had ever subscribed to that event — the reason a corpus of ten million
 * records held no energy at all and an Aether Hub resolution reads as an
 * ability that does nothing. Radiation, speed and day/night are each on a
 * dedicated event and were in exactly the same state: declared in the
 * vocabulary, reachable on the bus, and never read.
 */
@ExtendWith(ForgeExtension.class)
class BusEventsTest {

    private final Player player = new Player("counters", TestCards.game(), 1);

    /**
     * Energy is a player counter, so it arrives on the counters event and not on
     * one of its own. Nothing subscribed to that event, which is why a corpus of
     * ten million records contains no energy at all and Aether Hub reads as an
     * ability that does nothing.
     */
    @Test
    void anEnergyCounterBecomesAnEnergyChangeEvent() {
        EffectEvent event = BusEvents.playerCounter(
                new GameEventPlayerCounters(
                        player, CounterEnumType.ENERGY, 0, 2));

        assertNotNull(event);
        assertEquals(EffectEvent.ENERGY_CHANGE, event.type());
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

        assertEquals(EffectEvent.DAY_NIGHT_CHANGED, event.type());
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
     * Regeneration and shuffle, the two types the mode table's own dead and
     * misfiled entries left unreachable — {@code "Regenerated"} matches no
     * real Forge trigger or replacement mode, and reading the bus instead is
     * what makes the type reachable at all.
     */
    @Test
    void aRegeneratedCardNamesTheCard() {
        Card card = TestCards.build("Runeclaw Bear");

        EffectEvent event = BusEvents.regenerated(new GameEventCardRegenerated(card));

        assertEquals(EffectEvent.REGENERATED, event.type());
        assertEquals(List.of("E" + card.getId()), event.subjects());
    }

    @Test
    void aShuffleNamesThePlayer() {
        EffectEvent event = BusEvents.libraryShuffled(new GameEventShuffle(player));

        assertEquals(EffectEvent.LIBRARY_SHUFFLED, event.type());
        assertEquals(List.of("P" + player.getId()), event.subjects());
    }
}
