package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.event.GameEventZone;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.reflect.Method;
import java.util.LinkedHashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a fork hears, and how a move is read.
 *
 * <p>Two defects meet here. A fork described its resolution through seven bus
 * events against the sixteen the observed records use, so a forked draw or pump
 * recorded an empty event list and the counterfactual read as "the ability did
 * nothing" — the whole fork vocabulary across the sampled corpus was
 * {@code zone_change} and {@code counter_change}. And the zone reading itself
 * was wrong on both sides: one move rendered three events, one of them naming a
 * destination the card never reached.
 */
@ExtendWith(ForgeExtension.class)
class ForkEventSinkTest {

    /** The bus events a class subscribes to. */
    private static Set<Class<?>> subscriptions(Class<?> subscriber) {
        Set<Class<?>> types = new LinkedHashSet<>();
        for (Method method : subscriber.getDeclaredMethods()) {
            if (method.isAnnotationPresent(Subscribe.class)
                    && method.getParameterCount() == 1) {
                types.add(method.getParameterTypes()[0]);
            }
        }
        return types;
    }

    /**
     * Events that describe the shape of a game rather than an outcome.
     *
     * <p>A fork has no cast, no resolution it did not force, no phase and no
     * combat end: it is created, one ability is made to resolve, and it is
     * thrown away. {@link GameEventZone} is here for a different reason — it is
     * the per-zone-list notification that neither side reads any more.
     */
    private static final Set<Class<?>> STRUCTURAL = Set.of(
            GameEventSpellAbilityCast.class, GameEventSpellResolved.class,
            GameEventTurnPhase.class, GameEventCombatEnded.class,
            GameEventZone.class);

    /**
     * A fork describes an outcome the way an observed record does.
     *
     * <p>Asserted against the bracket collector rather than a fixed list, so the
     * two cannot drift apart again: adding an outcome event to the observed side
     * and forgetting the fork is exactly how the sink came to be missing
     * card_drawn, card_discarded, card_milled, pt_change, keyword_change,
     * type_change, color_change, attached, unattached, card_looked_at and
     * library_reordered — eleven of the seventeen types real resolutions
     * produce.
     */
    @Test
    void aForkHearsEveryOutcomeAnObservedRecordHears() {
        Set<Class<?>> observed = new LinkedHashSet<>(
                subscriptions(BusBracketCollector.class));
        observed.removeAll(STRUCTURAL);

        Set<Class<?>> forked = subscriptions(ForkEventSink.class);

        assertTrue(forked.containsAll(observed),
                "a fork would record nothing for " + missing(observed, forked));
    }

    private static Set<Class<?>> missing(Set<Class<?>> observed, Set<Class<?>> forked) {
        Set<Class<?>> gap = new LinkedHashSet<>(observed);
        gap.removeAll(forked);
        return gap;
    }

    /** And it no longer listens to the notification that fired three times. */
    @Test
    void aForkNoLongerReadsThePerZoneNotification() {
        assertFalse(subscriptions(ForkEventSink.class).contains(GameEventZone.class));
    }

    // ── one move, one event, both ends of it ────────────────────────────

    private static Zone zone(ZoneType type) {
        return new Zone(type, TestCards.game());
    }

    /**
     * A completed move says where the card came from as well as where it went.
     *
     * <p>{@code from_zone} is a documented parameter of the event type that
     * nothing had ever written, because the notification the collector read
     * names one zone and not a move.
     */
    @Test
    void aMoveNamesBothOfItsEnds() {
        Card card = TestCards.build("Mountain");

        EffectEvent moved = BusEvents.cardMoved(new GameEventCardChangeZone(
                card, zone(ZoneType.Hand), zone(ZoneType.Battlefield)));

        assertEquals(EffectEvent.ZONE_CHANGE, moved.type());
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
        assertEquals(null, BusEvents.cardMoved(
                new GameEventCardChangeZone(TestCards.build("Mountain"), null, null)));
    }
}
