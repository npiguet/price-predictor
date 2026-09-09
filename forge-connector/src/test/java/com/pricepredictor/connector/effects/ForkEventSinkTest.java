package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventGameFinished;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.event.GameEventZone;
import forge.game.spellability.SpellAbility;
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
     * <p>A fork has no cast, no resolution it did not force, no phase, no
     * combat end and no end of the game: it is created, one ability is made to
     * resolve, and it is thrown away. {@link GameEventZone} is here for a
     * different reason — it is the per-zone-list notification that neither side
     * reads any more.
     */
    private static final Set<Class<?>> STRUCTURAL = Set.of(
            GameEventSpellAbilityCast.class, GameEventSpellResolved.class,
            GameEventTurnPhase.class, GameEventCombatEnded.class,
            GameEventGameFinished.class, GameEventZone.class);

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

    // ── a fork event says where it came from ────────────────────────────

    /** Push one completed move through a sink and read back what it filed. */
    private static EffectEvent filedMove(SpellAbility acting) {
        ForkEventSink sink = new ForkEventSink(null, acting);
        sink.onCardChangeZone(new GameEventCardChangeZone(
                TestCards.build("Mountain"),
                zone(ZoneType.Hand), zone(ZoneType.Battlefield)));
        assertFalse(sink.isEmpty(), "the move was not filed at all");
        return sink.events().get(0);
    }

    /**
     * A fork's event answers "which clause did this" the way an observed one does.
     *
     * <p>{@code attributed_to} is a tri-state on the observed path — a
     * sub-ability acted, the root line acted, or no pointer was available — and
     * this sink named none of the three, so every event on every fork record
     * was silently null. Measured across the smoke corpus that was most of the
     * 30.9% of events with no attribution at all, and it made a fork record
     * unable to answer a question a real one answers.
     *
     * <p>{@code unresolved} is the honest answer here: nothing is resolving in
     * a unit test, and the point is that the field is <b>written</b> rather than
     * left null.
     */
    @Test
    void aForkEventNamesWhereItCameFrom() {
        String json = filedMove(null).toJson();

        assertFalse(json.contains("\"attributed_to\":null"), json);
        assertTrue(json.contains("\"attributed_to\":\"unresolved\""), json);
        assertTrue(json.contains("\"duration\":\"instant\""), json);
    }

    /**
     * And the acting line the fork forced is the line it reads against.
     *
     * <p>Duration is the half of the stamp a unit test can steer without a
     * resolving thread: it falls back to the root line's own {@code Duration$},
     * so a permanent-duration line reaching the sink is visible in the event it
     * files. A sink that ignored its acting line would say {@code instant}.
     */
    @Test
    void theForcedLineIsTheOneAForkEventIsReadAgainst() {
        SpellAbility pump = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | Defined$ Self | NumAtt$ 2 | NumDef$ 2"
                        + " | Duration$ Permanent",
                TestCards.build("Fountain of Youth"));

        String json = filedMove(pump).toJson();

        assertTrue(json.contains("\"duration\":\"permanent\""), json);
    }
}
