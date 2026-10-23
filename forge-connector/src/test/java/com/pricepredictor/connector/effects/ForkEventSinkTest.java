package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventGameFinished;
import forge.game.event.GameEventLandPlayed;
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
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a fork hears, and how a move is read.
 *
 * <p>Two properties meet here. A fork hears every bus event the observed
 * records use, so a forked draw or pump records what it did rather than an
 * empty event list that reads as "the ability did nothing". And a move is read
 * once, from the event that carries both ends, rather than as three per-zone
 * events, one of them naming a destination the card never reached.
 */
@ExtendWith(ForgeExtension.class)
class ForkEventSinkTest {

    /**
     * The bus events a class subscribes to, its superclasses' included, the
     * way Forge's bus finds them.
     */
    private static Set<Class<?>> subscriptions(Class<?> subscriber) {
        Set<Class<?>> types = new LinkedHashSet<>();
        for (Class<?> type = subscriber; type != null; type = type.getSuperclass()) {
            for (Method method : type.getDeclaredMethods()) {
                if (method.isAnnotationPresent(Subscribe.class)
                        && method.getParameterCount() == 1) {
                    types.add(method.getParameterTypes()[0]);
                }
            }
        }
        return types;
    }

    /**
     * Events that describe the shape of a game rather than an outcome.
     *
     * <p>A fork has no cast, no resolution it did not force, no phase, no
     * combat end, no land played and no end of the game: it is created, one
     * ability is made to resolve, and it is thrown away. The observed side
     * hears a land played only as the boundary that writes a held resolution.
     * {@link GameEventZone} is here for a different reason — it is the
     * per-zone-list notification that neither side reads any more.
     */
    private static final Set<Class<?>> STRUCTURAL = Set.of(
            GameEventSpellAbilityCast.class, GameEventSpellResolved.class,
            GameEventTurnPhase.class, GameEventCombatEnded.class,
            GameEventGameFinished.class, GameEventLandPlayed.class,
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

    /** And it does not listen to the per-zone notification, which fires three times per move. */
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
        ForkEventSink sink = ForkEventSink.forIntervention(null, acting);
        sink.onCardChangeZone(new GameEventCardChangeZone(
                TestCards.build("Mountain"),
                zone(ZoneType.Hand), zone(ZoneType.Battlefield)));
        assertFalse(sink.events().isEmpty(), "the move was not filed at all");
        return sink.events().get(0);
    }

    /**
     * A fork's event answers "which clause did this" the way an observed one does.
     *
     * <p>{@code attributed_to} is a tri-state on the observed path — a
     * sub-ability acted, the root line acted, or no pointer was available — and
     * a sink that named none of the three would leave every event on every fork
     * record silently null, unable to answer a question a real one answers.
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

    // ── damage reads as the observed side reads it ──────────────────────

    /** One card damage, dealt by {@code source}, pushed through a sink. */
    private static String filedDamage(ForkEventSink sink, Card source) {
        sink.onCardDamaged(new GameEventCardDamaged(
                TestCards.build("Grizzly Bears").getView(), source.getView(), 2,
                GameEventCardDamaged.DamageType.Normal));
        return sink.events().get(0).toJson();
    }

    /**
     * A forced resolution's damage is not combat damage, and names the
     * permanent the bus says dealt it, as the observed record does.
     */
    @Test
    void anInterventionsDamageIsNotCombatAndNamesItsSource() {
        Card source = TestCards.build("Prodigal Pyromancer");

        String json = filedDamage(ForkEventSink.forIntervention(null, null), source);

        assertTrue(json.contains("\"combat\":false"), json);
        assertTrue(json.contains("\"cause\":\"E" + source.getId() + "\""), json);
    }

    /** A damage-step probe deals combat damage, and says so. */
    @Test
    void aProbesDamageIsCombatDamage() {
        Card source = TestCards.build("Grizzly Bears");

        String json = filedDamage(ForkEventSink.forProbe(null), source);

        assertTrue(json.contains("\"combat\":true"), json);
        assertTrue(json.contains("\"cause\":\"E" + source.getId() + "\""), json);
    }

    // ── a repeat is read as the observed side reads it ──────────────────

    private static List<String> types(ForkEventSink sink) {
        List<String> types = new ArrayList<>();
        for (EffectEvent event : sink.events()) {
            types.add(event.type());
        }
        return types;
    }

    /**
     * A state transition the engine publishes twice is filed once, as the
     * bracket collector files it, so a fork and an observed record of the
     * same outcome carry the same events.
     */
    @Test
    void aRepeatedTransitionIsFiledOnce() {
        ForkEventSink sink = ForkEventSink.forIntervention(null, null);
        Card bears = TestCards.build("Grizzly Bears");

        sink.onTapped(new GameEventCardTapped(bears, true));
        sink.onTapped(new GameEventCardTapped(bears, true));

        assertEquals(List.of(EffectEvent.TAPPED), types(sink));
    }

    /**
     * Repeats are recognised within one chosen mode, as the live bracket
     * recognises them within one mode's half: a transition repeated by the
     * next mode is that mode's own event and is kept. What is filed before
     * the first mode's clause shares the first mode's half, as it does live.
     */
    @Test
    void aRepeatIsRecognisedWithinOneModeOnly() {
        ForkEventSink sink = ForkEventSink.forIntervention(null, null);
        Card bears = TestCards.build("Grizzly Bears");
        GameEventCardTapped tap = new GameEventCardTapped(bears, true);

        sink.onTapped(tap);
        assertEquals(1, sink.beginMode());
        sink.onTapped(tap);
        assertEquals(1, sink.beginMode());
        sink.onTapped(tap);
        sink.onTapped(tap);

        assertEquals(List.of(EffectEvent.TAPPED, EffectEvent.TAPPED), types(sink));
    }

    /** A quantitative repeat is two occurrences, and both are kept. */
    @Test
    void aRepeatedDamageIsKept() {
        ForkEventSink sink = ForkEventSink.forIntervention(null, null);
        Card bears = TestCards.build("Grizzly Bears");
        Card source = TestCards.build("Prodigal Pyromancer");

        for (int i = 0; i < 2; i++) {
            sink.onCardDamaged(new GameEventCardDamaged(bears.getView(), source.getView(), 1,
                    GameEventCardDamaged.DamageType.Normal));
        }

        assertEquals(List.of(EffectEvent.DAMAGE_DEALT, EffectEvent.DAMAGE_DEALT), types(sink));
    }

    /**
     * A state-based leave's position names the event that was filed.
     *
     * <p>A repeat dropped before the leave shifts it down, and a repeat of the
     * leave itself is no position at all: marked anyway, it would point at
     * whatever is filed next, here an ordinary tap, which a modal fork would
     * then file by the death rule.
     */
    @Test
    void aStateBasedLeaveIsPlacedAmongTheEventsFiled() {
        Game game = TestCards.game();
        ForkEventSink sink = ForkEventSink.forIntervention(game, null);
        Card bears = TestCards.build("Grizzly Bears");
        Card elves = TestCards.build("Llanowar Elves");
        GameEventCardChangeZone dies = new GameEventCardChangeZone(
                bears, zone(ZoneType.Battlefield), zone(ZoneType.Graveyard));

        sink.onTapped(new GameEventCardTapped(bears, true));
        sink.onTapped(new GameEventCardTapped(bears, true));
        game.getTracker().freeze();
        try {
            sink.onCardChangeZone(dies);
            sink.onCardChangeZone(dies);
        } finally {
            game.getTracker().unfreeze();
        }
        sink.onTapped(new GameEventCardTapped(elves, true));

        assertEquals(List.of(EffectEvent.TAPPED, EffectEvent.ZONE_CHANGE, EffectEvent.TAPPED),
                types(sink));
        assertEquals(Set.of(1), sink.stateBasedLeaves());
    }
}
