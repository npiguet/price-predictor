package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.EventValueChangeType;
import forge.game.event.GameEvent;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventGameFinished;
import forge.game.event.GameEventLandPlayed;
import forge.game.event.GameEventPlayerPriority;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.event.GameEventZone;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
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
 * What a damage-step probe's fork hears, and how a move is read.
 *
 * <p>Two properties meet here. A probe hears every bus event the observed
 * records use, so its damage step records what it did rather than an empty
 * event list that reads as "nothing happened". And a move is read
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
     * <p>A probe's fork has no cast, no resolution, no phase, no combat
     * end, no land played, no priority, no player action at priority and no
     * end of the game: it is created, one damage step is dealt, and it is
     * thrown away. The observed side hears a land played only as
     * the boundary that writes a held resolution, a priority only as a
     * boundary no cast is under way across, and any event at all only to
     * notice that a player's action has begun ({@link GameEvent}).
     * {@link GameEventZone} is here for a different reason — it is the
     * per-zone-list notification that neither side reads any more.
     */
    private static final Set<Class<?>> STRUCTURAL = Set.of(
            GameEventSpellAbilityCast.class, GameEventSpellResolved.class,
            GameEventTurnPhase.class, GameEventCombatEnded.class,
            GameEventGameFinished.class, GameEventLandPlayed.class,
            GameEventPlayerPriority.class, GameEvent.class, GameEventZone.class);

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

    /**
     * And it reads no move off the per-zone notification, which fires three
     * times per move: that notification is read for a player's command-zone
     * marks alone (the city's blessing, a completed dungeon), and a
     * notification of any other zone files nothing.
     */
    @Test
    void aForkReadsNoMoveOffThePerZoneNotification() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, you, "Grizzly Bears");
        ForkEventSink sink = ForkEventSink.forProbe(game);

        sink.onZone(new GameEventZone(ZoneType.Battlefield, you, EventValueChangeType.Added, bears));
        sink.onZone(new GameEventZone(ZoneType.Hand, you, EventValueChangeType.Removed, bears));

        assertTrue(sink.events().isEmpty(), sink.events().toString());
    }

    // ── a fork event says where it came from ────────────────────────────

    private static Zone zone(ZoneType type) {
        return new Zone(type, TestCards.game());
    }

    /** Push one completed move through a sink and read back what it filed. */
    private static EffectEvent filedMove() {
        ForkEventSink sink = ForkEventSink.forProbe(null);
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
        String json = filedMove().toJson();

        assertFalse(json.contains("\"attributed_to\":null"), json);
        assertTrue(json.contains("\"attributed_to\":\"unresolved\""), json);
        assertTrue(json.contains("\"duration\":\"instant\""), json);
    }

    // ── damage reads as the observed side reads it ──────────────────────

    /** One card damage, dealt by {@code source}, pushed through a sink. */
    private static String filedDamage(ForkEventSink sink, Card source) {
        sink.onCardDamaged(new GameEventCardDamaged(
                TestCards.build("Grizzly Bears").getView(), source.getView(), 2,
                GameEventCardDamaged.DamageType.Normal));
        return sink.events().get(0).toJson();
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

    private static List<EventType> types(ForkEventSink sink) {
        List<EventType> types = new ArrayList<>();
        for (EffectEvent event : sink.events()) {
            types.add(event.type());
        }
        return types;
    }

    /** A transition repeated within the probe's damage step is filed once, as the observed combat bracket files it. */
    @Test
    void aRepeatIsRecognisedWithinTheStep() {
        ForkEventSink sink = ForkEventSink.forProbe(null);
        GameEventCardTapped tap = new GameEventCardTapped(TestCards.build("Grizzly Bears"), true);

        sink.onTapped(tap);
        sink.onTapped(tap);

        assertEquals(List.of(EventType.TAPPED), types(sink));
    }

    /** A quantitative repeat is two occurrences, and both are kept. */
    @Test
    void aRepeatedDamageIsKept() {
        ForkEventSink sink = ForkEventSink.forProbe(null);
        Card bears = TestCards.build("Grizzly Bears");
        Card source = TestCards.build("Prodigal Pyromancer");

        for (int i = 0; i < 2; i++) {
            sink.onCardDamaged(new GameEventCardDamaged(bears.getView(), source.getView(), 1,
                    GameEventCardDamaged.DamageType.Normal));
        }

        assertEquals(List.of(EventType.DAMAGE_DEALT, EventType.DAMAGE_DEALT), types(sink));
    }

    /**
     * A probe's state-based leave is one of its damage step's events, in the
     * order it came, and a repeat of it is filed once, as the observed combat
     * bracket files it: a damage step is one record.
     */
    @Test
    void aProbesStateBasedLeaveIsAnOrdinaryEventOfItsStep() {
        Game game = TestCards.game();
        ForkEventSink sink = ForkEventSink.forProbe(game);
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

        assertEquals(List.of(EventType.TAPPED, EventType.ZONE_CHANGE, EventType.TAPPED),
                types(sink));
    }
}
