package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The turn-based actions of one step, from the step's start until anyone gets
 * priority (C7, FR-036).
 *
 * <p>Forge publishes {@code GameEventTurnPhase} and then performs the step's
 * turn-based actions in {@code PhaseHandler.onPhaseBegin}, before anyone
 * gets priority: the untap step's phasing, day/night check and untap
 * ({@code Untap.executeAt}), the draw, the precombat main phase's lore
 * counters, the cleanup's discard to hand size. No ability is resolving, so
 * without a window of their own their events fall into the stack brackets'
 * orphans and go nowhere. The window opens at the phase event for those four
 * steps and closes at the next priority or phase event; nothing can be cast
 * between the two, so what it takes is the actions and their direct
 * consequences.
 *
 * <p>It takes only the events an action is made of, read off the step and the
 * event type, and writes one record per action that produced any, in the
 * order the actions began. What else the step publishes — a static's
 * recomputed power after a permanent phased, an "until your next turn"
 * effect ending — is no turn-based action of the six and stays with the
 * orphans, as before.
 *
 * <p>The board is read when the window opens, before the first action, as a
 * combat record's is before the first damage. Reading it costs a snapshot,
 * so a cleanup or a main phase in which no action can happen — no hand over
 * its maximum, no Saga — opens no window at all.
 */
final class TurnActionWindow {

    /** The steps that begin with a turn-based action this window collects. */
    private static final Set<PhaseType> STEPS = Set.of(
            PhaseType.UNTAP, PhaseType.DRAW, PhaseType.MAIN1, PhaseType.CLEANUP);

    private final Game game;
    private final CollectorContext context;
    private final SnapshotBuilder snapshots;

    /** The step whose actions the window takes, or null while it is closed. */
    private PhaseType step;
    /** The active player, as a ref: every turn-based action here is theirs. */
    private String actor;
    /** The board as the step began. */
    private Snapshot state;
    /** Each action's events, in the order the actions began. */
    private final Map<TurnActionSubkind, EventBuffer> actions = new LinkedHashMap<>();

    TurnActionWindow(Game game, CollectorContext context, SnapshotBuilder snapshots) {
        this.game = game;
        this.context = context;
        this.snapshots = snapshots;
    }

    /**
     * A step has begun: open the window if its turn-based actions are ones
     * this collects and can happen. Closes the last one first.
     */
    void open(PhaseType begun) {
        close();
        Player active = game.getPhaseHandler().getPlayerTurn();
        if (begun == null || active == null || !STEPS.contains(begun)
                || !mayAct(begun, active)) {
            return;
        }
        step = begun;
        actor = Refs.of(active);
        state = snapshots.ofBoard();
    }

    /**
     * Whether {@code step}'s action can happen at all, as {@code
     * PhaseHandler.onPhaseBegin} decides it: a cleanup discards only from a
     * hand over its maximum (CR 514.1), a main phase adds lore counters only
     * to Sagas with chapters (CR 714.3b). The untap and the draw are taken to
     * happen every turn.
     */
    private static boolean mayAct(PhaseType step, Player active) {
        return switch (step) {
            case CLEANUP -> !active.isUnlimitedHandSize()
                    && active.getZone(ZoneType.Hand).size() > active.getMaxHandSize();
            case MAIN1 -> active.getCardsIn(ZoneType.Battlefield).anyMatch(
                    card -> card.isSaga() && card.hasChapter());
            default -> true;
        };
    }

    /**
     * File {@code event} into the action it is part of, if the window is open
     * and it is part of one.
     *
     * @param namedCause the ref the bus named as the causer, or null
     * @return whether the window took it; one it did not take is the caller's
     *         to route
     */
    boolean take(EffectEvent event, String namedCause) {
        TurnActionSubkind action = step == null ? null : actionOf(step, event);
        if (action == null) {
            return false;
        }
        actions.computeIfAbsent(action, ignored -> new EventBuffer())
                .file(EventAttribution.stamp(event, null, namedCause));
        return true;
    }

    /**
     * Which of {@code step}'s turn-based actions {@code event} is part of, or
     * null for none.
     *
     * <p>Read by what the event is, since the actions of one step are told
     * apart by nothing else: the untap step's three actions publish a phasing,
     * a day/night change and an untap, the draw a card reaching the hand (a
     * replaced draw, dredge's return from the graveyard, included), the
     * cleanup's discard a card leaving the hand (to wherever madness sends
     * it), the main phase a lore counter.
     */
    static TurnActionSubkind actionOf(PhaseType step, EffectEvent event) {
        EventType type = event.type();
        return switch (step) {
            case UNTAP -> switch (type) {
                case PHASED -> TurnActionSubkind.PHASING;
                case DAY_NIGHT_CHANGED -> TurnActionSubkind.DAY_NIGHT;
                case UNTAPPED -> TurnActionSubkind.UNTAP;
                default -> null;
            };
            case DRAW -> type == EventType.CARD_DRAWN || movedTo(event, "hand")
                    ? TurnActionSubkind.DRAW : null;
            case CLEANUP -> type == EventType.CARD_DISCARDED || movedFrom(event, "hand")
                    ? TurnActionSubkind.CLEANUP_DISCARD : null;
            case MAIN1 -> type == EventType.COUNTER_CHANGE
                    && "LORE".equals(event.params().get("counter_type"))
                    ? TurnActionSubkind.LORE_COUNTER : null;
            default -> null;
        };
    }

    private static boolean movedTo(EffectEvent event, String zone) {
        return event.type() == EventType.ZONE_CHANGE && zone.equals(event.params().get("to_zone"));
    }

    private static boolean movedFrom(EffectEvent event, String zone) {
        return event.type() == EventType.ZONE_CHANGE
                && zone.equals(event.params().get("from_zone"));
    }

    /** Write one record per action the window took events for, and close it. */
    void close() {
        if (step == null) {
            return;
        }
        for (Map.Entry<TurnActionSubkind, EventBuffer> action : actions.entrySet()) {
            List<EffectEvent> events = action.getValue().drain();
            // A draw names the card it drew, which joins the board as the
            // library card it was when the step began.
            context.emit(context.newTurnAction(action.getKey())
                    .actor(actor)
                    .state(state.withEventCards(events).render())
                    .payload(RecordPayloads.eventsPayload(events)));
        }
        actions.clear();
        step = null;
        actor = null;
        state = null;
    }
}
