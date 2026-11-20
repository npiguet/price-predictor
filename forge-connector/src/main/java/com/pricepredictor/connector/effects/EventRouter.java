package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The routing classifier of a {@link BusBracketCollector}: what an event
 * published now is a consequence of ({@link #classifyNow}), read off the
 * engine, and the record that cause files it into ({@link #route}), the
 * taps asked first.
 *
 * <p>It holds the collector's collaborators an event can be filed into, the
 * same objects the collector holds, and one flag of its own the collector
 * sets: whether a resolving effect has handed the game to another step
 * ({@link #stepTakenOver}).
 */
final class EventRouter {

    /**
     * What the router reads and files into, shared with the collector that
     * owns them.
     *
     * @param payingForACast whether a player is paying, now, for a spell the
     *                       resolving ability casts ({@link
     *                       BusEventReader#payingForACast}), read at each call
     */
    record Collaborators(
            Game game, StackBrackets brackets, CastsUnderWay castsUnderWay,
            PaymentLedger payments, DroppedEvents dropped, CombatStepBracket combatStep,
            TurnActionWindow turnActions, Predicate<SpellAbility> payingForACast) {
    }

    private final Game game;
    private final StackBrackets brackets;
    private final CastsUnderWay castsUnderWay;
    private final PaymentLedger payments;
    private final DroppedEvents dropped;
    private final CombatStepBracket combatStep;
    private final TurnActionWindow turnActions;
    private final Predicate<SpellAbility> payingForACast;

    /** The taps asked, in order, before an event is routed. */
    private final List<EventTap> taps = new java.util.concurrent.CopyOnWriteArrayList<>();

    /**
     * Whether the game has moved on to another step inside the resolution
     * running now: an effect that ends the turn or the combat phase
     * (Time Stop's {@code EndTurn}, Mandate of Peace's {@code
     * EndCombatPhase}) has run its own state-based check and handed the game
     * to the step it skips to, whose own actions follow while the stack
     * still resolves. From here to the resolution's end what the bus
     * publishes is that step's, routed as in an ordinary step ({@link
     * #route}), and not the resolving spell's. Set at the end of the
     * effect's own check, and at a phase event that arrives while the stack
     * resolves where no loop signal came; cleared when a resolution starts
     * or ends, and at a priority.
     */
    private boolean stepTakenOver;

    EventRouter(Collaborators parts) {
        this.game = parts.game();
        this.brackets = parts.brackets();
        this.castsUnderWay = parts.castsUnderWay();
        this.payments = parts.payments();
        this.dropped = parts.dropped();
        this.combatStep = parts.combatStep();
        this.turnActions = parts.turnActions();
        this.payingForACast = parts.payingForACast();
    }

    /** Whether a resolving effect has handed the game to another step ({@link #stepTakenOver}). */
    boolean stepTakenOver() {
        return stepTakenOver;
    }

    /** Set or clear {@link #stepTakenOver}. */
    void setStepTakenOver(boolean taken) {
        stepTakenOver = taken;
    }

    /**
     * What an event published now is a consequence of, read off the engine
     * at the moment it is filed, which decides the record it joins.
     */
    enum Cause {
        /**
         * An ability is resolving: the resolving item's bracket, or no
         * bracket while Forge finishes one whose resolution has been
         * published ({@link StackBrackets#followResolving}).
         */
        RESOLUTION,
        /**
         * A cost is being paid — for something being cast or activated, or,
         * while an ability resolves, for a spell it casts
         * ({@link BusEventReader#payingForACast}). The paid ability's cost half records
         * what was paid, in its {@code costs}; the event goes nowhere and
         * is counted as dropped, the costs' channels (tapped, sacrificed,
         * discarded, exiled, life, mana) and the rest alike.
         */
        PAYMENT,
        /**
         * A cast is under way and not yet announced ({@link CastsUnderWay}):
         * the static check its card's move onto the stack set off ({@code
         * GameAction.changeZone} runs one before the cast is pushed), and the
         * one {@code MagicStack.push} runs after the push and before the cast
         * event. Reaches no record, and is counted as dropped: the cost half
         * records the card's move and the costs alone, and the effect half's
         * board, read at the resolution's start, already shows it.
         */
        CAST,
        /**
         * A player's action at priority other than a cast: a special action
         * — unmorph, foretell, plot, which {@code MagicStack.add} resolves
         * with no push and no cast event — a mana ability activated outside
         * any payment, or an activation being announced. Every one goes
         * through {@code ComputerUtil.handlePlayingSpellAbility}, which names
         * it to the stack. Nobody's record: the event goes nowhere and is
         * counted as dropped, and the action ends the last cast's
         * consequences, so the state-based check after it is nobody's either
         * ({@link BusBracketCollector#onAnyEvent}).
         */
        ACTION,
        /**
         * What follows a cast until the next action: the state-based check
         * after it, and the rest of {@code MagicStack.add} after the cast is
         * published. The open bracket's — the last cast's, while no action
         * or resolution has ended its consequences — or the orphans'; in a
         * combat damage step before anything used the stack, the combat's. A
         * land played and a turn-based action fall here too, and find no
         * bracket open: neither happens with anything on the stack. A
         * state-based death nothing held is never the open bracket's: the
         * combat's, or nobody's (FR-029).
         */
        CONSEQUENCE
    }

    /** What kind of event {@link #route} is filing, which narrows where it may go. */
    enum Filing {
        /** An outcome: in a combat damage step before anything used the stack, the combat's. */
        OUTCOME,
        /** A damage the bus says is not combat damage: never the combat's. */
        NON_COMBAT_DAMAGE,
        /**
         * A permanent leaving the battlefield in a state-based check while no
         * resolution is held ({@link BusBracketCollector#fileMove}): as a consequence, a combat
         * damage step's or nobody's, counted as dropped; never the open
         * bracket's, whose cast did not leave the board in that state
         * (FR-029).
         */
        STATE_BASED_DEATH,
        /**
         * What a resolving clause did, told at the clause's end: the resolving
         * item's even when the clause handed the game to another step
         * ({@link #stepTakenOver}) — Mandate of Peace's {@code combat_ended}
         * is told after the combat phase it ended is over.
         */
        CLAUSE
    }

    /**
     * The {@link Cause} of an event published now, with the card being cast
     * when the cause is {@link Cause#CAST}, or null.
     */
    record Now(Cause cause, Card casting) {
        Now(Cause cause) {
            this(cause, null);
        }
    }

    /**
     * What an event published now is a consequence of, with this collector's
     * view of the engine brought up to date first.
     *
     * <p>Called once at each entry point that classifies an event or a
     * refusal, before the open bracket is read: Forge publishes nothing when a
     * resolution starts, so the first thing it does is what opens its
     * bracket.
     */
    Now classifyNow() {
        syncWithEngine();
        return now();
    }

    /**
     * Bring this collector's view of the engine up to date, so that what
     * {@link #now} reads off it is current: a resolution that has begun is
     * followed ({@link StackBrackets#followResolving}), and the marks of the
     * casts under way are kept ({@link CastsUnderWay#keepMarksCurrent}). The
     * two halves run under exactly the conditions {@link #now} reads them
     * under.
     */
    private void syncWithEngine() {
        if (game.getStack().isResolving()) {
            brackets.followResolving(this::idsOnStack);
        } else if (game.costPaymentStack.peek() == null) {
            castsUnderWay.keepMarksCurrent(StackActions.inProgress(game.getStack()));
        }
    }

    /**
     * What an event published now is a consequence of.
     *
     * <p>Read off the engine's own state, in this order: the stack resolving;
     * {@code game.costPaymentStack}, which {@code CostPayment} keeps for every
     * part being paid; a cast under way; the ability {@code
     * MagicStack.freezeStack} names as the action in progress ({@link
     * StackActions}). What is left — a state-based check, which freezes the
     * stack naming no action, or nothing at all — is a consequence of what
     * came before.
     *
     * <p>A pure read, of what {@link #syncWithEngine} has brought up to date.
     */
    private Now now() {
        if (game.getStack().isResolving()) {
            return new Now(
                    payingForACast.test(brackets.resolving()) ? Cause.PAYMENT : Cause.RESOLUTION);
        }
        if (game.costPaymentStack.peek() != null) {
            return new Now(Cause.PAYMENT);
        }
        SpellAbility acting = StackActions.inProgress(game.getStack());
        Card casting = castsUnderWay.cardBeingCast(acting);
        if (casting != null) {
            return new Now(Cause.CAST, casting);
        }
        if (acting != null) {
            return new Now(brackets.isOpenCast(acting) ? Cause.CONSEQUENCE : Cause.ACTION);
        }
        return new Now(Cause.CONSEQUENCE);
    }

    /** Whether what has this cause belongs with the open bracket, if one is open. */
    static boolean fromOpenBracket(Cause cause) {
        return cause == Cause.RESOLUTION || cause == Cause.CONSEQUENCE;
    }

    /** The ability ids the stack holds right now. */
    Set<Integer> idsOnStack() {
        Set<Integer> onStack = new LinkedHashSet<>();
        for (var instance : game.getStack()) {
            SpellAbility sa = instance.getSpellAbility();
            if (sa != null) {
                onStack.add(sa.getId());
            }
        }
        return onStack;
    }

    /**
     * File an event where its cause puts it.
     *
     * <p>Stamped here rather than where the event is built: which clause
     * produced it, how long it lasts and — absent a name off the event
     * itself — what caused it are properties of the bracket it landed in,
     * and only this side knows that.
     *
     * <p>An event that arrives while no record is open — at a player's action
     * at priority, for the orphans, or a state-based death nothing takes — is
     * counted as dropped ({@link DroppedEvents}); an orphan once it is
     * discarded, at the next cast, resolution end or the game's end ({@link
     * StackBrackets}). The game's outcome is never routed here ({@link
     * BusBracketCollector#fileGameOutcome}).
     *
     * @param filing what kind of event it is, which narrows where it may go
     */
    void route(EffectEvent event, String namedCause, Filing filing) {
        for (EventTap tap : taps) {
            if (tap.take(event, namedCause)) {
                return;
            }
        }
        Now now = classifyNow();
        switch (now.cause()) {
            // What a payment publishes: the paid ability's cost half holds
            // its costs, never an event, so the event reaches no record
            // (the counters and life it moved are kept for the costs).
            case PAYMENT -> filePayment(event);
            // What a card's move onto the stack set off: after the cost
            // half's board, which records only the move and the costs, and
            // before the effect half's (FR-042a, FR-042b).
            case CAST -> dropped.count(event.type(), DroppedEvents.phaseOf(game));
            case ACTION -> {
                brackets.closeForAction();
                dropped.count(event.type(), DroppedEvents.phaseOf(game));
            }
            case CONSEQUENCE -> fileConsequence(event, namedCause, filing);
            case RESOLUTION -> fileResolving(event, namedCause, filing);
        }
    }

    /** {@link #route}'s {@link Cause#PAYMENT} case. */
    private void filePayment(EffectEvent event) {
        if (!game.getStack().isResolving()) {
            brackets.closeForAction();
        }
        payments.note(event);
        dropped.count(event.type(), DroppedEvents.phaseOf(game));
    }

    /** {@link #route}'s {@link Cause#CONSEQUENCE} case. */
    private void fileConsequence(EffectEvent event, String namedCause, Filing filing) {
        if (filing != Filing.NON_COMBAT_DAMAGE && combatStep.takesEventsNow()) {
            combatStep.file(
                    EventAttribution.stamp(event, brackets.resolving(), namedCause));
        } else if (filing == Filing.STATE_BASED_DEATH) {
            dropped.count(event.type(), DroppedEvents.phaseOf(game));
        } else if (!turnActions.take(event, namedCause)) {
            brackets.file(
                    EventAttribution.stamp(event, brackets.resolving(), namedCause),
                    DroppedEvents.phaseOf(game));
        }
    }

    /** {@link #route}'s {@link Cause#RESOLUTION} case. */
    private void fileResolving(EffectEvent event, String namedCause, Filing filing) {
        if (stepTakenOver && filing != Filing.CLAUSE) {
            fileAsTheStepsOwn(event, namedCause, filing);
        } else {
            brackets.file(EventAttribution.stamp(event, brackets.resolving(),
                            brackets.lastClauseRan(), namedCause),
                    DroppedEvents.phaseOf(game));
        }
    }

    /**
     * An event of the step a resolving effect handed the game to ({@link
     * #stepTakenOver}), routed as the same event of an ordinary step with no
     * bracket open: a turn-based action's into its window (the cleanup's
     * discard to hand size, a {@code cleanup_discard} record), a state-based
     * death with nothing holding it counted as dropped (FR-029), and any
     * other — damage wearing off, an "until end of turn" or "until end of
     * combat" effect ending — to the orphans, which count it as dropped when
     * they are discarded, as an ordinary cleanup's are.
     */
    private void fileAsTheStepsOwn(EffectEvent event, String namedCause, Filing filing) {
        if (filing == Filing.STATE_BASED_DEATH) {
            dropped.count(event.type(), DroppedEvents.phaseOf(game));
        } else if (!turnActions.take(event, namedCause)) {
            brackets.fileOrphan(
                    EventAttribution.stampAt(event, null, null, namedCause),
                    DroppedEvents.phaseOf(game));
        }
    }

    /**
     * A channel that takes, before they are routed, the events of a moment it
     * records itself: a buffered damage ability's run ({@link RewriteChannel}),
     * a mana ability's own activation ({@link ManaChannel}).
     */
    interface EventTap {
        /** Whether the event, published now, is this tap's; it is then routed nowhere else. */
        boolean take(EffectEvent event, String namedCause);
    }

    /** Ask {@code tap} first, before routing any event. */
    void addTap(EventTap tap) {
        taps.add(tap);
    }
}
