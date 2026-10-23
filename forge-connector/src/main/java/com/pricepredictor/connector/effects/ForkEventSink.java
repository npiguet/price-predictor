package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * Everything a forked resolution did, in order.
 *
 * <p>The bracket collector sorts events into a resolution bracket or a combat
 * one, because a real game interleaves them. A fork is simpler: it is created,
 * one ability is forced to resolve or one damage step is dealt, and it is
 * thrown away. A probe's damage step is the whole fork, so a probe files every
 * event. A forced resolution shares its fork with whatever else resolves while
 * {@code GameSimulator.resolveStack} drains the stack, so an intervention files
 * only what the live bracket would give the forced ability ({@link Phase}).
 *
 * <p>The events are read by {@link BusEventReader}, the same reader the bracket
 * collector uses, and filed into an {@link EventBuffer}, split by {@link
 * ModeSplit} and held by {@link HeldHalves}, as its brackets are, so an
 * interventional or probe record and an observed one describe the same
 * outcome the same way: the same subscriptions, the same shaping, the same
 * cause the bus names for a damage, a poison or a radiation counter, a state
 * transition repeated within one record — the whole fork, or one chosen
 * mode's half — filed once, and a death joining the half the live rule puts
 * it in.
 *
 * <p>Every event is stamped by {@link EventAttribution} on its way in, for the
 * same reason: {@code attributed_to} and {@code duration} are how a reader asks
 * which clause of the line did this and how long it lasts, and a fork that
 * answered null to both was not describing the outcome the same way after all.
 */
final class ForkEventSink extends BusEventReader {

    /**
     * The line whose resolution is being watched, or null when nothing resolves.
     *
     * <p>An interventional fork forces one ability, and that ability is the
     * root the sub-ability pointer is read against. A damage-step probe forces
     * no ability at all: it deals combat damage directly, so there is no root,
     * and its events attribute exactly the way the observed combat record's do
     * — from the pointer alone, which for a turn-based action names nothing.
     * For an intervention it is the ability handed to the cast until the cast
     * is over, and then the one the cast put on the stack ({@link #castAs}).
     */
    private SpellAbility acting;
    /** The board at the forced cast, which a resolution no mode of which began is read against. */
    private Snapshot castState;
    /**
     * Whether a card damaged in this fork is damaged in combat: true for a
     * damage-step probe, which deals one combat damage step, and false for an
     * intervention, whose forced resolution is not one. The flag is what a
     * reader uses to tell the two apart, and a probe's damage is compared
     * against the observed combat record's.
     */
    private final boolean combatDamage;
    /**
     * The events of the open span: the whole fork, or the chosen mode now
     * resolving, which is the unit a repeat is recognised within, as it is
     * a mode's half on the live side.
     */
    private final EventBuffer open = new EventBuffer();
    /** The forced resolution's chosen modes, split as they resolve. */
    private final ModeSplit modes = new ModeSplit();
    /**
     * The forced resolution's effect halves once it has resolved, which the
     * deaths it left behind join; null until then.
     */
    private HeldHalves<ModeSplit.Mode> held;

    /**
     * Where the forced ability stands, which decides what an intervention
     * files: what the live bracket would put in the forced ability's record,
     * and nothing else the drained stack does.
     *
     * <ul>
     *   <li>{@link #PRE_CAST}: from before the forced cast moves its card to
     *   the stack until its cast event, everything is filed — the static check
     *   the move and the push set off — as live gives what a card's move onto
     *   the stack sets off to that cast's bracket. The fork pays no cost, so
     *   there is no payment to leave out.</li>
     *   <li>{@link #BEFORE}: between the cast and the resolution, what is
     *   published while the forced ability is the top of the fork's stack is
     *   filed, as live files the state-based check before priority into the
     *   spell's bracket. While anything sits above it — a cast trigger:
     *   prowess, a "whenever you cast" or becomes-target trigger, storm,
     *   cascade — nothing is: that is the item's own record live. What was
     *   filed before an item was pushed above is kept.</li>
     *   <li>{@link #ABOVE_AFTERMATH}: as {@link #BEFORE}, but an item above
     *   the forced ability has resolved, and until the next ability reaches
     *   the stack or resolves, nothing is filed, though the forced ability is
     *   the top again: live keeps the deaths for the item's held half and
     *   sends the rest to no record, since none of it belongs to an ability
     *   that has not begun to resolve.</li>
     *   <li>{@link #RESOLVING}: from the first event published while the
     *   forced ability is the one resolving, until its {@code
     *   GameEventSpellResolved}: everything is filed, as live files the open
     *   bracket's events, except the payment for a spell the forced ability
     *   casts — a cascaded spell's additional cost, a madness cast's mana —
     *   which is that spell's cost, as live reads it ({@link
     *   BusEventReader#payingForACast}). A permanent leaving the battlefield
     *   while the tracker is frozen mid-resolution — an end-the-turn effect's own
     *   state-based check, a token or counter batch — is filed as an
     *   ordinary event, checked for a repeat in its span, as live files it
     *   into the open bracket.</li>
     *   <li>{@link #HELD}: only the permanents leaving the battlefield in a
     *   state-based check, as live holds a resolved half for the deaths it
     *   caused. A stat change a static makes once the ability is gone, or the
     *   spell's own card going to the graveyard, is not the ability's.</li>
     *   <li>{@link #CLOSED}: once the next ability reaches the stack or
     *   resolves, as the next action closes live's held window. A trigger the
     *   resolution caused is that next ability, and has a record of its own
     *   live.</li>
     * </ul>
     */
    private enum Phase { PRE_CAST, BEFORE, ABOVE_AFTERMATH, RESOLVING, HELD, CLOSED }

    private Phase phase = Phase.BEFORE;

    private ForkEventSink(Game fork, SpellAbility acting, boolean combatDamage) {
        super(fork);
        this.acting = acting;
        this.combatDamage = combatDamage;
        primeBattlefield();
    }

    /**
     * A sink for a forced resolution of {@code ability}, subscribed before
     * the forced cast so its pre-push window is filed ({@link Phase#PRE_CAST});
     * {@link #castAs} names the ability the cast put on the stack.
     */
    static ForkEventSink forCast(Game fork, SpellAbility ability) {
        ForkEventSink sink = new ForkEventSink(fork, ability, false);
        sink.phase = Phase.PRE_CAST;
        return sink;
    }

    /**
     * The forced cast is over: {@code onStack} is the ability it put on the
     * stack — the one handed to it for a spell, a copy for an activation —
     * and the root everything after is read against.
     *
     * @param state the board at the cast, the moment the live bracket reads
     *              an observed cast's: what a resolution no mode of which
     *              began is read against
     */
    void castAs(SpellAbility onStack, Snapshot state) {
        acting = onStack;
        castState = state;
    }

    /** A sink for a damage step dealt in a fork, with no ability resolving. */
    static ForkEventSink forProbe(Game fork) {
        return new ForkEventSink(fork, null, true);
    }

    /** Every event filed, in order: the halves', then the open span's. */
    List<EffectEvent> events() {
        List<EffectEvent> all = new ArrayList<>();
        if (held != null) {
            for (HeldHalves.Half<ModeSplit.Mode> half : held.halves()) {
                all.addAll(half.events());
            }
        } else {
            for (ModeSplit.ModeHalf half : modes.complete()) {
                all.addAll(half.events());
            }
        }
        all.addAll(open.events());
        return List.copyOf(all);
    }

    /**
     * The forced resolution's effect halves, one per chosen mode or one for
     * the whole resolution, with the deaths it left behind filed in.
     *
     * <p>Held at the forced ability's resolution, and here for a fork whose
     * forced ability never published one, so that what it filed is still
     * divided as it would have been.
     */
    List<HeldHalves.Half<ModeSplit.Mode>> halves() {
        return hold().halves();
    }

    /**
     * A chosen mode's first clause is resolving: close the previous mode's
     * half, so a repeat is recognised within one mode, as the live bracket
     * recognises it within one mode's half (FR-029a).
     *
     * <p>The first mode closes nothing: what was filed before its clause
     * shares its half, as it shares the first half live.
     *
     * @param option the mode's {@code Choices$} position, or null
     * @param state  the fork's board at that clause (FR-029b), read against
     *               the forced ability on the fork's stack as the live
     *               bracket's {@code beginMode} reads it
     */
    void beginMode(Integer option, Snapshot state) {
        modes.begin(option, state, open);
    }

    /**
     * The forced resolution is over: its halves are fixed, and from here only
     * the deaths it left behind join them.
     */
    private HeldHalves<ModeSplit.Mode> hold() {
        if (held == null) {
            // The last mode's half closes with the resolution, since no next
            // mode's first clause comes to close it.
            modes.close(open);
            held = new HeldHalves<>();
            for (ModeSplit.ModeHalf half : modes.halves(castState, open.drain())) {
                held.add(half.mode(), half.events());
            }
        }
        return held;
    }

    @Override
    boolean cardDamageIsCombat() {
        return combatDamage;
    }

    @Override
    void fileDamage(EffectEvent damage, String namedCause, boolean combat) {
        fileOutcome(damage, namedCause);
    }

    /**
     * File one event, attributed.
     *
     * <p>The single door in, so no event can miss the stamp. The bracket
     * collector's {@code fileOutcome} is the sibling of this.
     */
    @Override
    void fileOutcome(EffectEvent event, String namedCause) {
        if (admits(null)) {
            file(event, namedCause);
        }
    }

    /**
     * An ability has finished resolving: the forced one, which opens the held
     * window, or the next one, which closes it.
     */
    @Subscribe
    public void onResolved(GameEventSpellResolved event) {
        if (!gated()) {
            return;
        }
        boolean forced = event.spell() != null && event.spell().getId() == acting.getId();
        if (forced && (phase == Phase.BEFORE || phase == Phase.ABOVE_AFTERMATH
                || phase == Phase.RESOLVING)) {
            phase = Phase.HELD;
            hold();
        } else if (phase == Phase.HELD) {
            phase = Phase.CLOSED;
        } else if (phase == Phase.BEFORE || phase == Phase.ABOVE_AFTERMATH) {
            phase = Phase.ABOVE_AFTERMATH;
        }
    }

    /** The next ability has reached the stack, which closes a held window. */
    @Subscribe
    public void onCast(GameEventSpellAbilityCast event) {
        if (!gated()) {
            return;
        }
        if (phase == Phase.PRE_CAST || phase == Phase.ABOVE_AFTERMATH) {
            // The forced cast's own, the first one the fork publishes; or an
            // ability going above the forced one, which ends the aftermath
            // of the item that resolved there.
            phase = Phase.BEFORE;
        } else if (phase == Phase.HELD) {
            phase = Phase.CLOSED;
        }
    }

    /**
     * Whether an event published now is the forced ability's, by {@link Phase}.
     *
     * @param move the move the event is, or null for any other event
     */
    private boolean admits(GameEventCardChangeZone move) {
        if (!gated()) {
            return true;
        }
        noticeResolutionStart();
        return switch (phase) {
            case PRE_CAST -> true;
            case BEFORE -> forcedOnTop();
            // Everything but the payment for a spell the forced ability
            // casts, which is that spell's cost and which live leaves out too.
            case RESOLVING -> !payingForACast(acting);
            case HELD -> move != null && isStateBasedLeave(move);
            case ABOVE_AFTERMATH, CLOSED -> false;
        };
    }

    /**
     * The forced ability has begun to resolve if it is the top of a stack
     * that is resolving: {@link Phase#RESOLVING} from here.
     *
     * <p>Noticed at the first event published after it begins, because Forge
     * publishes nothing when a resolution starts.
     */
    private void noticeResolutionStart() {
        if ((phase == Phase.BEFORE || phase == Phase.ABOVE_AFTERMATH)
                && game.getStack().isResolving() && forcedOnTop()) {
            phase = Phase.RESOLVING;
        }
    }

    /**
     * Whether this sink files by {@link Phase}: an intervention over a game
     * whose stack it can read. A probe's damage step is the whole fork, and a
     * sink with no game or no forced line has nothing to gate on.
     */
    private boolean gated() {
        return !combatDamage && acting != null && game != null;
    }

    /**
     * Whether the forced ability is the top of the fork's stack. While the
     * stack is resolving, that makes it the one resolving: {@code
     * MagicStack.resolveStack} leaves it on top until it has finished. Read
     * in {@link Phase#RESOLVING} no longer; anything pushed above it
     * mid-resolution does not end its span, the resolved event does.
     */
    private boolean forcedOnTop() {
        var top = game.getStack().peek();
        return top != null && top.getSpellAbility() != null
                && top.getSpellAbility().getId() == acting.getId();
    }

    /**
     * A death the forced resolution left behind joins the half the live rule
     * puts it in ({@link HeldHalves#fileDeath}), and is checked there for a
     * repeat, as the live collector checks a death against the held half it
     * files it into.
     *
     * <p>Every mode's half is complete by then: {@code
     * GameSimulator.resolveStack} runs the state-based check only once the
     * whole charm has resolved and published its {@code
     * GameEventSpellResolved}, which is what holds the halves.
     *
     * <p>Only an intervention's death in {@link Phase#HELD} joins a half that
     * way. One in the check before the forced ability resolves, or one while
     * the tracker is frozen mid-resolution, is filed as an ordinary event,
     * checked for a repeat in its span: live files either into the open
     * bracket like any other event, so on a modal resolution it rides with
     * the mode in whose span it fell. A probe's is ordinary too: a damage
     * step is one record, as the observed combat bracket is.
     */
    @Override
    void fileMove(GameEventCardChangeZone event, EffectEvent moved) {
        if (!admits(event)) {
            return;
        }
        if (phase == Phase.HELD) {
            held.fileDeath(EventAttribution.stamp(moved, acting, null), Refs.of(event.card()));
        } else {
            file(moved, null);
        }
    }

    private void file(EffectEvent event, String namedCause) {
        open.file(EventAttribution.stamp(event, acting, namedCause));
    }
}
