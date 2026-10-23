package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

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
 * collector uses, and filed into an {@link EventBuffer}, the same buffer its
 * brackets use, so an interventional or probe record and an observed one
 * describe the same outcome the same way: the same subscriptions, the same
 * shaping, the same cause the bus names for a damage, a poison or a
 * radiation counter, and a state transition repeated within one record —
 * the whole fork, or one chosen mode's half — filed once.
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
     */
    private final SpellAbility acting;
    /**
     * Whether a card damaged in this fork is damaged in combat: true for a
     * damage-step probe, which deals one combat damage step, and false for an
     * intervention, whose forced resolution is not one. The flag is what a
     * reader uses to tell the two apart, and a probe's damage is compared
     * against the observed combat record's.
     */
    private final boolean combatDamage;
    /** The events of the modes already closed, in order. */
    private final List<EffectEvent> closed = new ArrayList<>();
    /**
     * The events of the open span: the whole fork, or the chosen mode now
     * resolving, which is the unit a repeat is recognised within, as it is
     * a mode's half on the live side.
     */
    private final EventBuffer open = new EventBuffer();
    private boolean modeOpen;

    /**
     * Where the forced ability stands, which decides what an intervention
     * files: what the live bracket would put in the forced ability's record,
     * and nothing else the drained stack does.
     *
     * <ul>
     *   <li>{@link #BEFORE}: between the cast and the resolution, what is
     *   published while the forced ability is the top of the fork's stack is
     *   filed, as live files the state-based check before priority into the
     *   spell's bracket. While anything sits above it — a cast trigger:
     *   prowess, a "whenever you cast" or becomes-target trigger, storm,
     *   cascade — nothing is: that is the item's own record live. Nor is
     *   anything in that item's aftermath ({@link #aboveHeld}), though the
     *   forced ability is the top again: live keeps the deaths for the
     *   item's held half and sends the rest to no record, since none of it
     *   belongs to an ability that has not begun to resolve. What was filed
     *   before an item was pushed above is kept.</li>
     *   <li>{@link #RESOLVING}: from the first event published while the
     *   forced ability is the one resolving, until its {@code
     *   GameEventSpellResolved}: everything is filed, as live files the open
     *   bracket's events. A permanent leaving the battlefield while the
     *   tracker is frozen mid-resolution — an end-the-turn effect's own
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
    private enum Phase { BEFORE, RESOLVING, HELD, CLOSED }

    private Phase phase = Phase.BEFORE;
    /**
     * Whether an item that resolved above the forced ability is in its
     * aftermath: from its resolved event until the next ability reaches the
     * stack or resolves. Nothing published then is the forced ability's.
     */
    private boolean aboveHeld;
    /**
     * Positions in {@link #events()} of the deaths filed in {@link
     * Phase#HELD}, or of any state-based leave a probe or an ungated sink
     * filed: the moves a modal fork files by {@code halfForDeath}.
     */
    private final Set<Integer> stateBasedLeaves = new LinkedHashSet<>();

    private ForkEventSink(Game fork, SpellAbility acting, boolean combatDamage) {
        super(fork);
        this.acting = acting;
        this.combatDamage = combatDamage;
        primeBattlefield();
    }

    /** A sink for a forced resolution of {@code acting}. */
    static ForkEventSink forIntervention(Game fork, SpellAbility acting) {
        return new ForkEventSink(fork, acting, false);
    }

    /** A sink for a damage step dealt in a fork, with no ability resolving. */
    static ForkEventSink forProbe(Game fork) {
        return new ForkEventSink(fork, null, true);
    }

    List<EffectEvent> events() {
        List<EffectEvent> all = new ArrayList<>(closed);
        all.addAll(open.events());
        return List.copyOf(all);
    }

    /**
     * A chosen mode's first clause is resolving: close the previous mode's
     * span, so a repeat is recognised within one mode, as the live bracket
     * recognises it within one mode's half.
     *
     * <p>The first mode closes nothing: what was filed before its clause
     * shares its span, as it shares the first half live.
     *
     * @return the position in {@link #events()} of the mode's first event
     */
    int beginMode() {
        if (modeOpen) {
            closed.addAll(open.drain());
        }
        modeOpen = true;
        return size();
    }

    /**
     * Where in {@link #events()} the deaths a modal fork files by {@code
     * halfForDeath} sit: an intervention's in {@link Phase#HELD}, a probe's or
     * an ungated sink's in any state-based check, each read as the bracket
     * collector reads one ({@link #isStateBasedLeave}).
     */
    Set<Integer> stateBasedLeaves() {
        return Set.copyOf(stateBasedLeaves);
    }

    private int size() {
        return closed.size() + open.events().size();
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
        aboveHeld = false;
        if (forced && (phase == Phase.BEFORE || phase == Phase.RESOLVING)) {
            phase = Phase.HELD;
        } else if (phase == Phase.HELD) {
            phase = Phase.CLOSED;
        } else if (phase == Phase.BEFORE) {
            aboveHeld = true;
        }
    }

    /** The next ability has reached the stack, which closes a held window. */
    @Subscribe
    public void onCast(GameEventSpellAbilityCast event) {
        if (!gated()) {
            return;
        }
        aboveHeld = false;
        if (phase == Phase.HELD) {
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
        if (phase == Phase.BEFORE && forcedOnTop()) {
            if (game.getStack().isResolving()) {
                phase = Phase.RESOLVING;
            } else {
                return !aboveHeld;
            }
        }
        return switch (phase) {
            case RESOLVING -> true;
            case HELD -> move != null && isStateBasedLeave(move);
            case BEFORE, CLOSED -> false;
        };
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
     * A death the forced resolution left behind is marked where it was filed,
     * so the fork can file it by the live rule.
     *
     * <p>Filed unchecked, its repeat judged later by the half it joins
     * ({@code ForkCollector.splitByMode}): the live collector checks a death
     * against the half {@code halfForDeath} assigns it to, and filed here it
     * would be checked against the open span, the last mode's, whichever
     * half it joins. A probe's death is checked here: a damage step is one
     * record, as the observed combat bracket is.
     *
     * <p>A probe's leave is marked only when it was filed: a repeat the
     * buffer refused has no position, and the next free one belongs to
     * whatever is filed next.
     *
     * <p>Only an intervention's death in {@link Phase#HELD} is marked. One in
     * the check before the forced ability resolves, or one while the tracker
     * is frozen mid-resolution, is filed as an ordinary event, checked for a
     * repeat in its span: live files either into the open bracket like any
     * other event, so on a modal resolution it rides with the mode in whose
     * span it fell rather than joining a half by {@code halfForDeath}, which
     * is a resolved half's rule.
     */
    @Override
    void fileMove(GameEventCardChangeZone event, EffectEvent moved) {
        if (!admits(event)) {
            return;
        }
        int at = size();
        boolean probe = combatDamage;
        if (!isStateBasedLeave(event) || gated() && phase != Phase.HELD) {
            file(moved, null);
        } else if (probe) {
            if (file(moved, null)) {
                stateBasedLeaves.add(at);
            }
        } else {
            open.fileUnchecked(EventAttribution.stamp(moved, acting, null));
            stateBasedLeaves.add(at);
        }
    }

    private boolean file(EffectEvent event, String namedCause) {
        return open.file(EventAttribution.stamp(event, acting, namedCause));
    }
}
