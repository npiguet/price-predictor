package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;

/**
 * A modal resolution's chosen modes, split at each mode's first clause
 * (FR-029a).
 *
 * <p>The events a mode produced are its own and nothing else's: everything
 * filed between its first clause and the next mode's, or the end of the
 * resolution, is its half's. Events filed before the first mode's clause fall
 * into the first mode's half rather than vanishing, since the buffer they
 * wait in is drained only when a mode is complete.
 *
 * <p>One split for the live bracket and for a forced resolution, so an
 * observed and a forced resolution of the same charm divide their events the
 * same way (FR-029f).
 *
 * <p>Owned by the one resolution it splits, so no other resolution ever sees
 * it. That matters for a mode that casts a spell (Tetsuo, Imperial Champion's
 * second mode): Forge resolves nothing inside a resolution, so the cast
 * spell's {@code GameEventSpellAbilityCast} arrives in the middle of the mode,
 * the charm then finishes and publishes its {@code GameEventSpellResolved},
 * and the spell resolves afterwards, on its own. The charm keeps its split
 * until its own resolution closes it; the spell's opens later, empty.
 */
final class ModeSplit {

    /**
     * One chosen mode, as its half's record names it.
     *
     * @param option the mode's {@code Choices$} position, or null for a mode
     *               nothing could place, whose half keeps the root key
     * @param state  the board at the mode's first clause, so a later mode's
     *               board shows the earlier modes' effects (FR-029b)
     */
    record Mode(Integer option, Snapshot state) {
    }

    /**
     * One complete half: its mode, the events filed while it resolved, and
     * whether each conditional clause that resolved in it met its condition
     * (FR-034), in the order the clauses resolved.
     */
    record ModeHalf(Mode mode, List<EffectEvent> events, List<ConditionsMet> conditions) {
    }

    /**
     * The events filed since the last half completed: the open span, which
     * is the unit a repeat is recognised within — the chosen mode resolving
     * now, or the whole resolution where no mode has begun.
     */
    private final EventBuffer open = new EventBuffer();
    /** The open span's conditional clauses, drained with its events. */
    private final List<ConditionsMet> openConditions = new ArrayList<>();
    /** The modes whose halves are complete, in order. */
    private final List<ModeHalf> complete = new ArrayList<>();
    /** The mode whose clauses are resolving now, or null outside one. */
    private Mode current;

    /** Where the resolution's events are filed as they arrive. */
    EventBuffer buffer() {
        return open;
    }

    /** A conditional clause resolved in the open span. */
    void noteCondition(ConditionsMet condition) {
        if (condition != null) {
            openConditions.add(condition);
        }
    }

    /** Complete the current mode's half and open one for the next mode. */
    void begin(Integer option, Snapshot state) {
        close();
        current = new Mode(option, state);
    }

    /** Move the open span's events into the current mode's half and complete it. */
    void close() {
        if (current == null) {
            return;
        }
        complete.add(new ModeHalf(current, open.drain(), List.copyOf(openConditions)));
        openConditions.clear();
        current = null;
    }

    /** Whether nothing was filed: none waits in the open span, and no complete half holds one. */
    boolean isEmpty() {
        return open.isEmpty() && complete.stream().allMatch(half -> half.events().isEmpty());
    }

    /** The complete halves, in order. */
    List<ModeHalf> complete() {
        return List.copyOf(complete);
    }

    /**
     * The halves the resolution writes once it is over: one per chosen mode,
     * or, when no mode's half is complete, one root half.
     *
     * <p>A root half holds what the open span filed: with no mode begun,
     * that is everything the resolution filed.
     *
     * @param rootState the board a root half is read against
     */
    List<ModeHalf> halves(Snapshot rootState) {
        return complete.isEmpty()
                ? List.of(new ModeHalf(
                        new Mode(null, rootState), List.copyOf(open.events()),
                        List.copyOf(openConditions)))
                : List.copyOf(complete);
    }
}
