package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * The events one record is gathering, with repeated state transitions filed
 * once.
 *
 * <p>A resolution bracket, a combat bracket and a held effect half each gather
 * their events this way, so a repeat is recognised the same way in all three.
 */
final class EventBuffer {

    /**
     * Outcomes a second identical report of cannot mean a second occurrence.
     *
     * <p>These describe a transition into a state a thing is either in or not:
     * a card is on the battlefield, a permanent is tapped, an aura is attached.
     * Told twice in one record, the second telling is the engine publishing the
     * same change again — nothing in the record can distinguish it from the
     * first, and nothing downstream can use it.
     *
     * <p>Everything else is deliberately absent, because for a quantitative
     * outcome the repeat <b>is</b> the information: two creatures each dealing
     * one damage to the same blocker render two identical {@code damage_dealt}
     * events, and collapsing them would turn two damage into one.
     *
     * <p><b>{@code REGENERATED} does not actually fit the "in-or-out state"
     * reasoning above, unlike its neighbours here.</b> {@code destroyed} and
     * {@code sacrificed} are one-shot by construction — the object changes
     * zones and stops being the same object — and {@code tapped} has a real
     * re-fire guard ({@code Card.tap} returns false when already tapped).
     * Regeneration has neither: {@code RegenerationEffect.resolve()} is a
     * shield being <i>consumed</i>, not the ability granting one, and shields
     * are granted separately, one independent command-zone
     * {@code ReplacementEffect} per activation ({@code
     * RegenerateEffect.createRegenerationEffect}) — two activations on one
     * creature make two shields, each able to fire its own event. The engine
     * itself treats repeated regeneration as a real, distinguishable fact:
     * {@code Card.regeneratedThisTurn} is an incrementing counter, exposed to
     * card scripts as {@code RegeneratedThisTurn} X-math. So this dedup
     * <i>can</i>, in principle, drop a genuine second occurrence rather than
     * only a duplicate publication of the first.
     *
     * <p>It stays anyway, for reasons that hold regardless of that fact
     * rather than because a repeat is impossible: a combat bracket is one per
     * damage step — first strike and the regular step are different brackets,
     * flushed at the phase boundary ({@code BusBracketCollector.onPhase}) — so the textbook
     * "regenerate through first strike, then again in the regular step" lands
     * in two brackets and is never compared. And a second destroy check
     * driven by a separately-resolving ability changes {@code attributed_to}
     * (stamped in {@code BusBracketCollector.fileOutcome}), which breaks the
     * rendered-line equality {@link #file} dedups on. Nobody has
     * constructed, or ruled out, a same-bracket same-attribution double from
     * reasoning about the code alone — that is an open question static
     * analysis cannot settle, and one a corpus run can: look for brackets that
     * contain a {@code regenerated} event and check whether any single card
     * carries two.
     */
    static final Set<String> IDEMPOTENT_EVENTS = Set.of(
            EffectEvent.ZONE_CHANGE, EffectEvent.TAPPED, EffectEvent.UNTAPPED,
            EffectEvent.ATTACHED, EffectEvent.UNATTACHED, EffectEvent.PHASED,
            EffectEvent.FACE_CHANGE, EffectEvent.DESTROYED,
            EffectEvent.SACRIFICED, EffectEvent.REGENERATED);

    private final List<EffectEvent> events = new ArrayList<>();
    /** The rendered lines of the state transitions filed so far. */
    private final Set<String> seen = new LinkedHashSet<>();

    /**
     * A buffer holding these events as they are, repeats included, so that a
     * later repeat of any of their state transitions is refused.
     */
    static EventBuffer copyOf(List<EffectEvent> events) {
        EventBuffer copy = new EventBuffer();
        for (EffectEvent event : events) {
            copy.events.add(event);
            if (IDEMPOTENT_EVENTS.contains(event.type())) {
                copy.seen.add(event.toJson());
            }
        }
        return copy;
    }

    /**
     * Add an event unless it repeats a state transition already here.
     *
     * @return whether it was added
     */
    boolean file(EffectEvent event) {
        // Compared as the rendered line, so two events are the same when what
        // the record says about them is the same -- subjects, params, duration
        // and the clause they were attributed to, all of it.
        if (IDEMPOTENT_EVENTS.contains(event.type()) && !seen.add(event.toJson())) {
            return false;
        }
        events.add(event);
        return true;
    }

    /**
     * Add an event without checking it for a repeat, and without it making a
     * later event a repeat: for an event whose repeat is judged in another
     * buffer, as a forced resolution's state-based death is judged in the
     * half it joins ({@code ForkCollector.splitByMode}).
     */
    void fileUnchecked(EffectEvent event) {
        events.add(event);
    }

    /** The events filed so far, in order. */
    List<EffectEvent> events() {
        return Collections.unmodifiableList(events);
    }

    boolean isEmpty() {
        return events.isEmpty();
    }

    void clear() {
        events.clear();
        seen.clear();
    }

    /** Every event filed so far, leaving the buffer empty. */
    List<EffectEvent> drain() {
        List<EffectEvent> taken = List.copyOf(events);
        clear();
        return taken;
    }
}
