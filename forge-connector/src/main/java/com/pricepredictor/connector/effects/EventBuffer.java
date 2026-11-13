package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The events one record is gathering, with a state transition told twice in
 * a row filed once.
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
     * Told twice in a row for the same subjects, with no other transition of
     * theirs between, the second telling is the engine publishing the same
     * change again — nothing in the record can distinguish it from the first,
     * and nothing downstream can use it. The same transition after another of
     * the same subjects is a second occurrence and is kept: a creature
     * flickered twice by one resolution leaves and returns twice, a permanent
     * tapped, untapped and tapped again was tapped twice.
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
     * rendered-line equality {@link #file} dedups on; and two regenerations
     * with any other transition of the creature between them are both kept. Nobody has
     * constructed, or ruled out, a same-bracket same-attribution double from
     * reasoning about the code alone — that is an open question static
     * analysis cannot settle, and one a corpus run can: look for brackets that
     * contain a {@code regenerated} event and check whether any single card
     * carries two.
     */
    static final Set<EventType> IDEMPOTENT_EVENTS = EnumSet.of(
            EventType.ZONE_CHANGE, EventType.TAPPED, EventType.UNTAPPED,
            EventType.ATTACHED, EventType.UNATTACHED, EventType.PHASED,
            EventType.FACE_CHANGE, EventType.DESTROYED,
            EventType.SACRIFICED, EventType.REGENERATED);

    private final List<EffectEvent> events = new ArrayList<>();
    /** The rendered line of the last state transition filed, by its subjects. */
    private final Map<List<String>, String> lastTransition = new HashMap<>();

    /**
     * A buffer holding these events as they are, repeats included, so that a
     * later repeat of any of their state transitions is refused.
     */
    static EventBuffer copyOf(List<EffectEvent> events) {
        EventBuffer copy = new EventBuffer();
        for (EffectEvent event : events) {
            copy.events.add(event);
            if (IDEMPOTENT_EVENTS.contains(event.type())) {
                copy.lastTransition.put(List.copyOf(event.subjects()), event.toJson());
            }
        }
        return copy;
    }

    /**
     * Add an event unless it repeats the last state transition of its
     * subjects.
     *
     * @return whether it was added
     */
    boolean file(EffectEvent event) {
        // Compared as the rendered line, so two events are the same when what
        // the record says about them is the same -- subjects, params, duration
        // and the clause they were attributed to, all of it.
        if (IDEMPOTENT_EVENTS.contains(event.type())) {
            String line = event.toJson();
            if (line.equals(lastTransition.put(List.copyOf(event.subjects()), line))) {
                return false;
            }
        }
        if (event.type() == EventType.LIBRARY_SHUFFLED) {
            forgetLibraryPositions(event.subjects());
        }
        events.add(event);
        return true;
    }

    /**
     * A shuffle of these players' libraries takes the position off every
     * earlier move into them: a card shuffled in sits nowhere in particular,
     * and a reader that finds a move into a library with no position and a
     * shuffle of it reads the card as shuffled in.
     */
    private void forgetLibraryPositions(List<String> shuffled) {
        for (EffectEvent earlier : events) {
            if (earlier.type() == EventType.ZONE_CHANGE
                    && shuffled.contains(earlier.intoLibraryOf())) {
                earlier.params().remove("library_position");
            }
        }
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
        lastTransition.clear();
    }

    /** Every event filed so far, leaving the buffer empty. */
    List<EffectEvent> drain() {
        List<EffectEvent> taken = List.copyOf(events);
        clear();
        return taken;
    }
}
