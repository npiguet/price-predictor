package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A resolution's effect halves from the moment it resolves until its
 * state-based consequences are over, while the deaths it caused can still
 * join them.
 *
 * <p>Forge does not check state-based actions when an ability finishes: the
 * creature a Lightning Bolt left with lethal damage dies at the next check,
 * after {@code GameEventSpellResolved}. A half written at the resolved event
 * would hold the damage and never the death, so the halves are held, and each
 * death is filed into the half it belongs to as it arrives.
 *
 * <p>One rule for the live collector and for a forced resolution, so an
 * observed and a forced resolution of the same charm file the same death the
 * same way.
 *
 * @param <H> what the side holding the halves keeps about each besides its
 *            events
 */
final class HeldHalves<H> {

    /**
     * One held half: what its record says besides its events, the events, and
     * the entity refs those events named when the resolution ended.
     *
     * <p>What it touched is read once, before any death joins it, so a death
     * is placed by what the resolution's own events did and never by another
     * death.
     */
    static final class Half<H> {
        private final H about;
        private final EventBuffer events;
        private final Set<String> touched;

        private Half(H about, List<EffectEvent> events) {
            this.about = about;
            this.events = EventBuffer.copyOf(events);
            this.touched = touchedBy(events);
        }

        H about() {
            return about;
        }

        List<EffectEvent> events() {
            return events.events();
        }
    }

    private final List<Half<H>> halves = new ArrayList<>();

    /** Hold one more half, in order. */
    void add(H about, List<EffectEvent> events) {
        halves.add(new Half<>(about, events));
    }

    /** The halves, in order. */
    List<Half<H>> halves() {
        return Collections.unmodifiableList(halves);
    }

    /**
     * File a state-based death into the half it belongs to
     * ({@link #halfForDeath}), checked there for a repeat as any event of
     * that half is. The death is filed as given: each side stamps it its own
     * way first.
     *
     * @param subject the entity ref of the permanent that left
     */
    void fileDeath(EffectEvent death, String subject) {
        List<Set<String>> touched = new ArrayList<>();
        for (Half<H> half : halves) {
            touched.add(half.touched);
        }
        halves.get(halfForDeath(touched, subject)).events.file(death);
    }

    /**
     * The entity refs a half's own events name as subjects: what it damaged,
     * pumped, shrank, countered, animated or moved.
     */
    private static Set<String> touchedBy(List<EffectEvent> events) {
        Set<String> touched = new LinkedHashSet<>();
        for (EffectEvent event : events) {
            for (String subject : event.subjects()) {
                if (Refs.isEntity(subject)) {
                    touched.add(subject);
                }
            }
        }
        return touched;
    }

    /**
     * Which of a resolution's halves a state-based death belongs to: the last
     * whose events named the permanent — the mode that was the last to change
     * it — or the last half when none did.
     *
     * @param touchedPerHalf each half's {@link #touchedBy} set, in order;
     *                       never empty
     */
    static int halfForDeath(List<Set<String>> touchedPerHalf, String subject) {
        for (int i = touchedPerHalf.size() - 1; i >= 0; i--) {
            if (touchedPerHalf.get(i).contains(subject)) {
                return i;
            }
        }
        return touchedPerHalf.size() - 1;
    }
}
