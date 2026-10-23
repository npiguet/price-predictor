package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One attributed outcome, in the schema's event shape.
 *
 * <p>{@code {type, subjects, params, duration, attributed_to}}, the type one
 * of {@link EventType}'s members.
 */
public final class EffectEvent {

    private final EventType type;
    private final List<String> subjects = new ArrayList<>();
    private final Map<String, Object> params = new LinkedHashMap<>();
    private EventDuration duration;
    private String attributedTo;

    public EffectEvent(EventType type) {
        this.type = type;
    }

    public EffectEvent subject(String id) {
        if (id != null) {
            subjects.add(id);
        }
        return this;
    }

    public EffectEvent param(String key, Object value) {
        if (value != null) {
            params.put(key, value);
        }
        return this;
    }

    /**
     * The object the engine names as having caused this outcome, as a ref.
     *
     * <p><b>One channel, one spelling.</b> Every collector writes the cause
     * through this method and nowhere else, so the trait-derived side (a
     * replacement's or a trigger's run-parameter map) and the bus-derived side
     * (the event the bus delivered) put the same kind of value under the same
     * key. The Python schema keeps {@code cause} in {@code PROVENANCE_PARAMS}
     * rather than in one type's row, so every event type accepts it and no
     * per-type table has to be widened for a new one.
     *
     * <p>The value is an entity or player ref — {@code "E24"}, {@code "P1"} —
     * the same identity {@code state.entities} carries, never a description and
     * never a list of the run-parameter keys the map happened to hold.
     *
     * <p>Why it is not decoration: {@code subjects} says who an outcome
     * happened to and the normalized params say how much, which leaves two
     * attackers each dealing 1 damage to the same player serializing
     * byte-identically. Those are distinct events that must not be deduplicated,
     * and the cause is what tells them apart.
     *
     * <p>A null ref writes nothing rather than a placeholder. Forge genuinely
     * names no cause for a phase change, an untap, a block or a declared
     * attack, and an honest absence is what a reader can act on.
     */
    public EffectEvent cause(String ref) {
        return param("cause", ref);
    }

    /** How long the outcome lasts; null where nothing stamped it. */
    public EffectEvent duration(EventDuration value) {
        this.duration = value;
        return this;
    }

    /**
     * The sub-ability link this outcome came from, or null for the root line.
     *
     * <p>An event attributed to a link the sidecar does not map falls back to
     * the root line rather than being dropped, so attribution is never lossy.
     */
    public EffectEvent attributedTo(String link) {
        this.attributedTo = link;
        return this;
    }

    public EventType type() {
        return type;
    }

    public List<String> subjects() {
        return subjects;
    }

    /**
     * The normalized params, as a read accessor for tests and other readers
     * that check one key rather than parse {@link #toJson()}. The collectors
     * never read an event's params back; they only write them.
     */
    public Map<String, Object> params() {
        return params;
    }

    /** Every field on every event, a null one as JSON {@code null}. */
    public String toJson() {
        return Json.object()
                .value("type", type)
                .raw("subjects", Json.stringArray(subjects))
                .value("params", params)
                .value("duration", duration)
                .value("attributed_to", attributedTo)
                .render();
    }
}
