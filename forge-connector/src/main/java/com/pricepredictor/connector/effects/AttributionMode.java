package com.pricepredictor.connector.effects;

import java.lang.reflect.Method;

/**
 * Whether this JVM's Forge carries the attribution patch.
 *
 * <p>Detected at worker startup by probing for the patched hooks, not compiled
 * in. The patch lives in a sibling checkout that is rebuilt independently and
 * re-applied by hand after every Forge upgrade, so a build-time switch would
 * silently mislabel a corpus collected after the patch lapsed — and a corpus
 * whose attribution is wrong in a way nothing records is worse than one that
 * says it is degraded.
 *
 * <p>Every record carries the detected mode. Readers treat it as collection
 * metadata: it says how we came to observe the record, not anything about the
 * game, so it never reaches the model.
 */
public enum AttributionMode {

    /** The three attribution hooks are present; events name their cause. */
    PATCHED("patched"),

    /**
     * Stock Forge. Events are attributed to whatever was resolving when they
     * arrived — correct for the common case, approximate where two things
     * resolve in one bracket.
     */
    DEGRADED("degraded");

    private final String wireValue;

    AttributionMode(String wireValue) {
        this.wireValue = wireValue;
    }

    public String wireValue() {
        return wireValue;
    }

    /**
     * Hooks the patch adds. Probed by name so detection needs no compile-time
     * dependency on a patched checkout — the connector must build against stock
     * Forge.
     */
    private static final String TRIGGER_HANDLER = "forge.game.trigger.TriggerHandler";
    private static final String CAUSE_HOOK = "setEffectRecordCause";

    private static AttributionMode detected;

    /** Probe once per JVM and remember the answer. */
    public static synchronized AttributionMode detect() {
        if (detected == null) {
            detected = probe();
        }
        return detected;
    }

    private static AttributionMode probe() {
        try {
            Class<?> handler = Class.forName(TRIGGER_HANDLER);
            for (Method method : handler.getMethods()) {
                if (CAUSE_HOOK.equals(method.getName())) {
                    return PATCHED;
                }
            }
        } catch (ClassNotFoundException | LinkageError ignored) {
            // Forge without the trigger handler at all is not a patched Forge.
        }
        return DEGRADED;
    }

    /** For tests: forget the probe so the next call re-runs it. */
    static synchronized void reset() {
        detected = null;
    }
}
