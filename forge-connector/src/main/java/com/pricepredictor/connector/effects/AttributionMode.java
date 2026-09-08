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

    private static AttributionMode detected;

    /** Probe once per JVM and remember the answer. */
    public static synchronized AttributionMode detect() {
        if (detected == null) {
            detected = probe();
        }
        return detected;
    }

    /**
     * Whether the cause channel is there.
     *
     * <p>Read from {@link PatchHooks#REQUIRED} rather than naming the method
     * again here, so the mode and the hook inventory cannot drift apart. It is
     * deliberately one hook and not all of them: the mode says how an event was
     * attributed, and a checkout missing some later hook still attributes
     * exactly. {@link PatchHooks#report} is what names a partly-applied patch.
     */
    private static AttributionMode probe() {
        return PatchHooks.REQUIRED.get(0).present() ? PATCHED : DEGRADED;
    }

    /** For tests: forget the probe so the next call re-runs it. */
    static synchronized void reset() {
        detected = null;
    }
}
