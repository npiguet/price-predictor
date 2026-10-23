package com.pricepredictor.connector.effects;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

/**
 * A stderr line printed the first time it is raised and never again.
 *
 * <p>A broken rule or a renamed outcome fails on every clause after its first,
 * so silence would hide the failure and a line per occurrence would flood a
 * playthrough's log. "Once" means once per JVM, and per instance: each
 * failure gets its own, so one going off cannot silence an unrelated other.
 * The engine's side of the same hooks gives its listener failures the same
 * treatment, through Forge's {@code OnceOnlyWarning}.
 */
final class OnceWarning {

    private final AtomicBoolean raised = new AtomicBoolean();

    /** Print the message unless this warning has already been raised; a silenced call builds no message. */
    void raise(Supplier<String> message) {
        if (raised.compareAndSet(false, true)) {
            System.err.println(message.get());
        }
    }
}
