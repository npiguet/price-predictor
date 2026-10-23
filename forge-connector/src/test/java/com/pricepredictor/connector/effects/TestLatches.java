package com.pricepredictor.connector.effects;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Re-arms a report-once latch, so each test sees its own "did this print".
 *
 * <p>The latches are private static {@link OnceWarning}s that production
 * raises once per JVM and never clears, so a test reaches them reflectively
 * rather than production carrying a reset only tests would call.
 */
final class TestLatches {

    private TestLatches() {
    }

    static void reset(Class<?> owner, String latch) {
        try {
            var field = owner.getDeclaredField(latch);
            field.setAccessible(true);
            var raised = OnceWarning.class.getDeclaredField("raised");
            raised.setAccessible(true);
            ((AtomicBoolean) raised.get(field.get(null))).set(false);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(owner.getSimpleName() + " has no latch " + latch, e);
        }
    }
}
