package com.pricepredictor.connector.effects;

/**
 * One callback's arguments, read by position.
 *
 * <p>By position because nothing here compiles against the listener
 * interface. A typed read answers null for a value of another type, the
 * way an {@code instanceof} test does, so a drifted signature costs the
 * argument rather than throwing out of the engine's call.
 */
final class Args {
    private final Object[] values;

    /** The arguments the engine's call brought, which only {@link Listener} wraps. */
    Args(Object[] values) {
        this.values = values;
    }

    /** The argument at {@code index}, which the callback's arity guarantees. */
    Object get(int index) {
        return values[index];
    }

    /** The argument at {@code index} if it is a {@code type}, or null. */
    <T> T get(int index, Class<T> type) {
        return type.isInstance(values[index]) ? type.cast(values[index]) : null;
    }

    /** An argument past the arity, or null where an older jar did not pass it. */
    Object optional(int index) {
        return index < values.length ? values[index] : null;
    }
}
