package com.pricepredictor.connector.effects;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;

/**
 * A hook called on an engine object: a public instance method the patch adds,
 * or a component of a record the patch makes public.
 *
 * <p>Looked up on its owner for {@link #present()} and on the receiver's own
 * class for a {@link #read}, which answers null rather than throwing on a
 * checkout without it.
 *
 * @param site    the class it is declared on and its name
 * @param unlocks what stops working without it
 */
record InstanceAccessor(HookSite site, String unlocks) implements Hook {

    @Override
    public Optional<Method> lookUp() {
        return HookReflection.find(site, Reach.PUBLIC);
    }

    /** Read this no-argument accessor off {@code target}, or answer null. */
    Object read(Object target) {
        return HookReflection.read(
                target, new HookReflection.Shape(site.method(), HookReflection.Shape.NO_ARGUMENT));
    }

    /** Call this one-argument accessor on {@code target} with {@code argument}, or answer null. */
    Object read(Object target, Argument argument) {
        return HookReflection.read(target, argument.shapeOf(site), argument.value());
    }

    /**
     * Call this one-argument lend on {@code target}: what gives back what
     * it lent, or null when it lent nothing or the hook is absent.
     */
    Runnable lend(Object target, Argument argument) {
        return read(target, argument) instanceof Runnable giveBack ? giveBack : null;
    }

    @Override
    public String toString() {
        return site.toString();
    }

    /**
     * The one argument of a call: its value and the parameter type the
     * method declares it as, which the lookup is by.
     */
    record Argument(Class<?> type, Object value) {

        /** The shape of {@code site}'s method taking this argument. */
        HookReflection.Shape shapeOf(HookSite site) {
            return new HookReflection.Shape(site.method(), List.of(type));
        }
    }
}
