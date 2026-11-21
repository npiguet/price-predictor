package com.pricepredictor.connector.effects;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;

/**
 * A hook called on its class: a static method the patch adds.
 *
 * @param site    the class it is declared on and its name
 * @param unlocks what stops working without it
 * @param reach   where it is looked up: public, or the one private method the
 *                patch adds ({@link PatchHooks#NOTE_CONFIRM})
 */
record StaticAccessor(HookSite site, String unlocks, Reach reach) implements Hook {

    /** A public static hook. */
    StaticAccessor(HookSite site, String unlocks) {
        this(site, unlocks, Reach.PUBLIC);
    }

    @Override
    public Optional<Method> lookUp() {
        return HookReflection.find(site, reach);
    }

    /** Call this no-argument method and answer what it returns, or null when it is absent or fails. */
    Object read() {
        return HookReflection.callStatic(lookUp());
    }

    /**
     * Call this method and answer what it returns, for a hook a caller cannot
     * do without: absent, it throws naming itself, and what the method throws
     * is rethrown rather than read as absence.
     */
    Object call(Object... args) {
        Method found = lookUp().orElseThrow(() -> new IllegalStateException(
                "this Forge lacks " + this + ": " + unlocks));
        return HookReflection.callThrough(found, args);
    }

    /**
     * Call the overload of this method taking exactly {@code parameterTypes},
     * or answer null when it is absent or fails; see
     * {@link HookReflection#invokeStatic}.
     */
    Object invoke(Class<?>[] parameterTypes, Object... args) {
        return HookReflection.invokeStatic(new HookReflection.Overload(site, List.of(parameterTypes)), args);
    }

    @Override
    public String toString() {
        return site.toString();
    }
}
