package com.pricepredictor.connector.effects;

import java.lang.reflect.Method;
import java.util.Optional;

/**
 * One method the engine patch adds, and what having it buys.
 *
 * <p>Three kinds, each offering only what can be done with it: a
 * {@link ListenerSetter} takes a listener ({@code install}), a
 * {@link StaticAccessor} is called on its class and an
 * {@link InstanceAccessor} on an engine object ({@code read}). The inventory
 * of them is {@link PatchHooks#REQUIRED}.
 */
public sealed interface Hook permits ListenerSetter, StaticAccessor, InstanceAccessor {

    /** The class it is declared on and its name. */
    HookSite site();

    /** What stops working without it. */
    String unlocks();

    /** The method itself, or empty on a checkout without it; looked up once per JVM. */
    Optional<Method> lookUp();

    /** Whether this checkout has it. */
    default boolean present() {
        return lookUp().isPresent();
    }

    /**
     * What is wrong with this hook on this checkout, or empty: absent, or,
     * for a setter, its listener lacking a callback or an argument the
     * collectors read.
     */
    default Optional<String> gap() {
        return present() ? Optional.empty() : Optional.of(this + " — " + unlocks());
    }
}
