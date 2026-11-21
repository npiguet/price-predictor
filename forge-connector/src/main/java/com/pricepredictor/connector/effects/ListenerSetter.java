package com.pricepredictor.connector.effects;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;

/**
 * A hook that takes a listener: a static {@code set*Listener} method the patch
 * adds, and the callbacks of the listener it takes.
 *
 * <p>Presence is not the whole contract. A setter keeps its name when its
 * listener loses a callback or gains arguments, so {@link #gap()} also checks
 * the listener interface against {@link #callbacks}, by name and by the
 * arguments the handlers read.
 *
 * @param site      the class it is declared on and its name
 * @param unlocks   what stops working without it
 * @param callbacks the methods of the listener it takes, which are as much the
 *                  contract as the setter's own name
 */
record ListenerSetter(HookSite site, String unlocks, List<Callback> callbacks) implements Hook {

    @Override
    public Optional<Method> lookUp() {
        return HookReflection.find(site, Reach.PUBLIC);
    }

    /** The interface this setter's listener is declared as, or empty when it takes none. */
    Optional<Class<?>> listenerType() {
        return lookUp()
                .filter(setter -> setter.getParameterCount() == 1)
                .map(setter -> setter.getParameterTypes()[0]);
    }

    /** Absent, or its listener lacking a callback or an argument the collectors read. */
    @Override
    public Optional<String> gap() {
        List<String> lacking = listenerType().map(this::lacking).orElse(List.of());
        return lacking.isEmpty() ? Hook.super.gap()
                : Optional.of(this + " — " + unlocks + "; its listener: " + String.join(", ", lacking));
    }

    /** What {@code listener} lacks of {@link #callbacks}, one phrase each, in their order. */
    List<String> lacking(Class<?> listener) {
        return callbacks.stream().map(callback -> callback.lackIn(listener))
                .flatMap(Optional::stream).toList();
    }

    /** The callback the engine calls by {@code invoked}'s name, or empty for a method no collector reads. */
    Optional<Callback> callbackFor(Method invoked) {
        return callbacks.stream()
                .filter(callback -> callback.methodName().equals(invoked.getName())).findFirst();
    }

    /**
     * Whether this setter's listener interface declares {@code callback}: one
     * an older jar never makes, which a channel relying on it must do without.
     */
    boolean listenerDeclares(Callback callback) {
        return listenerType().map(callback::declaredBy).orElse(false);
    }

    /** Install a listener on this setter, or do nothing; see {@link HookReflection#install}. */
    boolean install(InvocationHandler handler) {
        return HookReflection.install(lookUp(), handler);
    }

    /** Remove the listener this setter holds, or do nothing. */
    void uninstall() {
        HookReflection.uninstall(lookUp());
    }

    /**
     * A handler for this setter's listener, which hands {@code body} each
     * of the {@link #callbacks} the engine makes, and which installs on
     * this setter and no other.
     */
    Listener listener(ListenerBody body) {
        return new Listener(this, body);
    }

    @Override
    public String toString() {
        return site.toString();
    }
}
