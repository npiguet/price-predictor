package com.pricepredictor.connector.effects;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

/**
 * A handler built for one setter's listener, carrying that setter.
 *
 * <p>Carrying it is what lets a collector install the handler without
 * naming the hook a second time. A handler built from one hook and
 * installed on another hears none of the callbacks it dispatches on, and
 * the channel behind it goes empty without a word; built and installed
 * through the same {@link ListenerSetter}, the two cannot disagree.
 *
 * <p>Any other method the proxy is asked for, and a callback called with
 * fewer arguments than its {@link Callback#arity}, answers null and does
 * nothing, the way an absent hook would: a handler that threw instead
 * would throw out of the engine's own call. A short call is counted
 * ({@link HookFailures#countShortCall}): it loses what it would have
 * recorded.
 *
 * <p>What the body throws is counted against the hook
 * ({@link HookFailures}) and goes on to Forge, which contains it.
 */
record Listener(ListenerSetter hook, ListenerBody body) implements InvocationHandler {

    @Override
    public Object invoke(Object proxy, Method invoked, Object[] args) {
        if (invoked.getDeclaringClass() == Object.class) {
            return answerObjectMethod(proxy, invoked, args);
        }
        hook.callbackFor(invoked).ifPresent(callback -> hear(callback, args));
        return null;
    }

    /** Hand {@code callback}'s arguments to the body once the call brought every one it needs. */
    private void hear(Callback callback, Object[] args) {
        int given = args == null ? 0 : args.length;
        if (given < callback.arity()) {
            HookFailures.countShortCall(hook, callback.methodName(), given, callback.arity());
            return;
        }
        try {
            body.on(callback, new Args(args));
        } catch (RuntimeException failure) {
            HookFailures.count(hook, failure);
            throw failure;
        }
    }

    /** The proxy's own identity, as any object answers it: a collection can hold the proxy. */
    private Object answerObjectMethod(Object proxy, Method invoked, Object[] args) {
        return switch (invoked.getName()) {
            case "hashCode" -> System.identityHashCode(proxy);
            case "equals" -> proxy == args[0];
            default -> "listener on " + hook;
        };
    }

    /** Install this handler on its own hook, or do nothing; see {@link HookReflection#install}. */
    boolean install() {
        return hook.install(this);
    }

    /** Remove whatever listener this handler's hook holds, or do nothing. */
    void uninstall() {
        hook.uninstall();
    }
}
