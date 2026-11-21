package com.pricepredictor.connector.effects;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Reaches the engine patch's hooks without linking against them.
 *
 * <p>{@code forge-connector} compiles against <b>stock</b> Forge, and a worker
 * must degrade rather than fail when the patch is absent. A collector that
 * called a patched method directly would not compile on an unpatched checkout,
 * so every hook is looked up by name and every listener is installed through a
 * dynamic proxy over the interface the patch declares.
 *
 * <p>The cost is that a typo in a hook's name shows up at runtime as "degraded"
 * rather than at compile time as an error. {@link AttributionMode} reports which
 * was detected, and the integration test asserts the mode, which is what keeps
 * that honest.
 *
 * <p>Every lookup is cached, absent ones included: the classpath is fixed for
 * the life of the JVM, so an answer never changes.
 */
final class HookReflection {

    private HookReflection() {
    }

    /** A method's site and reach, the key {@link #find} looks up by. */
    private record Named(HookSite site, Reach reach) {
    }

    /**
     * Every lookup made so far, absent ones included, by site and reach.
     *
     * <p>The sub-ability pointer is read through here for every event the
     * collectors stamp.
     *
     * <p>Keyed by name only, and kept apart from {@link #STATIC_OVERLOADS} on
     * purpose: a listener setter's parameter type is an interface the patch
     * declares, which this side never compiles against and so cannot name in
     * a signature.
     */
    private static final Map<Named, Optional<Method>> LOOKUPS = new ConcurrentHashMap<>();

    /**
     * Find a method by its site, among the methods {@code reach} sees, or
     * report it absent.
     *
     * <p>By name alone, the first match of any arity: the setters and the
     * pointer the patch adds have no overload to be confused with.
     */
    static Optional<Method> find(HookSite site, Reach reach) {
        return LOOKUPS.computeIfAbsent(new Named(site, reach), HookReflection::lookUp);
    }

    private static Optional<Method> lookUp(Named named) {
        try {
            for (Method method : named.reach().candidates(Class.forName(named.site().owner()))) {
                if (method.getName().equals(named.site().method())) {
                    return Optional.of(named.reach().reachable(method));
                }
            }
        } catch (ClassNotFoundException | LinkageError | RuntimeException ignored) {
            // An unpatched checkout, or a Forge upgrade that moved the class;
            // or a private method a module path refuses access to.
        }
        return Optional.empty();
    }

    /**
     * Install a listener on a patched setter, or do nothing.
     *
     * <p>The listener is a dynamic proxy over the interface the patch declares,
     * so this side needs no compile-time knowledge of that interface's shape —
     * only of the method names it will be called with.
     *
     * @return whether the hook was found and the listener installed
     */
    static boolean install(Optional<Method> setter, InvocationHandler handler) {
        if (setter.isEmpty() || setter.get().getParameterCount() != 1) {
            return false;
        }
        Class<?> listenerType = setter.get().getParameterTypes()[0];
        if (!listenerType.isInterface()) {
            return false;
        }
        Object proxy = Proxy.newProxyInstance(
                listenerType.getClassLoader(),
                new Class<?>[]{listenerType},
                handler);
        return call(setter, null, proxy) != FAILED;
    }

    /**
     * Remove a listener the same way it was installed, or do nothing: there
     * is nothing to undo on an unpatched checkout.
     */
    static void uninstall(Optional<Method> setter) {
        call(setter, null, (Object) null);
    }

    /** Call a looked-up static method, or answer null when it is absent or the call failed. */
    static Object callStatic(Optional<Method> method, Object... args) {
        return invokeOrNull(method, null, args);
    }

    /**
     * Call a static method the caller cannot do without, rethrowing what it
     * throws rather than reading it as absence.
     */
    static Object callThrough(Method method, Object... args) {
        try {
            return method.invoke(null, args);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        } catch (InvocationTargetException e) {
            throw e.getCause() instanceof RuntimeException runtime ? runtime
                    : new IllegalStateException(e.getCause());
        }
    }

    /** A static method's site and exact parameter types. */
    record Overload(HookSite site, List<Class<?>> parameterTypes) {
    }

    /**
     * Call a patched static method, or answer null when it is absent.
     *
     * <p>Looked up by exact parameter types rather than by name alone, because
     * these hooks are overloads of methods stock Forge already has: calling the
     * wrong arity would throw rather than degrade.
     */
    static Object invokeStatic(Overload overload, Object... args) {
        Optional<Method> method = STATIC_OVERLOADS.computeIfAbsent(overload, HookReflection::staticOverload);
        return invokeOrNull(method, null, args);
    }

    /** {@link #invokeStatic}'s lookups, absent ones included. */
    private static final Map<Overload, Optional<Method>> STATIC_OVERLOADS =
            new ConcurrentHashMap<>();

    private static Optional<Method> staticOverload(Overload overload) {
        try {
            return Optional.of(Class.forName(overload.site().owner()).getMethod(
                    overload.site().method(), overload.parameterTypes().toArray(Class<?>[]::new)));
        } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
            return Optional.empty();
        }
    }

    /** An instance method's name and exact parameter types. */
    record Shape(String method, List<Class<?>> parameterTypes) {

        /** A no-argument method's parameter list, built once rather than per entity. */
        static final List<Class<?>> NO_ARGUMENT = List.of();
    }

    /**
     * Public methods by receiver class, absent ones included, keyed by name
     * and parameter types.
     *
     * <p>A class's methods never change, and {@link #read} runs for every
     * entity of every snapshot. A {@link ClassValue} rather than a map keyed
     * by class, so the cache never keeps a receiver class or its loader alive.
     */
    private static final ClassValue<Map<Shape, Optional<Method>>> INSTANCE_METHODS =
            new ClassValue<>() {
                @Override
                protected Map<Shape, Optional<Method>> computeValue(Class<?> type) {
                    return new ConcurrentHashMap<>();
                }
            };

    /** A public method of {@code type}, or empty when it has none of that shape. */
    private static Optional<Method> instanceMethod(Class<?> type, Shape shape) {
        return INSTANCE_METHODS.get(type).computeIfAbsent(shape, wanted -> {
            try {
                return Optional.of(type.getMethod(
                        wanted.method(), wanted.parameterTypes().toArray(Class<?>[]::new)));
            } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
                return Optional.empty();
            }
        });
    }

    /**
     * Call a method of {@code shape} on an instance, or answer null when the
     * instance is null, the method absent or the call failed.
     *
     * <p>Looked up on the receiver's own class, so both the method and its
     * declaring class have to be public — which is why the patch widens
     * {@code Card.CardColor} rather than leaving this side to force access.
     * Forcing it would work today and break the first time Forge is run on a
     * module path, and the failure would be a silently empty channel.
     */
    static Object read(Object target, Shape shape, Object... args) {
        if (target == null) {
            return null;
        }
        return invokeOrNull(instanceMethod(target.getClass(), shape), target, args);
    }

    /**
     * What {@link #call} answers for an absent method or a call that threw,
     * kept distinct from null because null is also what a void setter answers
     * on success.
     */
    private static final Object FAILED = new Object();

    /**
     * Invoke a looked-up method, or answer {@link #FAILED} when the lookup
     * came back empty or the call threw.
     *
     * <p>Every reflective call into the patch but {@link #callThrough} goes
     * through here, because a hook that is present but misbehaving must
     * degrade the way an absent one does rather than throw out of a
     * collector, or out of the engine's own call into a listener.
     *
     * @param receiver the instance, or null for a static method
     */
    private static Object call(Optional<Method> method, Object receiver, Object... args) {
        if (method.isEmpty()) {
            return FAILED;
        }
        try {
            return method.get().invoke(receiver, args);
        } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
            return FAILED;
        }
    }

    /** {@link #call}, answering null for an absent method or a failed call. */
    private static Object invokeOrNull(Optional<Method> method, Object receiver, Object... args) {
        Object result = call(method, receiver, args);
        return result == FAILED ? null : result;
    }
}
