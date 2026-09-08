package com.pricepredictor.connector.effects;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

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
 */
public final class PatchHooks {

    private PatchHooks() {
    }

    static final String TRIGGER_HANDLER = "forge.game.trigger.TriggerHandler";
    static final String REPLACEMENT_HANDLER =
            "forge.game.replacement.ReplacementHandler";
    static final String ABILITY_UTILS = "forge.game.ability.AbilityUtils";
    static final String AI_CONTROLLER = "forge.ai.AiController";
    static final String ABILITY_MANA_PART =
            "forge.game.spellability.AbilityManaPart";
    static final String CANT_ATTACK_BLOCK =
            "forge.game.staticability.StaticAbilityCantAttackBlock";

    /** What a hook lookup produced, or why it did not. */
    public record Lookup(Class<?> owner, Method method) {

        public boolean present() {
            return method != null;
        }

        static Lookup absent() {
            return new Lookup(null, null);
        }
    }

    /** Find a static method by owner and name, or report it absent. */
    public static Lookup find(String className, String methodName) {
        try {
            Class<?> owner = Class.forName(className);
            for (Method method : owner.getMethods()) {
                if (method.getName().equals(methodName)) {
                    return new Lookup(owner, method);
                }
            }
        } catch (ClassNotFoundException | LinkageError ignored) {
            // An unpatched checkout, or a Forge upgrade that moved the class.
        }
        return Lookup.absent();
    }

    /** Read a no-argument static getter, or null when the hook is absent. */
    public static Object readStatic(String className, String methodName) {
        Lookup lookup = find(className, methodName);
        if (!lookup.present()) {
            return null;
        }
        try {
            return lookup.method().invoke(null);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
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
    public static boolean install(
            String className, String setterName, InvocationHandler handler) {
        Lookup lookup = find(className, setterName);
        if (!lookup.present() || lookup.method().getParameterCount() != 1) {
            return false;
        }
        Class<?> listenerType = lookup.method().getParameterTypes()[0];
        if (!listenerType.isInterface()) {
            return false;
        }
        Object proxy = Proxy.newProxyInstance(
                listenerType.getClassLoader(),
                new Class<?>[]{listenerType},
                handler);
        try {
            lookup.method().invoke(null, proxy);
            return true;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return false;
        }
    }

    /** Remove a listener the same way it was installed. */
    public static void uninstall(String className, String setterName) {
        Lookup lookup = find(className, setterName);
        if (!lookup.present()) {
            return;
        }
        try {
            lookup.method().invoke(null, new Object[]{null});
        } catch (ReflectiveOperationException | RuntimeException ignored) {
            // Nothing to undo on an unpatched checkout.
        }
    }

    /**
     * Call a patched static method, or answer null when it is absent.
     *
     * <p>Looked up by exact parameter types rather than by name alone, because
     * these hooks are overloads of methods stock Forge already has: calling the
     * wrong arity would throw rather than degrade.
     */
    public static Object invokeStatic(
            String className, String methodName, Class<?>[] parameterTypes,
            Object... args) {
        try {
            Method method = Class.forName(className)
                    .getMethod(methodName, parameterTypes);
            return method.invoke(null, args);
        } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
            return null;
        }
    }

    /**
     * Read a no-argument method on an instance, or null when it is absent.
     *
     * <p>Looked up on the receiver's own class, so both the method and its
     * declaring class have to be public — which is why the patch widens
     * {@code Card.CardColor} rather than leaving this side to force access.
     * Forcing it would work today and break the first time Forge is run on a
     * module path, and the failure would be a silently empty channel.
     */
    public static Object read(Object target, String methodName) {
        if (target == null) {
            return null;
        }
        try {
            return target.getClass().getMethod(methodName).invoke(target);
        } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
            return null;
        }
    }

    /** Call a one-{@code long}-argument method on an instance, or answer null. */
    public static Object read(Object target, String methodName, long argument) {
        if (target == null) {
            return null;
        }
        try {
            return target.getClass()
                    .getMethod(methodName, long.class)
                    .invoke(target, argument);
        } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
            return null;
        }
    }

    /** Whether every named no-argument-or-long method is present on a class. */
    public static boolean present(String className, String... methodNames) {
        for (String methodName : methodNames) {
            if (!find(className, methodName).present()) {
                return false;
            }
        }
        return true;
    }

    /** The trigger currently running, when the cause hook is present. */
    public static Object currentTriggerCause() {
        return readStatic(TRIGGER_HANDLER, "getEffectRecordCause");
    }

    /** The sub-ability currently resolving, when the pointer hook is present. */
    public static Object currentSubAbility() {
        return readStatic(ABILITY_UTILS, "getEffectRecordSubAbility");
    }
}
