package com.pricepredictor.connector.effects;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;

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
    static final String CARD = "forge.game.card.Card";
    static final String COMBAT = "forge.game.combat.Combat";

    /**
     * One method the engine patch adds, and what having it buys.
     *
     * @param owner   the Forge class it is declared on
     * @param method  its name, which is the whole contract — the collectors look
     *                it up by string and a rename degrades a channel silently
     * @param unlocks what stops working without it
     */
    public record Hook(String owner, String method, String unlocks) {

        boolean present() {
            return find(owner, method).present();
        }

        @Override
        public String toString() {
            return owner.substring(owner.lastIndexOf('.') + 1) + "." + method;
        }
    }

    /**
     * Every hook the collectors need, and the specification of the patch.
     *
     * <p>This list is the patch's description. The patch itself is a branch in
     * the sibling checkout — {@code effect-record-hooks} in {@code ../forge} —
     * because a branch is what git is for; keeping exported {@code .patch} files
     * beside it meant maintaining a second copy of the same history by hand,
     * and the two could disagree. What could not live in a branch is *why* each
     * hook exists and what breaks without it, so that lives here, next to the
     * code that reads them.
     *
     * <p>Reconstructing a lapsed patch starts here: each entry names the class,
     * the method, and the record channel it feeds.
     *
     * <p>One change in the patch is not listed, because it adds no method to
     * probe for: {@code GameAction.destroy} puts the causing ability into the
     * {@code Destroyed} trigger's run parameters, where it was in scope and
     * being dropped. Its absence shows up as a destroy record that cannot name
     * what destroyed the permanent, not as a missing hook.
     */
    public static final List<Hook> REQUIRED = List.of(
            new Hook(TRIGGER_HANDLER, "getEffectRecordCause",
                    "cause attribution; this is the hook mode detection probes"),
            new Hook(TRIGGER_HANDLER, "setEffectRecordTriggerListener",
                    "trigger records, fired and not"),
            new Hook(REPLACEMENT_HANDLER, "setEffectRecordListener",
                    "rewrite records"),
            new Hook(ABILITY_UTILS, "getEffectRecordSubAbility",
                    "per-clause attribution: an event's attributed_to"),
            new Hook(AI_CONTROLLER, "setEffectRecordPlayabilityListener",
                    "playability records, the decision subkind"),
            new Hook(AI_CONTROLLER, "setEffectRecordCombatListener",
                    "the playability record's attackers and blockers subkinds"),
            new Hook(CANT_ATTACK_BLOCK, "cantAttackStatic",
                    "each forbidden attacker's responsible_static"),
            new Hook(CANT_ATTACK_BLOCK, "cantBlockByStatic",
                    "each forbidden blocker's responsible_static"),
            new Hook(ABILITY_MANA_PART, "setEffectRecordManaListener",
                    "mana records, and with them the role-polarity probe"),
            new Hook(CARD, "getChangedCardTypesByStatic",
                    "a continuous contribution's types"),
            new Hook(CARD, "getChangedCardColorsByStatic",
                    "a continuous contribution's colors"),
            new Hook(CARD, "getTypeWithout",
                    "removing the acting static's types from a continuous snapshot"),
            new Hook(CARD, "getColorWithout",
                    "removing the acting static's colours from a continuous snapshot"),
            new Hook(CARD, "getKeywordsWithout",
                    "removing the acting static's keywords from a continuous snapshot"),
            new Hook(COMBAT, "getAssignedDamage",
                    "a combat record's assignment_choices"));

    /** The required hooks this checkout does not have. */
    public static List<Hook> missing() {
        List<Hook> absent = new ArrayList<>();
        for (Hook hook : REQUIRED) {
            if (!hook.present()) {
                absent.add(hook);
            }
        }
        return absent;
    }

    /**
     * One line per missing hook, for a worker to print at startup.
     *
     * <p>A partly-applied patch is the state worth naming: the mode still reads
     * {@code patched} because that is one hook's answer, while a channel this
     * run was meant to collect is quietly empty.
     */
    public static String report() {
        List<Hook> absent = missing();
        if (absent.isEmpty()) {
            return "effect-record hooks: all " + REQUIRED.size() + " present";
        }
        StringBuilder out = new StringBuilder(
                "effect-record hooks: " + absent.size() + " of " + REQUIRED.size()
                        + " missing, so these channels stay empty:");
        for (Hook hook : absent) {
            out.append("\n  ").append(hook).append(" — ").append(hook.unlocks());
        }
        return out.toString();
    }

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
