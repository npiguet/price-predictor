package com.pricepredictor.connector.effects;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
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
 */
public final class PatchHooks {

    private PatchHooks() {
    }

    static final String TRIGGER_HANDLER = "forge.game.trigger.TriggerHandler";
    static final String REPLACEMENT_HANDLER =
            "forge.game.replacement.ReplacementHandler";
    static final String ABILITY_UTILS = "forge.game.ability.AbilityUtils";
    static final String EFFECT_RECORD_OUTCOMES = "forge.game.ability.EffectRecordOutcomes";
    static final String AI_CONTROLLER = "forge.ai.AiController";
    static final String COMBAT_LEGALITY_OBSERVER =
            "forge.game.combat.CombatLegalityObserver";
    static final String ABILITY_MANA_PART =
            "forge.game.spellability.AbilityManaPart";
    static final String CANT_ATTACK_BLOCK =
            "forge.game.staticability.StaticAbilityCantAttackBlock";
    static final String CARD = "forge.game.card.Card";
    static final String COMBAT = "forge.game.combat.Combat";
    static final String PLAYER_CONTROLLER_AI = "forge.ai.PlayerControllerAi";
    static final String EFFECT_RECORD_HOOK = "forge.game.ability.EffectRecordHook";

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
            return lookUp().isPresent();
        }

        /** The method itself, found the way {@link PatchHooks#find} finds it. */
        Optional<Method> lookUp() {
            return find(owner, method);
        }

        /** Install a listener on this setter, or do nothing; see {@link PatchHooks#install}. */
        boolean install(InvocationHandler handler) {
            return PatchHooks.install(owner, method, handler);
        }

        /** Remove the listener this setter holds, or do nothing. */
        void uninstall() {
            PatchHooks.uninstall(owner, method);
        }

        /** Call this static method, or answer null; see {@link PatchHooks#invokeStatic}. */
        Object invoke(Class<?>[] parameterTypes, Object... args) {
            return invokeStatic(owner, method, parameterTypes, args);
        }

        /** Read this no-argument accessor off {@code target}, or answer null. */
        Object read(Object target) {
            return PatchHooks.read(target, method);
        }

        /** Call this one-{@code long}-argument accessor on {@code target}, or answer null. */
        Object read(Object target, long argument) {
            return PatchHooks.read(target, method, argument);
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
     *
     * <p>Presence is not the whole contract for the {@code set*Listener} hooks,
     * and {@code setEffectRecordListener} is where that first bit. Its listener
     * grew from three arguments to five so a rewrite record could say which
     * {@code ReplacementResult} happened and which ability stood in for the
     * event; the setter's name did not change, so this list still reports the
     * hook present against either shape. What tells them apart at runtime is the
     * record: an older jar yields a {@code result} of null on every rewrite. The
     * handlers therefore read the tail arguments by {@code args.length} rather
     * than assuming them, because a partly-current patch must degrade the way an
     * absent one does instead of throwing inside a dynamic proxy.
     */
    public static final List<Hook> REQUIRED;

    /** The resolving clause, and the hook mode detection probes. */
    static final Hook SUB_ABILITY = new Hook(ABILITY_UTILS, "getEffectRecordSubAbility",
            "per-clause attribution: an event's attributed_to; this is "
            + "the hook mode detection probes");
    static final Hook TRIGGER_LISTENER = new Hook(
            TRIGGER_HANDLER, "setEffectRecordTriggerListener",
            "trigger records, fired and not");
    static final Hook REPLACEMENT_LISTENER = new Hook(
            REPLACEMENT_HANDLER, "setEffectRecordListener",
            "rewrite records: the event a replacement received, what it "
            + "rewrote in place, which of the five ReplacementResults "
            + "it returned and which ability stood in for the event");
    static final Hook CLAUSE_LISTENER = new Hook(
            ABILITY_UTILS, "setEffectRecordClauseListener",
            "the events an effect API describes but the bus never announces");
    static final Hook PLAYABILITY_LISTENER = new Hook(
            AI_CONTROLLER, "setEffectRecordPlayabilityListener",
            "playability records, the decision subkind");
    static final Hook COMBAT_LISTENER = new Hook(
            COMBAT_LEGALITY_OBSERVER, "setEffectRecordCombatListener",
            "the playability record's attackers and blockers subkinds");
    static final Hook CANT_ATTACK_STATIC = new Hook(CANT_ATTACK_BLOCK, "cantAttackStatic",
            "each forbidden attacker's responsible_static");
    static final Hook CANT_BLOCK_BY_STATIC = new Hook(CANT_ATTACK_BLOCK, "cantBlockByStatic",
            "each forbidden blocker's responsible_static");
    static final Hook MANA_LISTENER = new Hook(
            ABILITY_MANA_PART, "setEffectRecordManaListener",
            "mana records, and with them the role-polarity probe");
    static final Hook CHANGED_TYPES_BY_STATIC = new Hook(CARD, "getChangedCardTypesByStatic",
            "a continuous contribution's types");
    static final Hook CHANGED_COLORS_BY_STATIC = new Hook(CARD, "getChangedCardColorsByStatic",
            "a continuous contribution's colors");
    static final Hook TYPE_WITHOUT = new Hook(CARD, "getTypeWithout",
            "removing the acting static's types from a continuous snapshot");
    static final Hook COLOR_WITHOUT = new Hook(CARD, "getColorWithout",
            "removing the acting static's colours from a continuous snapshot");
    static final Hook KEYWORDS_WITHOUT = new Hook(CARD, "getKeywordsWithout",
            "removing the acting static's keywords from a continuous snapshot");
    static final Hook CHANGED_NAMES = new Hook(CARD, "getChangedCardNames",
            "a continuous contribution's name");
    static final Hook ASSIGNED_DAMAGE = new Hook(COMBAT, "getAssignedDamage",
            "a combat record's assignment_choices");
    static final Hook CONFIRM_LISTENER = new Hook(
            PLAYER_CONTROLLER_AI, "setEffectRecordConfirmListener",
            "the declined outcome: an optional effect the controller "
            + "was offered and turned down");
    static final Hook OUTCOME_LISTENER = new Hook(
            EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener",
            "the outcomes an effect computes and discards: a coin's "
            + "result, a clash's winner, a vote's tally, a pile's size");

    /**
     * The components of the two layer records {@link #CHANGED_COLORS_BY_STATIC}
     * and {@link #CHANGED_NAMES} hand out.
     *
     * <p>Not in {@link #REQUIRED}, because a lookup cannot tell them from stock
     * Forge's: the records exist there too, private, and a record's accessors
     * are public whatever the record's own access, so {@link #find} would
     * report them present on a checkout where reading them fails. What the
     * patch changes is the records' visibility, and the two accessors that
     * return them are what imply it.
     */
    static final Hook COLOR_CHANGE_ADDITIONAL = new Hook(CARD + "$CardColor", "additional",
            "whether a continuous contribution adds its colours or sets them");
    static final Hook COLOR_CHANGE_COLORS = new Hook(CARD + "$CardColor", "color",
            "a continuous contribution's colours");
    static final Hook NAME_CHANGE_NAME = new Hook(CARD + "$CardChangedName", "newName",
            "a continuous contribution's name");

    static {
        REQUIRED = List.of(
                SUB_ABILITY, TRIGGER_LISTENER, REPLACEMENT_LISTENER, CLAUSE_LISTENER,
                PLAYABILITY_LISTENER, COMBAT_LISTENER, CANT_ATTACK_STATIC,
                CANT_BLOCK_BY_STATIC, MANA_LISTENER, CHANGED_TYPES_BY_STATIC,
                CHANGED_COLORS_BY_STATIC, TYPE_WITHOUT, COLOR_WITHOUT, KEYWORDS_WITHOUT,
                CHANGED_NAMES, ASSIGNED_DAMAGE, CONFIRM_LISTENER, OUTCOME_LISTENER);
    }

    /**
     * The hooks whose listener has thrown in this JVM, which the engine has
     * since been ignoring: each slot reports its first failure once on
     * stderr and carries on, so a channel can go quiet mid-run. Empty on an
     * unpatched checkout.
     */
    @SuppressWarnings("unchecked")
    public static List<String> silencedHooks() {
        Object failed = readStatic(EFFECT_RECORD_HOOK, "failedHooks");
        return failed instanceof List<?> names ? (List<String>) names : List.of();
    }

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

    /** A static method's owner and name, the key {@link #find} looks up by. */
    private record Named(String owner, String method) {
    }

    /**
     * Every lookup made so far, absent ones included, by owner and name.
     *
     * <p>The classpath is fixed for the life of the JVM, so an answer never
     * changes, and the sub-ability pointer is read through here for every
     * event the collectors stamp.
     */
    private static final Map<Named, Optional<Method>> LOOKUPS = new ConcurrentHashMap<>();

    /**
     * Find a public static method by owner and name, or report it absent.
     *
     * <p>By name alone, the first match of any arity: the setters and the
     * pointer the patch adds have no overload to be confused with.
     */
    public static Optional<Method> find(String className, String methodName) {
        return LOOKUPS.computeIfAbsent(new Named(className, methodName), PatchHooks::lookUp);
    }

    private static Optional<Method> lookUp(Named named) {
        try {
            for (Method method : Class.forName(named.owner()).getMethods()) {
                if (method.getName().equals(named.method())) {
                    return Optional.of(method);
                }
            }
        } catch (ClassNotFoundException | LinkageError ignored) {
            // An unpatched checkout, or a Forge upgrade that moved the class.
        }
        return Optional.empty();
    }

    /** Read a no-argument static getter, or null when the hook is absent. */
    public static Object readStatic(String className, String methodName) {
        Optional<Method> getter = find(className, methodName);
        if (getter.isEmpty()) {
            return null;
        }
        try {
            return getter.get().invoke(null);
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
        Optional<Method> setter = find(className, setterName);
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
        try {
            setter.get().invoke(null, proxy);
            return true;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return false;
        }
    }

    /** Remove a listener the same way it was installed. */
    public static void uninstall(String className, String setterName) {
        Optional<Method> setter = find(className, setterName);
        if (setter.isEmpty()) {
            return;
        }
        try {
            setter.get().invoke(null, new Object[]{null});
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
        Optional<Method> method = STATIC_OVERLOADS.computeIfAbsent(
                new Overload(className, methodName, List.of(parameterTypes)),
                PatchHooks::staticOverload);
        if (method.isEmpty()) {
            return null;
        }
        try {
            return method.get().invoke(null, args);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** A static method's owner, name and exact parameter types. */
    private record Overload(String owner, String method, List<Class<?>> parameterTypes) {
    }

    /** {@link #invokeStatic}'s lookups, absent ones included. */
    private static final Map<Overload, Optional<Method>> STATIC_OVERLOADS =
            new ConcurrentHashMap<>();

    private static Optional<Method> staticOverload(Overload overload) {
        try {
            return Optional.of(Class.forName(overload.owner()).getMethod(
                    overload.method(), overload.parameterTypes().toArray(Class<?>[]::new)));
        } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
            return Optional.empty();
        }
    }

    /** An instance method's name and exact parameter types. */
    private record Shape(String method, List<Class<?>> parameterTypes) {
    }

    /** {@link #read}'s two parameter lists, built once rather than per entity. */
    private static final List<Class<?>> NO_ARGUMENT = List.of();
    private static final List<Class<?>> ONE_LONG = List.of(long.class);

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
    private static Optional<Method> instanceMethod(
            Class<?> type, String methodName, List<Class<?>> parameterTypes) {
        return INSTANCE_METHODS.get(type).computeIfAbsent(
                new Shape(methodName, parameterTypes), shape -> {
                    try {
                        return Optional.of(type.getMethod(
                                shape.method(), shape.parameterTypes().toArray(Class<?>[]::new)));
                    } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
                        return Optional.empty();
                    }
                });
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
        Optional<Method> method = instanceMethod(target.getClass(), methodName, NO_ARGUMENT);
        if (method.isEmpty()) {
            return null;
        }
        try {
            return method.get().invoke(target);
        } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
            return null;
        }
    }

    /** Call a one-{@code long}-argument method on an instance, or answer null. */
    public static Object read(Object target, String methodName, long argument) {
        if (target == null) {
            return null;
        }
        Optional<Method> method = instanceMethod(target.getClass(), methodName, ONE_LONG);
        if (method.isEmpty()) {
            return null;
        }
        try {
            return method.get().invoke(target, argument);
        } catch (ReflectiveOperationException | LinkageError | RuntimeException e) {
            return null;
        }
    }

    /** The sub-ability currently resolving, when the pointer hook is present. */
    public static Object currentSubAbility() {
        return readStatic(SUB_ABILITY.owner(), SUB_ABILITY.method());
    }
}
