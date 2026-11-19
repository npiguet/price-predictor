package com.pricepredictor.connector.effects;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
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
    static final String COMBAT_LEGALITY_OBSERVER =
            "forge.game.combat.CombatLegalityObserver";
    static final String ABILITY_MANA_PART =
            "forge.game.spellability.AbilityManaPart";
    static final String CANT_ATTACK_BLOCK =
            "forge.game.staticability.StaticAbilityCantAttackBlock";
    static final String CARD = "forge.game.card.Card";
    static final String COMBAT = "forge.game.combat.Combat";
    static final String PLAYER_CONTROLLER_AI = "forge.ai.PlayerControllerAi";
    static final String PHASE_HANDLER = "forge.game.phase.PhaseHandler";
    static final String MAGIC_STACK = "forge.game.zone.MagicStack";
    static final String COST_PAYMENT = "forge.game.cost.CostPayment";
    /** The enum the clause hook's "resolved" call carries, read by constant name. */
    static final String EFFECT_RECORD_HOOK = "forge.game.ability.EffectRecordHook";
    static final String CLAUSE_RUN = "forge.game.ability.ClauseRun";

    /**
     * One method the engine patch adds, and what having it buys.
     *
     * @param owner     the Forge class it is declared on
     * @param method    its name, which is the whole contract — the collectors
     *                  look it up by string and a rename degrades a channel
     *                  silently
     * @param unlocks   what stops working without it
     * @param callbacks for a listener setter, the methods of the listener it
     *                  takes, which are as much the contract as the setter's
     *                  own name; empty for every other hook
     * @param reach     where the method is looked up
     */
    public record Hook(String owner, String method, String unlocks, List<Callback> callbacks, Reach reach) {

        /** A public hook that takes no listener. */
        Hook(String owner, String method, String unlocks) {
            this(owner, method, unlocks, List.of());
        }

        /** A public hook. */
        Hook(String owner, String method, String unlocks, List<Callback> callbacks) {
            this(owner, method, unlocks, callbacks, Reach.PUBLIC);
        }

        boolean present() {
            return lookUp().isPresent();
        }

        /** The method itself, found the way {@link PatchHooks#find} finds it, or among the private ones. */
        Optional<Method> lookUp() {
            return LOOKUPS.computeIfAbsent(new Named(owner, method, reach), PatchHooks::lookUp);
        }

        /** The interface this setter's listener is declared as, or empty for a hook that takes none. */
        Optional<Class<?>> listenerType() {
            return lookUp()
                    .filter(setter -> setter.getParameterCount() == 1)
                    .map(setter -> setter.getParameterTypes()[0]);
        }

        /**
         * What is wrong with this hook on this checkout, or empty: absent, or
         * its listener lacking a callback or an argument the collectors read.
         */
        Optional<String> gap() {
            if (!present()) {
                return Optional.of(this + " — " + unlocks);
            }
            List<String> lacking = listenerType()
                    .map(type -> callbacksLacking(type, callbacks)).orElse(List.of());
            return lacking.isEmpty() ? Optional.empty()
                    : Optional.of(this + " — " + unlocks + "; its listener: " + String.join(", ", lacking));
        }

        /**
         * Call this static method and answer what it returns, for a hook a
         * caller cannot do without: absent, it throws naming itself, and what
         * the method throws is rethrown rather than read as absence.
         */
        Object call(Object... args) {
            Method found = lookUp().orElseThrow(() -> new IllegalStateException(
                    "this Forge lacks " + this + ": " + unlocks));
            try {
                return found.invoke(null, args);
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            } catch (InvocationTargetException e) {
                throw e.getCause() instanceof RuntimeException runtime ? runtime
                        : new IllegalStateException(e.getCause());
            }
        }

        /** Install a listener on this setter, or do nothing; see {@link PatchHooks#install}. */
        boolean install(InvocationHandler handler) {
            return PatchHooks.install(owner, method, handler);
        }

        /** Remove the listener this setter holds, or do nothing. */
        void uninstall() {
            PatchHooks.uninstall(owner, method);
        }

        /**
         * Whether this setter's listener interface declares a method named
         * {@code name}: a callback an older jar never makes, which a channel
         * relying on it must do without.
         */
        boolean listenerDeclares(String name) {
            return listenerType().map(type -> mostArguments(type, name) >= 0).orElse(false);
        }

        /**
         * A handler for this setter's listener, which hands {@code body} each
         * of the {@link #callbacks} the engine makes, and which installs on
         * this setter and no other.
         */
        Listener listener(ListenerBody body) {
            return new Listener(this, body);
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

        /**
         * Call this one-argument lend on {@code target}: what gives back what
         * it lent, or null when it lent nothing or the hook is absent.
         */
        Runnable lend(Object target, Class<?> parameterType, Object argument) {
            return PatchHooks.invoke(target, method, parameterType, argument)
                    instanceof Runnable giveBack ? giveBack : null;
        }

        @Override
        public String toString() {
            return owner.substring(owner.lastIndexOf('.') + 1) + "." + method;
        }
    }

    /**
     * One method of a listener interface the patch declares: the name the
     * engine calls it by, and the arguments a handler needs before it acts.
     *
     * <p>The arity is a floor rather than the declared count. A revision that
     * widened a listener appended its new arguments, and a worker still runs
     * against an older jar, so a handler reads those through
     * {@link Args#optional} and the shorter call still reaches it.
     *
     * <p>{@link #read} counts those optional arguments too: what a current
     * Forge passes, and what {@link PatchHooks#report()} checks each
     * listener against, so a shorter one is named rather than read whole.
     */
    enum Callback {
        ON_REPLACING("onReplacing", 2),
        ON_REPLACEMENT("onReplacement", 3, 6),
        ON_BUFFERED_ABILITY_STARTING("onBufferedAbilityStarting", 2),
        ON_BUFFERED_ABILITY_FINISHED("onBufferedAbilityFinished", 2),
        ON_CONDITION_EVALUATED("onConditionEvaluated", 3),
        ON_TRIGGER_RUN("onTriggerRun", 2),
        ON_ATTACKERS_DECLARABLE("onAttackersDeclarable", 4),
        ON_BLOCKERS_DECLARABLE("onBlockersDeclarable", 5),
        ON_CLAUSE_RESOLVING("onClauseResolving", 1),
        ON_CLAUSE_RESOLVED("onClauseResolved", 1, 2),
        ON_MANA_PRODUCED("onManaProduced", 3),
        ON_CONFIRM("onConfirm", 2),
        ON_OUTCOME("onOutcome", 3),
        ON_STATE_BASED_LOOP_BEGIN("onStateBasedLoopBegin", 1),
        ON_STATE_BASED_LOOP_END("onStateBasedLoopEnd", 1),
        ON_PRIORITY_SETTLED("onPrioritySettled", 2),
        ON_CAST_STARTING("onCastStarting", 1),
        ON_RESOLUTION_STARTING("onResolutionStarting", 2, 3),
        ON_PAYING_COSTS("onPayingCosts", 3),
        ON_COSTS_PAID("onCostsPaid", 3);

        private final String methodName;
        private final int arity;
        private final int read;

        /** A callback whose handler reads no optional argument. */
        Callback(String methodName, int arity) {
            this(methodName, arity, arity);
        }

        Callback(String methodName, int arity, int read) {
            this.methodName = methodName;
            this.arity = arity;
            this.read = read;
        }

        String methodName() {
            return methodName;
        }

        int arity() {
            return arity;
        }

        /** The arguments a handler reads, the optional ones past {@link #arity} included. */
        int read() {
            return read;
        }
    }

    /**
     * What {@code listener} lacks of {@code callbacks}, one phrase each: a
     * callback it declares no method for, or declares with fewer arguments
     * than the handler reads. A widened listener declares its old call and
     * its new one under one name, so the longest counts.
     */
    static List<String> callbacksLacking(Class<?> listener, List<Callback> callbacks) {
        List<String> lacking = new ArrayList<>();
        for (Callback callback : callbacks) {
            int passed = mostArguments(listener, callback.methodName());
            if (passed < 0) {
                lacking.add(callback.methodName() + " is not declared");
            } else if (passed < callback.read()) {
                lacking.add(callback.methodName() + " passes " + passed + " of the "
                        + callback.read() + " arguments the collectors read");
            }
        }
        return lacking;
    }

    /** The most parameters a public method {@code name} of {@code type} takes, or -1 when it has none. */
    private static int mostArguments(Class<?> type, String name) {
        return java.util.Arrays.stream(type.getMethods())
                .filter(declared -> declared.getName().equals(name))
                .mapToInt(Method::getParameterCount).max().orElse(-1);
    }

    /**
     * A handler built for one hook's listener, carrying that hook.
     *
     * <p>Carrying it is what lets a collector install the handler without
     * naming the hook a second time. A handler built from one hook and
     * installed on another hears none of the callbacks it dispatches on, and
     * the channel behind it goes empty without a word; built and installed
     * through the same {@link Hook}, the two cannot disagree.
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
    record Listener(Hook hook, ListenerBody body) implements InvocationHandler {

        @Override
        public Object invoke(Object proxy, Method invoked, Object[] args) {
            if (invoked.getDeclaringClass() == Object.class) {
                return answerObjectMethod(proxy, invoked, args);
            }
            for (Callback callback : hook.callbacks()) {
                if (callback.methodName().equals(invoked.getName())) {
                    int given = args == null ? 0 : args.length;
                    if (given < callback.arity()) {
                        HookFailures.countShortCall(
                                hook, callback.methodName(), given, callback.arity());
                        return null;
                    }
                    try {
                        body.on(callback, new Args(args));
                    } catch (RuntimeException failure) {
                        HookFailures.count(hook, failure);
                        throw failure;
                    }
                    return null;
                }
            }
            return null;
        }

        /** The proxy's own identity, as any object answers it: a collection can hold the proxy. */
        private Object answerObjectMethod(Object proxy, Method invoked, Object[] args) {
            return switch (invoked.getName()) {
                case "hashCode" -> System.identityHashCode(proxy);
                case "equals" -> proxy == args[0];
                default -> "listener on " + hook;
            };
        }

        /** Install this handler on its own hook, or do nothing; see {@link PatchHooks#install}. */
        boolean install() {
            return hook.install(this);
        }

        /** Remove whatever listener this handler's hook holds, or do nothing. */
        void uninstall() {
            hook.uninstall();
        }
    }

    /** What a listener does with one callback, once its arguments are there. */
    @FunctionalInterface
    interface ListenerBody {
        void on(Callback callback, Args args);
    }

    /**
     * One callback's arguments, read by position.
     *
     * <p>By position because nothing here compiles against the listener
     * interface. A typed read answers null for a value of another type, the
     * way an {@code instanceof} test does, so a drifted signature costs the
     * argument rather than throwing out of the engine's call.
     */
    static final class Args {
        private final Object[] values;

        private Args(Object[] values) {
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

    /** The resolving clause, and the hook mode detection probes. */
    static final Hook SUB_ABILITY = new Hook(ABILITY_UTILS, "getEffectRecordSubAbility",
            "per-clause attribution: an event's attributed_to; this is "
            + "the hook mode detection probes");
    static final Hook TRIGGER_LISTENER = new Hook(
            TRIGGER_HANDLER, "setEffectRecordTriggerListener",
            "trigger records, fired and not; a firing is confirmed by the trigger's run",
            List.of(Callback.ON_CONDITION_EVALUATED, Callback.ON_TRIGGER_RUN));
    static final Hook REPLACEMENT_LISTENER = new Hook(
            REPLACEMENT_HANDLER, "setEffectRecordListener",
            "rewrite records: the event a replacement received, what it "
            + "rewrote in place, which of the five ReplacementResults "
            + "it returned and which ability stood in for the event; "
            + "their board, read before the replacement runs; a buffered "
            + "damage ability and the events of its later run",
            List.of(Callback.ON_REPLACING, Callback.ON_REPLACEMENT,
                    Callback.ON_BUFFERED_ABILITY_STARTING,
                    Callback.ON_BUFFERED_ABILITY_FINISHED));
    static final Hook CLAUSE_LISTENER = new Hook(
            ABILITY_UTILS, "setEffectRecordClauseListener",
            "the events an effect API describes but the bus never announces",
            List.of(Callback.ON_CLAUSE_RESOLVING, Callback.ON_CLAUSE_RESOLVED));
    static final Hook COMBAT_LISTENER = new Hook(
            COMBAT_LEGALITY_OBSERVER, "setEffectRecordCombatListener",
            "the playability record's attackers and blockers subkinds",
            List.of(Callback.ON_ATTACKERS_DECLARABLE, Callback.ON_BLOCKERS_DECLARABLE));
    static final Hook CANT_ATTACK_STATIC = new Hook(CANT_ATTACK_BLOCK, "cantAttackStatic",
            "each forbidden attacker's responsible_static");
    static final Hook CANT_BLOCK_BY_STATIC = new Hook(CANT_ATTACK_BLOCK, "cantBlockByStatic",
            "each forbidden blocker's responsible_static");
    static final Hook CANT_BLOCK_STATIC = new Hook(CANT_ATTACK_BLOCK, "cantBlockStatic",
            "the responsible_static of a blocker no static lets block at all (Pacifism)");
    static final Hook FAILURE_COUNTS = new Hook(EFFECT_RECORD_HOOK, "failureCounts",
            "the worker's report of the failures Forge contained in every hook slot,"
                    + " argument builds included");
    static final Hook MANA_LISTENER = new Hook(
            ABILITY_MANA_PART, "setEffectRecordManaListener",
            "mana records, and with them the role-polarity probe",
            List.of(Callback.ON_MANA_PRODUCED));
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
    static final Hook PT_WITHOUT = new Hook(CARD, "getNetPowerBreakdownWithout",
            "removing the acting static's set and boosted power and toughness from a continuous snapshot");
    static final Hook CONTROLLER_WITHOUT = new Hook(CARD, "getControllerWithout",
            "removing the acting static's control change from a continuous snapshot");
    static final Hook TRAITS_WITHOUT = new Hook("forge.game.card.CardState", "getSpellAbilitiesWithout",
            "removing the lines the acting static granted, and restoring those it removed,"
                    + " in a continuous snapshot");
    static final Hook CHANGED_NAMES = new Hook(CARD, "getChangedCardNames",
            "a continuous contribution's name");
    static final Hook COPY_SOURCE = new Hook(CARD, "getEffectRecordCopySource",
            "the keyword keys and printed script of a token copy of a copy (Kiki-Jiki's token"
                    + " of a Clone): the card whose printed values its prototype took, read when it was made");
    static final Hook COPIED_PERMANENT = new Hook(CARD, "getEffectRecordCopiedPermanent",
            "the keyword keys of a token copy of a copy whose copy effect added keywords (Kiki-Jiki's"
                    + " token of Dack's Duplicate): the permanent its prototype copied");
    static final Hook CLONE_PRINTER = new Hook("forge.game.card.CardCloneStates", "getEffectRecordPrinter",
            "the keyword keys and printed script of a Clone whose original's copy effect has since ended"
                    + " (a Clone of a Cytoshaped creature after the turn): the card whose printed values it"
                    + " took, noted when it copied");
    static final Hook PRIOR_CONTROLLER = new Hook(CARD, "getEffectRecordPriorController",
            "a continuous record's sickness of a permanent its static took: the controller it"
                    + " had before its most recent control change");
    static final Hook PRIOR_SICKNESS = new Hook(CARD, "isEffectRecordSickForPriorController",
            "a continuous record's sickness of a permanent its static took: its summoning-sickness"
                    + " flag had that controller kept it");
    static final Hook LEND_TYPE = new Hook("forge.game.card.CardState", "lendEffectRecordType",
            "a continuous record's static conditions and untapped_production on the types without"
                    + " its static: a card's type line lent for the read and given back");
    static final Hook SET_ASIDE_COLORS = new Hook(CARD, "setAsideEffectRecordColors",
            "a continuous record's static conditions and untapped_production on the colours without"
                    + " its static: the static's colour changes set aside for the read and put back");
    static final Hook ASSIGNED_DAMAGE = new Hook(COMBAT, "getAssignedDamage",
            "a combat record's assignment_choices");
    static final Hook CONFIRM_LISTENER = new Hook(
            PLAYER_CONTROLLER_AI, "setEffectRecordConfirmListener",
            "the declined outcome: an optional effect the controller "
            + "was offered and turned down",
            List.of(Callback.ON_CONFIRM));
    /**
     * Private to Forge, and called by the directed forks rather than the
     * engine: an answer a fork forces is reported through it, the way the
     * AI's own answers are. On a jar without it, every fork that forces an
     * answer would throw from inside the engine's confirm call.
     */
    static final Hook NOTE_CONFIRM = new Hook(
            PLAYER_CONTROLLER_AI, "noteEffectRecordConfirm",
            "a directed fork's forced answer to an optional effect, reported to the confirm "
            + "listener as an AI answer is; without it every fork that forces one fails",
            List.of(), Reach.PRIVATE);
    static final Hook OUTCOME_LISTENER = new Hook(
            EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener",
            "the outcomes an effect computes and discards: a coin's "
            + "result, a clash's winner, a vote's tally, a pile's size",
            List.of(Callback.ON_OUTCOME));
    static final Hook PENDING_DELAYED_TRIGGERS = new Hook(
            TRIGGER_HANDLER, "getPendingDelayedTriggers",
            "a snapshot's delayed_triggers and the links naming what each "
            + "is pending on; a fork copy's delayed triggers");
    static final Hook SIMULTANEOUS_STACK_ENTRIES = new Hook(
            MAGIC_STACK, "getSimultaneousStackEntries",
            "the collector's own enumeration at a priority stop taking off the triggers it "
            + "queued, and none the game had queued before it");
    static final Hook STATE_BASED_LOOP_LISTENER = new Hook(
            PHASE_HANDLER, "setEffectRecordStateBasedLoopListener",
            "the state-based loop's extent: a resolution's held effect half "
            + "spans it, and every death inside it is filed there",
            List.of(Callback.ON_STATE_BASED_LOOP_BEGIN, Callback.ON_STATE_BASED_LOOP_END));
    static final Hook PRIORITY_SETTLED_LISTENER = new Hook(
            PHASE_HANDLER, "setEffectRecordPrioritySettledListener",
            "the moment both seats' decision candidates are enumerated, "
            + "before the priority holder chooses",
            List.of(Callback.ON_PRIORITY_SETTLED));
    static final Hook CAST_LISTENER = new Hook(
            MAGIC_STACK, "setEffectRecordCastListener",
            "a cost half's board, read before the spell's card moves to the "
            + "stack or the activation's costs are paid (FR-042a)",
            List.of(Callback.ON_CAST_STARTING));
    static final Hook RESOLUTION_LISTENER = new Hook(
            MAGIC_STACK, "setEffectRecordResolutionListener",
            "an effect half's board, read as the resolution starts (FR-042b)",
            List.of(Callback.ON_RESOLUTION_STARTING));
    static final Hook PAYMENT_LISTENER = new Hook(
            COST_PAYMENT, "setEffectRecordPaymentListener",
            "a mana record's board, read before the mana ability's costs are "
            + "paid, and the permanents its cost tapped (FR-042c)",
            List.of(Callback.ON_PAYING_COSTS, Callback.ON_COSTS_PAID));

    /**
     * Every hook the collectors need, and the specification of the patch.
     *
     * <p>This list is the patch's description. The patch itself is a branch in
     * the sibling checkout — {@code effect-record-hooks} in {@code ../forge} —
     * and this repo keeps no exported copy of it, which could disagree with the
     * branch. What a branch cannot hold is *why* each hook exists and what
     * breaks without it, so that lives here, next to the code that reads them.
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
     * <p>Presence is not the whole contract for the {@code set*Listener} hooks.
     * A setter keeps its name when its listener loses a callback or gains
     * arguments, so {@link #report()} also checks each listener interface
     * against its hook's {@link Callback}s, by name and by the arguments the
     * handlers read. A jar whose {@code setEffectRecordListener} listener does
     * not pass the {@code ReplacementResult} is named there; its rewrite
     * records would carry a {@code result} of null on every rewrite. The
     * handlers still read the tail arguments through {@link Args#optional}
     * rather than assuming them, because a partly-current patch must degrade
     * the way an absent one does instead of throwing inside a dynamic proxy.
     */
    public static final List<Hook> REQUIRED = List.of(
            SUB_ABILITY, TRIGGER_LISTENER, REPLACEMENT_LISTENER, CLAUSE_LISTENER,
            COMBAT_LISTENER, CANT_ATTACK_STATIC,
            CANT_BLOCK_BY_STATIC, CANT_BLOCK_STATIC, FAILURE_COUNTS, MANA_LISTENER, CHANGED_TYPES_BY_STATIC,
            CHANGED_COLORS_BY_STATIC, TYPE_WITHOUT, COLOR_WITHOUT, KEYWORDS_WITHOUT,
            PT_WITHOUT, CONTROLLER_WITHOUT, TRAITS_WITHOUT,
            CHANGED_NAMES, COPY_SOURCE, COPIED_PERMANENT, CLONE_PRINTER, PRIOR_CONTROLLER, PRIOR_SICKNESS,
            LEND_TYPE, SET_ASIDE_COLORS, ASSIGNED_DAMAGE, CONFIRM_LISTENER, NOTE_CONFIRM, OUTCOME_LISTENER,
            PENDING_DELAYED_TRIGGERS, SIMULTANEOUS_STACK_ENTRIES, STATE_BASED_LOOP_LISTENER,
            PRIORITY_SETTLED_LISTENER,
            CAST_LISTENER, RESOLUTION_LISTENER, PAYMENT_LISTENER);

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
     * One line per required hook this checkout lacks, or whose listener lacks
     * a callback or an argument the collectors read; empty on a current
     * patched checkout.
     */
    public static List<String> gaps() {
        return gaps(REQUIRED);
    }

    private static List<String> gaps(List<Hook> required) {
        List<String> gaps = new ArrayList<>();
        for (Hook hook : required) {
            hook.gap().ifPresent(gaps::add);
        }
        return gaps;
    }

    /**
     * One line per {@link #gaps() gap}, for a worker to print at startup.
     *
     * <p>A partly-applied patch is the state worth naming: the mode still reads
     * {@code patched} because that is one hook's answer, while a channel this
     * run was meant to collect is quietly empty. A stale installed jar is the
     * likely case, whose setters are all there and whose listeners are older.
     */
    public static String report() {
        return report(REQUIRED);
    }

    static String report(List<Hook> required) {
        List<String> gaps = gaps(required);
        if (gaps.isEmpty()) {
            return "effect-record hooks: all " + required.size() + " present, each listener whole";
        }
        StringBuilder out = new StringBuilder(
                "effect-record hooks: " + gaps.size() + " of " + required.size()
                        + " missing or short, so these channels stay empty or partial:");
        for (String gap : gaps) {
            out.append("\n  ").append(gap);
        }
        return out.toString();
    }

    /**
     * Where a hook's method is looked up: among its owner's public methods,
     * or among its own declared ones, made accessible, for the one private
     * method the patch adds ({@link #NOTE_CONFIRM}).
     */
    enum Reach {
        PUBLIC, PRIVATE;

        Method[] candidates(Class<?> owner) {
            return this == PUBLIC ? owner.getMethods() : owner.getDeclaredMethods();
        }

        Method reachable(Method method) {
            if (this == PRIVATE) {
                method.setAccessible(true);
            }
            return method;
        }
    }

    /** A method's owner, name and reach, the key {@link #find} looks up by. */
    private record Named(String owner, String method, Reach reach) {
    }

    /**
     * Every lookup made so far, absent ones included, by owner, name and reach.
     *
     * <p>The classpath is fixed for the life of the JVM, so an answer never
     * changes, and the sub-ability pointer is read through here for every
     * event the collectors stamp.
     *
     * <p>Keyed by name only, and kept apart from {@link #STATIC_OVERLOADS} on
     * purpose: a listener setter's parameter type is an interface the patch
     * declares, which this side never compiles against and so cannot name in
     * a signature.
     */
    private static final Map<Named, Optional<Method>> LOOKUPS = new ConcurrentHashMap<>();

    /**
     * Find a public static method by owner and name, or report it absent.
     *
     * <p>By name alone, the first match of any arity: the setters and the
     * pointer the patch adds have no overload to be confused with.
     */
    public static Optional<Method> find(String className, String methodName) {
        return LOOKUPS.computeIfAbsent(new Named(className, methodName, Reach.PUBLIC), PatchHooks::lookUp);
    }

    private static Optional<Method> lookUp(Named named) {
        try {
            for (Method method : named.reach().candidates(Class.forName(named.owner()))) {
                if (method.getName().equals(named.method())) {
                    return Optional.of(named.reach().reachable(method));
                }
            }
        } catch (ClassNotFoundException | LinkageError | RuntimeException ignored) {
            // An unpatched checkout, or a Forge upgrade that moved the class;
            // or a private method a module path refuses access to.
        }
        return Optional.empty();
    }

    /** Read a no-argument static getter, or null when the hook is absent. */
    public static Object readStatic(String className, String methodName) {
        return invokeOrNull(find(className, methodName), null);
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
        return call(setter, null, proxy) != FAILED;
    }

    /**
     * Remove a listener the same way it was installed, or do nothing: there
     * is nothing to undo on an unpatched checkout.
     */
    public static void uninstall(String className, String setterName) {
        call(find(className, setterName), null, (Object) null);
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
        return invokeOrNull(method, null, args);
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
        return invokeOrNull(instanceMethod(target.getClass(), methodName, NO_ARGUMENT), target);
    }

    /** Call a one-{@code long}-argument method on an instance, or answer null. */
    public static Object read(Object target, String methodName, long argument) {
        if (target == null) {
            return null;
        }
        return invokeOrNull(
                instanceMethod(target.getClass(), methodName, ONE_LONG), target, argument);
    }

    /**
     * Call a one-argument method on an instance, its parameter declared as
     * {@code parameterType}, or answer null.
     */
    public static Object invoke(
            Object target, String methodName, Class<?> parameterType, Object argument) {
        if (target == null) {
            return null;
        }
        return invokeOrNull(instanceMethod(
                target.getClass(), methodName, List.of(parameterType)), target, argument);
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
     * <p>Every reflective call into the patch goes through here, because a
     * hook that is present but misbehaving must degrade the way an absent one
     * does rather than throw out of a collector, or out of the engine's own
     * call into a listener.
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

    /**
     * The constant name of the {@code ClauseRun} the clause hook's "resolved"
     * call carries, or null where the argument is absent or not an enum: a
     * jar whose listener predates it, or one that passed a boolean.
     */
    static String clauseRunName(Object run) {
        return run instanceof Enum<?> constant ? constant.name() : null;
    }

    /**
     * Forge's own count of the failures each hook slot contained in this JVM
     * ({@code EffectRecordHook.failureCounts()}), by slot name; empty on a
     * checkout without it.
     */
    public static Map<String, Long> forgeFailureCounts() {
        Object counts = readStatic(FAILURE_COUNTS.owner(), FAILURE_COUNTS.method());
        Map<String, Long> read = new java.util.TreeMap<>();
        if (counts instanceof Map<?, ?> map) {
            map.forEach((slot, n) -> {
                if (n instanceof Number number) {
                    read.put(String.valueOf(slot), number.longValue());
                }
            });
        }
        return read;
    }

    /** The sub-ability currently resolving, when the pointer hook is present. */
    public static Object currentSubAbility() {
        return readStatic(SUB_ABILITY.owner(), SUB_ABILITY.method());
    }
}
