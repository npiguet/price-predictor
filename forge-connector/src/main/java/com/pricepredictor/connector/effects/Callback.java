package com.pricepredictor.connector.effects;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Optional;

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

    /** Whether {@code listener} declares a method by this callback's name. */
    boolean declaredBy(Class<?> listener) {
        return mostArgumentsIn(listener) >= 0;
    }

    /**
     * What {@code listener} lacks of this callback, as a phrase, or empty: it
     * declares no method for it, or declares it with fewer arguments than the
     * handler reads. A widened listener declares its old call and its new one
     * under one name, so the longest counts.
     */
    Optional<String> lackIn(Class<?> listener) {
        int passed = mostArgumentsIn(listener);
        if (passed < 0) {
            return Optional.of(methodName + " is not declared");
        }
        if (passed < read) {
            return Optional.of(methodName + " passes " + passed + " of the "
                    + read + " arguments the collectors read");
        }
        return Optional.empty();
    }

    /** The most parameters a public method of {@code type} by this name takes, or -1 when it has none. */
    private int mostArgumentsIn(Class<?> type) {
        return Arrays.stream(type.getMethods())
                .filter(declared -> declared.getName().equals(methodName))
                .mapToInt(Method::getParameterCount).max().orElse(-1);
    }
}
