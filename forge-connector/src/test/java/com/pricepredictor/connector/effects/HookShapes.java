package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameObject;
import forge.game.ability.AbilityKey;
import forge.game.ability.ClauseRun;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementResult;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;

/**
 * The patched engine's listener interfaces, restated argument for argument.
 *
 * <p>Restated rather than imported: this module compiles against stock Forge,
 * where none of them exists, and a test that imported one would stop the
 * module building on an unpatched checkout. The collectors install their
 * handlers as dynamic proxies for the same reason and read every argument by
 * position, so no compiler checks the order on either side; a test calls a
 * handler through one of these, and {@code PatchHooksTest} checks each one
 * against the declaration the checkout actually has.
 */
final class HookShapes {

    private HookShapes() {
    }

    /** {@code AbilityUtils.EffectRecordClauseListener}. */
    interface Clause {
        void onClauseResolving(SpellAbility clause);

        void onClauseResolved(SpellAbility clause, ClauseRun run);
    }

    /** {@code EffectRecordOutcomes.EffectRecordOutcomeListener}. */
    interface Outcome {
        void onOutcome(SpellAbility sa, String eventType, Map<String, Object> params);
    }

    /** {@code TriggerHandler.EffectRecordTriggerListener}. */
    interface TriggerFire {
        void onConditionEvaluated(Trigger trigger, Map<AbilityKey, Object> runParams, boolean fired);

        void onTriggerRun(Trigger trigger, Map<AbilityKey, Object> runParams, SpellAbility ability);
    }

    /** {@code CombatLegalityObserver.EffectRecordCombatListener}. */
    interface CombatLegality {
        void onAttackersDeclarable(Player decider, GameEntity defender,
                List<Card> candidates, List<Card> legal);

        void onBlockersDeclarable(Player decider, Card attacker,
                List<Card> candidates, List<Card> legal, int minBlockers);
    }

    /** {@code AbilityManaPart.EffectRecordManaListener}. */
    interface Mana {
        void onManaProduced(SpellAbility ability, Player player, String produced);
    }

    /**
     * {@code ReplacementHandler.EffectRecordListener}: Forge calls the
     * six-argument {@code onReplacement}, whose last argument is the ability
     * a damage replacement buffered. Forge's interface also declares the
     * five-argument call as a default it delegates to, restated here because
     * tests call a handler with it, the way an older jar's engine does.
     */
    interface Replacement {
        void onReplacing(ReplacementEffect effect, Map<AbilityKey, Object> runParams);

        void onReplacement(ReplacementEffect effect, Map<AbilityKey, Object> before,
                Map<AbilityKey, Object> after, ReplacementResult result,
                SpellAbility replacedBy, SpellAbility buffered);

        default void onReplacement(ReplacementEffect effect, Map<AbilityKey, Object> before,
                Map<AbilityKey, Object> after, ReplacementResult result,
                SpellAbility replacedBy) {
            onReplacement(effect, before, after, result, replacedBy, null);
        }

        void onBufferedAbilityStarting(ReplacementEffect effect, SpellAbility ability,
                Map<AbilityKey, Object> runParams);

        void onBufferedAbilityFinished(ReplacementEffect effect, SpellAbility ability,
                Map<AbilityKey, Object> runParams);
    }

    /**
     * The same listener before it named the buffered ability: the call a jar
     * without the buffered argument makes. The handler reads the buffered
     * ability past the fifth argument, so the shorter call still reaches it.
     */
    interface FiveArgumentReplacement {
        void onReplacing(ReplacementEffect effect, Map<AbilityKey, Object> runParams);

        void onReplacement(ReplacementEffect effect, Map<AbilityKey, Object> before,
                Map<AbilityKey, Object> after, ReplacementResult result,
                SpellAbility replacedBy);
    }

    /**
     * The same listener before it learnt the result and the ability that
     * stood in for the event.
     *
     * <p>A worker still runs against a jar this old: the setter's name did not
     * change, so nothing warns, and the handler reads the two tail arguments
     * by length. The widening appended them, which is what keeps the first
     * three where the handler reads them in either shape.
     */
    interface OlderReplacement {
        void onReplacement(ReplacementEffect effect, Map<AbilityKey, Object> before,
                Map<AbilityKey, Object> after);
    }

    /** {@code PlayerControllerAi.EffectRecordConfirmListener}. */
    interface Confirm {
        void onConfirm(SpellAbility sa, boolean confirmed);
    }

    /** {@code PhaseHandler.EffectRecordStateBasedLoopListener}. */
    interface StateBasedLoop {
        void onStateBasedLoopBegin(Game game);

        void onStateBasedLoopEnd(Game game);
    }

    /** {@code PhaseHandler.EffectRecordPrioritySettledListener}. */
    interface PrioritySettled {
        void onPrioritySettled(Game game, Player priorityPlayer);
    }

    /** {@code MagicStack.EffectRecordCastListener}. */
    interface Cast {
        void onCastStarting(SpellAbility sa);
    }

    /**
     * {@code MagicStack.EffectRecordResolutionListener}: Forge calls the
     * three-argument {@code onResolutionStarting}, whose last argument is
     * each chain link's targets as cast.
     */
    interface Resolution {
        void onResolutionStarting(SpellAbility sa, boolean fizzled, List<List<GameObject>> targetsAtCast);
    }

    /** The same listener before it handed over the targets as cast. */
    interface OlderResolution {
        void onResolutionStarting(SpellAbility sa, boolean fizzled);
    }

    /** {@code CostPayment.EffectRecordPaymentListener}. */
    interface Payment {
        void onPayingCosts(SpellAbility ability, Player payer, boolean effect);

        void onCostsPaid(SpellAbility ability, Player payer, boolean paid);
    }

    /**
     * A listener setter, the stand-in for what it takes, and any older shape a
     * worker still accepts in its place.
     */
    record Declared(ListenerSetter setter, Class<?> shape, List<Class<?>> older) {

        @Override
        public String toString() {
            return shape.getSimpleName();
        }
    }

    /** Every stand-in above, with the setter whose parameter it restates. */
    static List<Declared> declared() {
        return List.of(
                new Declared(PatchHooks.CLAUSE_LISTENER, Clause.class, List.of()),
                new Declared(PatchHooks.OUTCOME_LISTENER, Outcome.class, List.of()),
                new Declared(PatchHooks.TRIGGER_LISTENER, TriggerFire.class, List.of()),
                new Declared(PatchHooks.COMBAT_LISTENER, CombatLegality.class, List.of()),
                new Declared(PatchHooks.MANA_LISTENER, Mana.class, List.of()),
                new Declared(PatchHooks.REPLACEMENT_LISTENER, Replacement.class,
                        List.of(FiveArgumentReplacement.class, OlderReplacement.class)),
                new Declared(PatchHooks.CONFIRM_LISTENER, Confirm.class, List.of()),
                new Declared(PatchHooks.STATE_BASED_LOOP_LISTENER, StateBasedLoop.class, List.of()),
                new Declared(PatchHooks.PRIORITY_SETTLED_LISTENER, PrioritySettled.class, List.of()),
                new Declared(PatchHooks.CAST_LISTENER, Cast.class, List.of()),
                new Declared(PatchHooks.RESOLUTION_LISTENER, Resolution.class, List.of(OlderResolution.class)),
                new Declared(PatchHooks.PAYMENT_LISTENER, Payment.class, List.of()));
    }

    /** The method {@code shape} declares under {@code name}. */
    static Method method(Class<?> shape, String name) {
        for (Method method : shape.getMethods()) {
            if (method.getName().equals(name)) {
                return method;
            }
        }
        throw new AssertionError(shape + " declares no " + name);
    }

    /** {@link Clause#onClauseResolving}, to invoke a clause handler with. */
    static Method clauseResolving() {
        return method(Clause.class, "onClauseResolving");
    }

    /** {@code handler} behind {@code shape}, as the engine would call it. */
    @SuppressWarnings("unchecked")
    static <T> T proxy(Class<T> shape, InvocationHandler handler) {
        return (T) Proxy.newProxyInstance(shape.getClassLoader(), new Class<?>[]{shape}, handler);
    }
}
