package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.Map;

import static com.pricepredictor.connector.effects.ApiEvents.amount;
import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for schedules and replacements: delayed and
 * immediate triggers, and the six {@code Replace*} APIs.
 */
final class ScheduleAndReplacementApiRules {

    private ScheduleAndReplacementApiRules() {
    }

    static void scheduleAndReplacementRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.DELAYED_TRIGGER_CREATED, null, ScheduleAndReplacementApiRules::scheduler),
                "DelayedTrigger");
        put(rules, Rule.shaped(EventType.DELAYED_TRIGGER_CREATED,
                        ScheduleAndReplacementApiRules::triggerAmount, ScheduleAndReplacementApiRules::immediateTrigger),
                "ImmediateTrigger");
        put(rules, Rule.shaped(EventType.REPLACEMENT_APPLIED, null, ScheduleAndReplacementApiRules::replacedBy),
                "ReplaceEffect", "ReplaceCounter", "ReplaceDamage", "ReplaceMana",
                "ReplaceSplitDamage", "ReplaceToken");
    }

    /**
     * delayed_trigger_created: the card that scheduled it. Always the host --
     * neither effect reads {@code Defined$}/{@code ValidTgts$} for the trigger
     * itself, only for what the delayed trigger later acts on.
     */
    private static EffectEvent scheduler(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        return event.subject(Refs.of(host));
    }

    /**
     * How many times an immediate trigger is to fire.
     *
     * <p>{@code ImmediateTriggerEffect.resolve()} returns before registering
     * anything when {@code TriggerAmount$} (default 1) calculates to {@code
     * <= 0} -- CR 603.12a's "once for each of those times" can be zero times.
     * Captured as a memo (read the way {@link ApiEvents#amount} reads it elsewhere)
     * rather than re-read in the emitter, matching this file's own
     * convention: nothing in the effect mutates whatever {@code
     * TriggerAmount$}'s expression depends on before this is the first thing
     * {@code resolve()} computes.
     */
    private static Integer triggerAmount(SpellAbility sa, Card host) {
        return amount(sa, host, "TriggerAmount", "1");
    }

    /** As {@link #scheduler}, for an immediate trigger that fires at least once. */
    private static EffectEvent immediateTrigger(
            EffectEvent event, SpellAbility sa, Card host, Integer times) {
        return times != null && times <= 0 ? null : scheduler(event, sa, host, times);
    }

    /**
     * replacement_applied: the card whose replacement applied, nothing more.
     *
     * <p>The six {@code Replace*} APIs are exactly {@code
     * ReplacementEffect.getOverridingAbility()} -- the ability the replacement
     * substitutes in -- and all six report the same thing, so they share this
     * one emitter.
     *
     * <p>This is a strict subset of the {@code rewrite} record this connector
     * already writes for every replacement, from {@link
     * RewriteChannel#record} -- see the comment on {@code
     * EventType.REPLACEMENT_APPLIED} in {@code event_schema.py} for why the two
     * must not be read as independent evidence of the same replacement.
     */
    private static EffectEvent replacedBy(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        return event.subject(Refs.of(host));
    }
}
