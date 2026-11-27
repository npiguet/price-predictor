package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;

/**
 * The event an effect API promises, built from the clause that ran.
 *
 * <p>Most of Forge's effects broadcast nothing. Damage and zone changes reach a
 * collector because the engine fires a bus event for them; an extra turn, a
 * prevented damage, a chosen colour reach nobody, and without an event of
 * their own those abilities read as abilities that did nothing. What every
 * one of them does carry is
 * its own {@code SpellAbility}: the parameters it was scripted with, and for
 * some, a result remembered on the host.
 *
 * <p>Keyed by API name rather than by {@code ApiType} member so a Forge rename
 * costs a missing event rather than a compile error, and so this file names no
 * member of an enum whose members move. A name that is no {@code ApiType}
 * member matches nothing, silently, so {@code ApiEventsTest} checks every key
 * against the enum.
 *
 * <p>An API absent from this table emits nothing. That is the right default:
 * damage, counters, taps and zone changes already arrive on the bus, and a
 * second event for them would double-count an outcome.
 */
final class ApiEvents {

    private ApiEvents() {
    }

    /** What an emitter needs from before the clause ran. */
    @FunctionalInterface
    interface Memo<M> {
        M take(SpellAbility sa, Card host);
    }

    /**
     * The event, given the clause and whatever its memo captured; empty when
     * the clause did nothing its API promises.
     *
     * <p>The memo is null when the rule has none and when it threw, so an
     * emitter that reads one checks for null.
     */
    @FunctionalInterface
    interface Emitter<M> {
        Optional<EffectEvent> emit(SpellAbility sa, Card host, M memo);
    }

    /**
     * The events, given the clause and whatever its memo captured: one per
     * object the clause made or changed, where each is an outcome of its own
     * (an effect card, a healed permanent). Empty when the clause did nothing
     * its API promises. The memo is null as for {@link Emitter}.
     */
    @FunctionalInterface
    interface Emitters<M> {
        List<EffectEvent> emit(SpellAbility sa, Card host, M memo);
    }

    /**
     * @param <M> what the memo takes and the emitter reads; {@code Object}
     *            for a rule with no memo
     * @param eventType named in the once-only failure line an emitter's throw prints
     */
    record Rule<M>(EventType eventType, Memo<M> memo, Emitters<M> emitters) {

        /** A rule that writes at most one event per clause: {@code emitter}'s, unless empty. */
        static <M> Rule<M> one(EventType eventType, Memo<M> memo, Emitter<M> emitter) {
            return new Rule<>(eventType, memo,
                    (sa, host, taken) -> emitter.emit(sa, host, taken).stream().toList());
        }

        /** A rule whose emitter fills in a fresh event of {@code eventType}. */
        static <M> Rule<M> shaped(EventType eventType, Memo<M> memo, Shaper<M> shaper) {
            return one(eventType, memo, (sa, host, taken) ->
                    shaper.shape(new EffectEvent(eventType), sa, host, taken));
        }

        /** A rule that writes one event per object its clause made or changed. */
        static <M> Rule<M> each(EventType eventType, Memo<M> memo, Emitters<M> emitters) {
            return new Rule<>(eventType, memo, emitters);
        }
    }

    /**
     * An emitter handed an event of its rule's own type to fill in, so a rule
     * names that type once.
     */
    @FunctionalInterface
    interface Shaper<M> {
        Optional<EffectEvent> shape(EffectEvent event, SpellAbility sa, Card host, M memo);
    }

    /**
     * One clause between its two hooks: the rule its API names, and what that
     * rule's memo took before the clause ran.
     *
     * <p>What {@code ClauseOutcomeRouter} keeps from {@code onClauseResolving}
     * to {@code onClauseResolved}, so the rule whose memo was taken is the
     * rule whose emitter reads it, at the type that memo has.
     *
     * @param <M> the rule's memo type
     */
    static final class ClauseObservation<M> {

        /** A clause whose API promises no event. */
        static final ClauseObservation<Object> NOTHING = new ClauseObservation<>(null, null, null);

        private final SpellAbility clause;
        private final Rule<M> rule;
        private final M memo;

        private ClauseObservation(SpellAbility clause, Rule<M> rule, M memo) {
            this.clause = clause;
            this.rule = rule;
            this.memo = memo;
        }

        /** What the memo took: null where there is no rule, no memo, or the memo threw. */
        M memo() {
            return memo;
        }

        /**
         * The events this clause promises: none when its API promises none
         * or its effect did nothing, one for most APIs, one per object made
         * or changed for a {@link Rule#each} rule.
         *
         * <p>Contained: an emitter reads engine state that a fizzled or
         * redirected clause may have left in a shape it did not expect. A
         * missing event is a gap; a thrown one would end the resolution the
         * observer is only watching -- so the throw stays swallowed. But
         * swallowed and "this mechanic never happens" must not look
         * identical, so the first one is not silent.
         */
        List<EffectEvent> finish() {
            if (rule == null) {
                return List.of();
            }
            try {
                return rule.emitters().emit(clause, clause.getHostCard(), memo);
            } catch (RuntimeException e) {
                reportEmitterFailure(rule.eventType(), e);
                return List.of();
            }
        }
    }

    /**
     * The rule for each API name.
     *
     * <p>Built when the class initialises, so every memo and emitter the table
     * holds is a static method, referenced rather than read: a {@code static
     * final} field declared below this one would still be null when {@link
     * #rules} runs.
     */
    private static final Map<String, Rule<?>> RULES = rules();

    /**
     * The rule for this clause, and what its memo takes before the clause runs.
     *
     * <p>Contained the way {@link ClauseObservation#finish} is: a memo reads
     * engine state before the clause has resolved, which can throw on an
     * edge-case board the same way an emitter can. Left uncontained, the throw
     * would propagate out of this call before {@code
     * ClauseOutcomeRouter.clauseHandler()}'s own push ran, so the {@code Deque}
     * that hook pairs pushes and pops through would go one push short: the
     * next sibling or enclosing clause's {@code onClauseResolved} would pop
     * <em>this</em> clause's absent observation and get someone else's instead,
     * and the clause that should have received that observation would pop an
     * empty deque -- one throw, two clauses misattributed.
     *
     * <p>Catching here and always returning an observation keeps that push
     * unconditional: a memo that failed reads back as a {@code null} memo,
     * exactly as a memo-less rule's does, which is what keeps the deque
     * balanced.
     *
     * @param rules the rule for each API, {@link #ruleFor} outside a test
     */
    static ClauseObservation<?> observe(SpellAbility sa, Function<ApiType, Rule<?>> rules) {
        Rule<?> rule = sa == null ? null : rules.apply(sa.getApi());
        return rule == null ? ClauseObservation.NOTHING : observe(sa, rule);
    }

    private static <M> ClauseObservation<M> observe(SpellAbility sa, Rule<M> rule) {
        M memo = null;
        if (rule.memo() != null) {
            try {
                memo = rule.memo().take(sa, sa.getHostCard());
            } catch (RuntimeException e) {
                // Own report, own latch -- a memo that fails must not look
                // identical to "this mechanic never happens".
                reportMemoFailure(rule.eventType(), e);
            }
        }
        return new ClauseObservation<>(sa, rule, memo);
    }

    /**
     * This clause's rule with no memo taken, for an {@code onClauseResolved}
     * that met no observation of its own -- one whose {@code
     * onClauseResolving} this listener never saw.
     *
     * @param rules the rule for each API, {@link #ruleFor} outside a test
     */
    static ClauseObservation<?> unobserved(SpellAbility sa, Function<ApiType, Rule<?>> rules) {
        Rule<?> rule = sa == null ? null : rules.apply(sa.getApi());
        return rule == null ? ClauseObservation.NOTHING : new ClauseObservation<>(sa, rule, null);
    }

    /** Raised the first time any rule's emitter throws. */
    private static final OnceWarning EMITTER_FAILURE = new OnceWarning();

    /**
     * Mirror the treatment Forge's {@code EffectRecordHook.contain} gives
     * this same hook's other end: latched to the first throw, naming what
     * failed and why, rather than repeated once per clause -- a real
     * playthrough resolves thousands of them, and a broken rule would flood
     * the log for the whole run instead of announcing itself once.
     *
     * <p>Package-private rather than private, the same reason {@code
     * clauseHandler} is: forcing a rule to throw reliably would mean
     * contriving a malformed script value or a null host card, so a test
     * calls this with a synthetic exception instead -- as Forge's {@code
     * OnceOnlyWarningTest} raises its warning directly rather than through a
     * real listener throw.
     */
    static void reportEmitterFailure(EventType eventType, RuntimeException e) {
        EMITTER_FAILURE.raise(() -> "ApiEvents: the emitter for \"" + eventType.wireName()
                + "\" threw and was ignored; further emitter failures will "
                + "not be logged: " + e);
    }

    /** Raised the first time any rule's memo throws. */
    private static final OnceWarning MEMO_FAILURE = new OnceWarning();

    /**
     * As {@link #reportEmitterFailure}, for a rule's memo instead of its
     * emitter.
     *
     * <p>Its own message, because a memo ({@link #observe}) and an emitter
     * ({@link ClauseObservation#finish}) are different halves of one {@link Rule}, and a memo
     * failure reported as "the emitter" sends its reader to the wrong one.
     * Its own latch, because a memo failure raising
     * {@link #EMITTER_FAILURE} would silence a later, unrelated emitter
     * failure for the rest of the JVM.
     */
    static void reportMemoFailure(EventType eventType, RuntimeException e) {
        MEMO_FAILURE.raise(() -> "ApiEvents: the memo for \"" + eventType.wireName()
                + "\" threw and was ignored; further memo failures will "
                + "not be logged: " + e);
    }

    /**
     * The rule for an API, or null when it promises no event.
     *
     * <p>What {@link #observe} is handed outside a test.
     * A caller hands them its own lookup instead to stand a rule that throws
     * in for a real one: no real script is known to make a memo or an emitter
     * throw, so that is the only way to prove the containment around them.
     */
    static Rule<?> ruleFor(ApiType api) {
        return api == null ? null : RULES.get(api.name());
    }

    /** Every API name the table has a rule for. */
    static Set<String> apiNames() {
        return RULES.keySet();
    }

    /** The table, one family of APIs at a time. */
    private static Map<String, Rule<?>> rules() {
        Map<String, Rule<?>> rules = new HashMap<>();
        TurnAndPhaseApiRules.turnAndPhaseRules(rules);
        ValueAndOwnershipApiRules.valueAndOwnershipRules(rules);
        ChoiceApiRules.choiceRules(rules);
        HealGrantAndRetargetApiRules.healGrantAndRetargetRules(rules);
        ScheduleAndReplacementApiRules.scheduleAndReplacementRules(rules);
        TitleApiRules.titleRules(rules);
        CombatRemovalApiRules.combatRemovalRules(rules);
        TextApiRules.textRules(rules);
        CombatEntryAndSearchApiRules.combatEntryAndSearchRules(rules);
        PermanentStateApiRules.permanentStateRules(rules);
        return Map.copyOf(rules);
    }

    /** One rule for each API that reports the same event the same way. */
    static void put(Map<String, Rule<?>> into, Rule<?> rule, String... apis) {
        for (String api : apis) {
            if (into.putIfAbsent(api, rule) != null) {
                throw new IllegalStateException("two rules for " + api);
            }
        }
    }

    // ── readings several families share ─────────────────────────────────

    static int amount(SpellAbility sa, Card host, String key, String fallback) {
        return AbilityUtils.calculateAmount(host, sa.getParamOrDefault(key, fallback), sa);
    }

    static String playerOf(SpellAbility sa) {
        return Refs.ofNullable(sa.getActivatingPlayer());
    }
}
