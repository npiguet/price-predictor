package com.pricepredictor.connector.effects;

import forge.game.GameEntity;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
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
 * member of an enum whose members move. The names are pinned by
 * {@code test_every_effect_api_maps_to_an_event} on the Python side.
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
     * The event, given the clause and whatever its memo captured.
     *
     * <p>The memo is null when the rule has none and when it threw, so an
     * emitter that reads one checks for null.
     */
    @FunctionalInterface
    interface Emitter<M> {
        EffectEvent emit(SpellAbility sa, Card host, M memo);
    }

    /**
     * @param <M> what the memo takes and the emitter reads; {@code Object}
     *            for a rule with no memo
     * @param eventType named in the once-only failure line an emitter's throw prints
     */
    record Rule<M>(EventType eventType, Memo<M> memo, Emitter<M> emitter) {

        /** A rule whose emitter fills in a fresh event of {@code eventType}. */
        static <M> Rule<M> shaped(EventType eventType, Memo<M> memo, Shaper<M> shaper) {
            return new Rule<>(eventType, memo, (sa, host, taken) ->
                    shaper.shape(new EffectEvent(eventType), sa, host, taken));
        }
    }

    /**
     * An emitter handed an event of its rule's own type to fill in, so a rule
     * names that type once.
     */
    @FunctionalInterface
    interface Shaper<M> {
        EffectEvent shape(EffectEvent event, SpellAbility sa, Card host, M memo);
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
         * The event this clause promises, or null when its API promises none.
         *
         * <p>Contained: an emitter reads engine state that a fizzled or
         * redirected clause may have left in a shape it did not expect. A
         * missing event is a gap; a thrown one would end the resolution the
         * observer is only watching -- so the throw stays swallowed. But
         * swallowed and "this mechanic never happens" must not look
         * identical, so the first one is not silent.
         */
        EffectEvent finish() {
            if (rule == null) {
                return null;
            }
            try {
                return rule.emitter().emit(clause, clause.getHostCard(), memo);
            } catch (RuntimeException e) {
                reportEmitterFailure(rule.eventType(), e);
                return null;
            }
        }
    }

    /**
     * Damage on the clause's affected cards, before it heals any of it away.
     *
     * <p>Declared ahead of {@link #RULES} on purpose: the table is built when
     * the class initialises, and a field declared after it would still be
     * null then -- unlike the methods the lambdas below call, which have no
     * such ordering.
     *
     * <p>Read through {@link #affectedCards} rather than off {@code
     * sa.getTargets().getTargetCards()}, the clause's own {@code
     * TargetChoices}. {@code HealDamageEffect.resolve()} finds its cards
     * through {@code
     * SpellAbilityEffect.getTargetCards(sa)}, which consults {@code
     * sa.getTargets()} only when {@code sa.usesTargeting()} is true, and
     * neither real {@code HealDamage} card in the cardsfolder declares any
     * targeting on the clause itself -- both use {@code Defined$
     * Replaced*}, resolved through a replacement effect's {@code
     * getReplacingObject}, which never touches {@code sa.getTargets()}. A
     * direct read would find the clause's own (always-empty, per {@code
     * SpellAbility.targetChosen}'s eager initialisation) {@code
     * TargetChoices} on every real firing and report {@code amount: 0} -- a
     * wrong value indistinguishable from a real "healed an undamaged
     * creature," worse than a missing record because nothing downstream
     * could tell the two apart.
     */
    private static final Memo<Integer> DAMAGE_BEFORE = (sa, host) -> {
        int total = 0;
        for (Card card : affectedCards(sa)) {
            total += card.getDamage();
        }
        return total;
    };

    /**
     * The six {@code Replace*} APIs are exactly {@code
     * ReplacementEffect.getOverridingAbility()} -- the ability the replacement
     * substitutes in -- and all six report the same thing: the card whose
     * replacement applied, nothing more. Declared ahead of {@link #RULES} for
     * the same reason as {@link #DAMAGE_BEFORE}.
     *
     * <p>This is a strict subset of the {@code rewrite} record this connector
     * already writes for every replacement, from {@code
     * ReplacementObserver.observe} -- see the comment on {@code
     * EventType.REPLACEMENT_APPLIED} in {@code event_schema.py} for why the two
     * must not be read as independent evidence of the same replacement.
     */
    private static final Shaper<Object> REPLACEMENT_APPLIED_EMITTER = (event, sa, host, memo) ->
            event.subject(SnapshotBuilder.entityId(host));

    private static final Map<String, Rule<?>> RULES = rules();

    /** One rule for each API that reports the same event the same way. */
    private static void put(Map<String, Rule<?>> into, Rule<?> rule, String... apis) {
        for (String api : apis) {
            if (into.putIfAbsent(api, rule) != null) {
                throw new IllegalStateException("two rules for " + api);
            }
        }
    }

    private static Map<String, Rule<?>> rules() {
        Map<String, Rule<?>> rules = new java.util.HashMap<>();
        Shaper<List<Player>> turns = (event, sa, host, memo) -> withSubjects(
                event.param("count", amount(sa, host, "NumTurns", "1")), memo);
        put(rules, Rule.shaped(EventType.TURN_ADDED, (sa, host) -> affectedPlayers(sa), turns),
                "AddTurn");
        put(rules, Rule.shaped(EventType.TURN_SKIPPED, (sa, host) -> affectedPlayers(sa), turns),
                "SkipTurn");
        put(rules, Rule.shaped(EventType.PHASE_ADDED, null, (event, sa, host, memo) -> event
                        .subject(playerOf(sa))
                        .param("phase", sa.getParamOrDefault("ExtraPhase", "?"))
                        .param("count", amount(sa, host, "NumPhases", "1"))),
                "AddPhase");
        put(rules, Rule.shaped(EventType.PHASE_SKIPPED, (sa, host) -> affectedPlayers(sa),
                        (event, sa, host, memo) -> withSubjects(
                                event.param("phase", sa.getParamOrDefault("Phase", "?")), memo)),
                "SkipPhase");
        put(rules, Rule.shaped(EventType.X_CHANGED, null, (event, sa, host, memo) ->
                        event.param("value", amount(sa, host, "Value", "0"))),
                "ChangeX");
        put(rules, Rule.shaped(EventType.OWNERSHIP_CHANGE, null, (event, sa, host, memo) ->
                        event.param("owner", ownerOf(sa))),
                "GainOwnership");

        // ── choices, healed damage, granted abilities, retargets ─────────
        //
        // Two keys below are NOT spelled like their effect classes (ApiType.java):
        //   NameCard      declares ChooseCardNameEffect.class
        //   GenericChoice declares ChooseGenericEffect.class
        // A rule keyed by the CLASS stem matches nothing and fails silently.
        //
        // GenericChoice, ChooseDirection, ChooseEvenOdd and NameCard are not
        // optional extras needing a judgment call: all four are present in
        // event_schema.py's EFFECT_API_EVENTS (mapped to CHOICE_MADE) and absent
        // from EXCLUDED_EFFECT_APIS, so the schema itself declares that these APIs
        // produce choice_made. A rule that matches here and reads nothing back
        // would be worse than one left out of this table entirely -- it looks
        // wired. Their choice() read branches, below, are required, not optional.
        put(rules, new Rule<>(EventType.CHOICE_MADE, null, (sa, host, memo) -> choice(host)),
                "ChooseColor", "ChooseType", "ChooseCard", "NameCard", "ChooseNumber",
                "ChoosePlayer", "ChooseDirection", "ChooseEvenOdd", "ChooseSource",
                "GenericChoice");

        // ChooseSector is deliberately NOT wired here, though ApiType.java has
        // it and Card.getChosenSector() is a real accessor: event_schema.py's
        // EXCLUDED_EFFECT_APIS carries
        //   "ChooseSector": "Unfinity attraction; not reachable in sealed or draft"
        // with ChooseSector correspondingly absent from EFFECT_API_EVENTS. This
        // corpus is collected from sealed and draft pools only, which never
        // include Unfinity's silver-bordered attraction cards, so the clause this
        // would read can never actually run here. Wiring it needs the schema
        // changed first -- everyChoiceMadeApiHasAReadableChoice in ApiEventsTest
        // fails the moment the two disagree, which is the point.
        put(rules, Rule.shaped(EventType.DAMAGE_HEALED, DAMAGE_BEFORE, (event, sa, host, memo) ->
                        event.param("amount", memo == null ? 0 : memo)),
                "HealDamage");
        put(rules, Rule.shaped(EventType.ABILITY_CHANGE, null, (event, sa, host, memo) -> event
                        .param("abilities", sa.getParamOrDefault("Abilities", ""))
                        .param("removed", sa.hasParam("RemoveAllAbilities"))),
                "Animate", "AnimateAll");
        put(rules, Rule.shaped(EventType.CONTINUOUS_EFFECT_CREATED, null,
                        (event, sa, host, memo) ->
                                event.param("layers", sa.getParamOrDefault("StaticAbilities", ""))),
                "Effect");
        put(rules, new Rule<>(EventType.TARGETS_CHANGED, null, (sa, host, memo) -> retargeted(sa)),
                "ChangeTargets");

        // ── schedules, replacements, titles and combat removals ──────────

        // delayed_trigger_created: the card that scheduled it. Always the
        // host -- neither effect reads Defined$/ValidTgts$ for the trigger
        // itself, only for what the delayed trigger later acts on.
        put(rules, Rule.shaped(EventType.DELAYED_TRIGGER_CREATED, null, (event, sa, host, memo) ->
                        event.subject(SnapshotBuilder.entityId(host))),
                "DelayedTrigger");
        // ImmediateTriggerEffect.resolve() returns before registering
        // anything when TriggerAmount$ (default 1) calculates to <= 0 --
        // CR 603.12a's "once for each of those times" can be zero times.
        // Captured as a memo (read the same way ApiEvents' own #amount
        // helper reads it elsewhere) rather than re-read in the emitter,
        // matching this file's own convention: nothing in the effect
        // mutates whatever TriggerAmount$'s expression depends on before
        // this is the first thing resolve() computes.
        put(rules, Rule.shaped(EventType.DELAYED_TRIGGER_CREATED,
                        (sa, host) -> amount(sa, host, "TriggerAmount", "1"),
                        (event, sa, host, memo) -> memo != null && memo <= 0 ? null
                                : event.subject(SnapshotBuilder.entityId(host))),
                "ImmediateTrigger");

        // replacement_applied: the card whose replacement applied. See
        // REPLACEMENT_APPLIED_EMITTER's own javadoc above for why all six
        // share one emitter and why this is a strict subset of `rewrite`.
        put(rules, Rule.shaped(EventType.REPLACEMENT_APPLIED, null, REPLACEMENT_APPLIED_EMITTER),
                "ReplaceEffect", "ReplaceCounter", "ReplaceDamage", "ReplaceMana",
                "ReplaceSplitDamage", "ReplaceToken");

        // monarch_changed: the new monarch, and only when the monarch
        // genuinely changed. GameAction.becomeMonarch is a no-op (no
        // state change, no trigger) when the named player already is the
        // monarch or canBecomeMonarch() says no (a static ability, e.g.
        // Crown of Vigor's granted "monarch can't lose the crown" combined
        // with a would-be usurper who cannot become monarch some other
        // way) -- replicating that condition here would duplicate the
        // engine's own classification, a second copy that drifts. Reading
        // the actual before/after monarch
        // instead needs no such duplication and is correct regardless of
        // how many players BecomeMonarchEffect's loop iterates.
        put(rules, Rule.shaped(EventType.MONARCH_CHANGED,
                        (sa, host) -> host.getGame().getMonarch(),
                        (event, sa, host, memo) -> {
                            Player after = host.getGame().getMonarch();
                            if (after == null || after.equals(memo)) {
                                return null;
                            }
                            return event.subject(SnapshotBuilder.playerId(after));
                        }),
                "BecomeMonarch");

        // ring_tempts: the tempted player, always the activator --
        // RingTemptsYouEffect.resolve() reads sa.getActivatingPlayer()
        // unconditionally, never Defined$/targeting. The Ring-bearer is a
        // separable second fact from the same clause (Card.setRingBearer,
        // read back post-resolution since it is only known once the
        // player's controller has chosen one) and is named as a second
        // subject when one was actually chosen -- null when the player
        // controls no creatures, per chooseSingleEntityForEffect's own
        // contract on an empty candidate list.
        put(rules, Rule.shaped(EventType.RING_TEMPTS, null, (event, sa, host, memo) -> {
                    Player p = sa.getActivatingPlayer();
                    if (p == null) {
                        return null;
                    }
                    event.subject(SnapshotBuilder.playerId(p));
                    Card ringBearer = p.getRingBearer();
                    if (ringBearer != null) {
                        event.subject(SnapshotBuilder.entityId(ringBearer));
                    }
                    return event;
                }),
                "RingTemptsYou");

        // removed_from_combat: every creature actually removed, not just
        // the first. RemoveFromCombatEffect gates each target on several
        // conditions (combat null, not in play, a stale LKI reference)
        // this file cannot see from outside; reading whether each
        // memoized combatant is still attacking or blocking afterward
        // sidesteps replicating them, the same before/after-diff shape as
        // BecomeMonarch above.
        put(rules, new Rule<>(EventType.REMOVED_FROM_COMBAT,
                        (sa, host) -> combatantsBefore(sa, host),
                        (sa, host, memo) -> noLongerInCombat(host, memo)),
                "RemoveFromCombat");
        // ChangeCombatants reselects an attacker's defender by calling
        // addToCombat, whose only removeFromCombat call sits immediately
        // before re-adding the same creature against the new defender
        // (SpellAbilityEffect.java:762) -- so "removed from combat" here
        // means a targeted attacker's defender assignment actually
        // changed, read the same way ChangeCombatantsEffect itself reads
        // its own originalDefender/defender pair for retargeting
        // triggers, not replicated from the Optional-decline or
        // already-attacking-that-defender guards.
        put(rules, new Rule<>(EventType.REMOVED_FROM_COMBAT,
                        (sa, host) -> attackerDefendersBefore(sa, host),
                        (sa, host, memo) -> reselectedAttackers(host, memo)),
                "ChangeCombatants");
        // combat_ended: the attacking player whose combat phase the clause
        // ended. EndCombatPhaseEffect does nothing outside a combat phase
        // (CR 723.2g), so the memo is that combat's attacker, read before the
        // clause clears it, and no combat emits nothing.
        put(rules, Rule.shaped(EventType.COMBAT_ENDED,
                        (sa, host) -> {
                            Combat combat = host.getGame().getCombat();
                            return combat == null ? null : combat.getAttackingPlayer();
                        },
                        (event, sa, host, memo) -> memo == null
                                ? null
                                : event.subject(SnapshotBuilder.playerId(memo))),
                "EndCombatPhase");

        // initiative_taken: the player taking the initiative.
        // GameAction.takeInitiative fires its trigger unconditionally --
        // "You can take the initiative even if you already have it" is in
        // the engine's own comment -- unlike becomeMonarch, which is a
        // no-op when the target already holds the title. So this is NOT
        // a before/after diff (that would wrongly suppress the
        // already-has-it re-affirmation): it reads the post-resolution
        // holder and confirms they are one of this clause's own targets,
        // which is true whether the holder changed or was reconfirmed.
        //
        // Known limit (record-schema.md "initiative_taken can miss a
        // redirect"): a target who satisfies
        // isInGame() but has also already hasLost() triggers
        // GameAction.takeInitiative's own redirect to the next player,
        // which that method's own unconditional fall-through then
        // overwrites back to the original (lost) target before this rule
        // ever reads game.getHasInitiative() -- so the event still fires,
        // but can name the lost player instead of who the redirect
        // actually landed on. Not replicated here: doing so would mean
        // re-deriving GameAction's own fall-through as a second copy of it,
        // which drifts the way any duplicated classification does.
        put(rules, Rule.shaped(EventType.INITIATIVE_TAKEN, null, (event, sa, host, memo) -> {
                    Player holder = host.getGame().getHasInitiative();
                    if (holder == null || !affectedPlayers(sa).contains(holder)) {
                        return null;
                    }
                    return event.subject(SnapshotBuilder.playerId(holder));
                }),
                "TakeInitiative");

        // text_change: the card(s) whose text changed. Only when the
        // clause actually names a word to replace -- sa.hasParam reads
        // its own params directly, no post-resolution state needed --
        // since ChangeTextEffect's target loop still fires
        // GameEventCardStatsChanged even when neither ChangeColorWord$
        // nor ChangeTypeWord$ is set. ExchangeTextBox's own contribution
        // to text_change is a note() call from TextBoxExchangeEffect
        // itself (see that class), not a rule here: its two early returns
        // (checkValidDuration, tgtCards.size() < 2) are not cheaply
        // replicable from outside without duplicating engine logic.
        put(rules, new Rule<>(EventType.TEXT_CHANGE, null, (sa, host, memo) -> changeTextEvent(sa)),
                "ChangeText");
        return Map.copyOf(rules);
    }

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
     * balanced. The alternative -- skip this push, and have the matching pop
     * skip too -- was rejected: nothing about the failure is visible at
     * {@code onClauseResolved} time except by consulting a second piece of
     * state kept in lockstep with this one, and that second state would
     * itself have to be pushed unconditionally to stay balanced, which is
     * this design again with extra steps.
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

    /** Set the first time any rule's emitter throws, so the line below prints once. */
    private static final AtomicBoolean EMITTER_FAILURE_REPORTED = new AtomicBoolean(false);

    /**
     * Mirror the treatment {@code AbilityUtils.reportClauseListenerFailure}
     * gives this same hook's other end, in the forge repo: latched to the
     * first throw, naming what failed and why, rather than repeated once per
     * clause -- a real playthrough resolves thousands of them, and a broken
     * rule would flood the log for the whole run instead of announcing itself
     * once.
     *
     * <p>Package-private rather than private, the same reason {@code
     * clauseHandler} is: forcing a rule to throw reliably would mean
     * contriving a malformed script value or a null host card, and this is
     * what {@code AbilityUtils} does instead for the hook's other end --
     * {@code EffectRecordClauseHookTest} calls {@code
     * reportClauseListenerFailure} through a seam, not through a real throw.
     */
    static void reportEmitterFailure(EventType eventType, RuntimeException e) {
        if (EMITTER_FAILURE_REPORTED.compareAndSet(false, true)) {
            System.err.println("ApiEvents: the emitter for \"" + eventType.wireName()
                    + "\" threw and was ignored; further emitter failures will "
                    + "not be logged: " + e);
        }
    }

    /** Set the first time any rule's memo throws, so the line below prints once. */
    private static final AtomicBoolean MEMO_FAILURE_REPORTED = new AtomicBoolean(false);

    /**
     * As {@link #reportEmitterFailure}, for a rule's memo instead of its
     * emitter.
     *
     * <p>Its own message, because a memo ({@link #observe}) and an emitter
     * ({@link ClauseObservation#finish}) are different halves of one {@link Rule}, and a memo
     * failure reported as "the emitter" sends its reader to the wrong one.
     * Its own latch, because a memo failure latching
     * {@link #EMITTER_FAILURE_REPORTED} would silence a later, unrelated
     * emitter failure for the rest of the JVM.
     */
    static void reportMemoFailure(EventType eventType, RuntimeException e) {
        if (MEMO_FAILURE_REPORTED.compareAndSet(false, true)) {
            System.err.println("ApiEvents: the memo for \"" + eventType.wireName()
                    + "\" threw and was ignored; further memo failures will "
                    + "not be logged: " + e);
        }
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

    private static int amount(SpellAbility sa, Card host, String key, String fallback) {
        return AbilityUtils.calculateAmount(host, sa.getParamOrDefault(key, fallback), sa);
    }

    private static String playerOf(SpellAbility sa) {
        return sa.getActivatingPlayer() == null
                ? null : SnapshotBuilder.playerId(sa.getActivatingPlayer());
    }

    /**
     * Every player {@code AddTurn}, {@code SkipTurn} and {@code SkipPhase}
     * actually affect -- the ones their own {@code resolve()} iterates, not
     * whoever activated them.
     *
     * <p>All three iterate {@code SpellAbilityEffect.getPlayers(false,
     * "Defined", sa)}, which is private to its package and so mirrored here:
     * <b>targeting wins over {@code Defined$}</b> whenever the ability
     * targets at all. The activator is not a safe default: most
     * {@code SkipPhase} scripts, and {@code empty_city_ruse.txt}'s
     * {@code Opponent} target, name someone else.
     *
     * <p>Captured as a memo, before the clause resolves: none of the three
     * changes {@code sa}'s targets, so either side reads the same, and the
     * memo is what pins the clause hook's push/pop pairing against a
     * non-null value.
     */
    private static List<Player> affectedPlayers(SpellAbility sa) {
        if (sa.usesTargeting()) {
            List<Player> targeted = new ArrayList<>();
            sa.getTargets().getTargetPlayers().forEach(targeted::add);
            return targeted;
        }
        return definedPlayers(sa, sa.getParamOrDefault("Defined", "You"));
    }

    /**
     * One subject per affected player, in the order the memo carries them.
     *
     * <p>{@code getTargetPlayers} returns every target, and {@code AddTurn}
     * naming more than one is a real script shape (a duel-taker rewarding
     * both duelists, say) -- collapsing to the first would silently drop
     * every player after it, the same failure shape as reading the wrong
     * player entirely.
     */
    private static EffectEvent withSubjects(EffectEvent event, List<Player> players) {
        if (players != null) {
            for (Player player : players) {
                event.subject(SnapshotBuilder.playerId(player));
            }
        }
        return event;
    }

    /**
     * The new owner {@code GainOwnership} promises: a named player where the
     * clause names one, the activating player otherwise.
     *
     * <p>{@code OwnershipGainEffect.resolve} reads {@code DefinedPlayer} first
     * and falls back to {@code sa.getActivatingPlayer()} only when that
     * resolves to nobody. A purely parameter-based rule that always read the
     * activating player would name the wrong owner on a real card: Tempest
     * Efreet's second clause is {@code DB$ GainOwnership | Defined$
     * CorrectedSelf | DefinedPlayer$ Targeted}, which hands the card to the
     * opponent it targeted, not to whoever sacrificed the Efreet; Bronze
     * Tablet's is {@code DefinedPlayer$ Remembered}, naming a player chosen
     * three clauses earlier. Both are real printed cards, not hypotheticals.
     */
    private static String ownerOf(SpellAbility sa) {
        if (sa.hasParam("DefinedPlayer")) {
            List<Player> named = definedPlayers(sa, sa.getParam("DefinedPlayer"));
            if (!named.isEmpty()) {
                return SnapshotBuilder.playerId(named.get(0));
            }
        }
        return playerOf(sa);
    }

    /**
     * {@code AbilityUtils.getDefinedPlayers} for one {@code Defined$}/
     * {@code DefinedPlayer$} value, extended to the {@code " & "}-joined
     * multi-value form {@code SpellAbilityEffect.getPlayers} also accepts
     * ({@code SpellAbilityEffect.java:314}): {@code String[] def =
     * sa.getParamOrDefault(definedParam, "You").split(" & ")}, one lookup per
     * token, unioned. No card in the cardsfolder uses this today for either
     * {@code Defined$} on these three rules or {@code DefinedPlayer$} on
     * {@code GainOwnership}, but an approximation this cheap to close is not
     * worth keeping.
     */
    private static List<Player> definedPlayers(SpellAbility sa, String def) {
        List<Player> players = new ArrayList<>();
        for (String token : def.split(" & ")) {
            players.addAll(AbilityUtils.getDefinedPlayers(sa.getHostCard(), token, sa));
        }
        return players;
    }

    /**
     * The cards a clause with no dedicated targeting concept of its own
     * acts on -- {@link #affectedPlayers}'s gate, mirrored for cards rather
     * than players, because {@code HealDamage} is exactly that shape:
     * {@code SpellAbilityEffect.getCards} (which {@code
     * HealDamageEffect.resolve()} calls through {@code getTargetCards(sa)})
     * reads {@code sa.getTargets()} only when {@code sa.usesTargeting()}
     * is true, and otherwise falls back to a {@code " & "}-split {@code
     * Defined$}, defaulted to {@code "Self"} ({@code
     * SpellAbilityEffect.java:257,265}) -- the same default used here, not
     * a different one invented for this file.
     */
    private static List<Card> affectedCards(SpellAbility sa) {
        if (sa.usesTargeting()) {
            List<Card> targeted = new ArrayList<>();
            sa.getTargets().getTargetCards().forEach(targeted::add);
            return targeted;
        }
        return definedCards(sa, sa.getParamOrDefault("Defined", "Self"));
    }

    /** {@link #definedPlayers}, mirrored for cards via {@code AbilityUtils.getDefinedCards}. */
    private static List<Card> definedCards(SpellAbility sa, String def) {
        List<Card> cards = new ArrayList<>();
        for (String token : def.split(" & ")) {
            cards.addAll(AbilityUtils.getDefinedCards(sa.getHostCard(), token, sa));
        }
        return cards;
    }

    /**
     * The choice a Choose* clause left on its host, as a kind and a value.
     *
     * <p>Reads whichever dedicated accessor the clause populated, in a fixed
     * priority order -- the API that actually ran is not consulted, only what
     * state it left behind, so a choice left on the host by an earlier clause
     * in the same chain can in principle outrank a later clause that
     * legitimately produced no reading of its own. This mirrors the
     * approximation {@link #ownerOf} already accepts for {@code
     * GainOwnership}'s single-scalar shape rather than introducing a new one.
     *
     * <p><b>{@code getChosenColor()} and {@code getNamedCard()} never return
     * {@code null}</b> -- an unset colour reads back as {@code ""} ({@code
     * Card.getChosenColor}), the same way an unset named card reads back as
     * {@code ""} rather than {@code null}. A bare {@code != null} check
     * against either would always be true, which for the colour branch --
     * checked first -- would report <i>every</i> clause routed through this
     * method as a colour choice, direction and number and all. {@code
     * hasChosenColor()} is what {@code Card} itself defines for this
     * distinction, so this reads that instead; the named-card branch is
     * guarded by {@link #nonEmpty}.
     *
     * <p>The direction, even/odd and mode readings are not optional extras,
     * though few cards reach them: {@code event_schema.py}'s {@code
     * EFFECT_API_EVENTS} maps all three (and {@code NameCard}) to {@code
     * CHOICE_MADE} and excludes none of them, so the schema itself declares
     * that these APIs produce {@code choice_made} -- a rule that matches and
     * then reads nothing back would be worse than one left out of {@code
     * RULES} entirely, because it looks wired. {@code
     * ChooseDirectionEffect.resolve()} always calls {@code
     * setChosenDirection}, and {@code ChooseEvenOddEffect.resolve()} always
     * calls {@code setChosenEvenOdd} -- the identical "dedicated accessor
     * that stays there" shape as the six branches already read. {@code
     * GenericChoice} only reaches its branch for the {@code SetChosenMode$
     * True} scripts (Tarkir's Sieges, Fallout's Hoover Dam, 15 real cards);
     * the rest resolve a chosen sub-ability directly and leave no scalar on
     * the host to describe which one -- the schema still declares the API,
     * so the branch is what makes the reachable subset of it reachable.
     *
     * <p>There is no {@code getChosenSector()} reading, for the reason there
     * is no {@code ChooseSector} rule (see the comment above {@code
     * HealDamage}): nothing routes that API here, and the reading would be a
     * stale-state misattribution risk (a host carrying a leftover {@code
     * chosenSector} from something else entirely would outrank a later, real
     * choice on the same card).
     */
    private static EffectEvent choice(Card host) {
        if (host == null) {
            return null;
        }
        for (ChoiceReading reading : CHOICES) {
            String value = reading.read().apply(host);
            if (value != null) {
                return new EffectEvent(EventType.CHOICE_MADE)
                        .param("choice_kind", reading.kind())
                        .param("value", value);
            }
        }
        // The clause chose something this has no reading for. An event naming
        // the choice kind "?" would train the head on a distinction the corpus
        // cannot make.
        return null;
    }

    /**
     * One accessor a choice can be left in, and its reading: the value, or
     * null where the accessor holds nothing.
     */
    private record ChoiceReading(String kind, Function<Card, String> read) {
    }

    /** The accessors {@link #choice} reads, in its priority order. */
    private static final List<ChoiceReading> CHOICES = List.of(
            new ChoiceReading("color",
                    host -> host.hasChosenColor() ? host.getChosenColor() : null),
            new ChoiceReading("type", host -> nonEmpty(host.getChosenType())),
            new ChoiceReading("number", host -> host.getChosenNumber() == null
                    ? null : String.valueOf(host.getChosenNumber())),
            new ChoiceReading("card_name", host -> nonEmpty(host.getNamedCard())),
            new ChoiceReading("player", host -> host.getChosenPlayer() == null
                    ? null : SnapshotBuilder.playerId(host.getChosenPlayer())),
            new ChoiceReading("cards",
                    host -> host.getChosenCards() == null || host.getChosenCards().isEmpty()
                            ? null : String.valueOf(host.getChosenCards().size())),
            new ChoiceReading("direction", host -> host.getChosenDirection() == null
                    ? null : host.getChosenDirection().name()),
            new ChoiceReading("even_odd", host -> host.getChosenEvenOdd() == null
                    ? null : host.getChosenEvenOdd().name()),
            new ChoiceReading("mode", host -> nonEmpty(host.getChosenMode())));

    private static String nonEmpty(String value) {
        return value == null || value.isEmpty() ? null : value;
    }

    /**
     * What the retargeted spells point at now.
     *
     * <p>Read after the clause rather than reported by it, because a spell it
     * retargeted is still on the stack carrying its new {@code TargetChoices}
     * -- this is the one outcome in this group that survives on an object
     * other than the host.
     *
     * <p>Reads {@code getTargetEntities()} rather than only {@code
     * getTargetCards()}. Redirecting a burn spell at a <em>player</em> is the classic use of
     * this mechanic -- Deflection, Misdirection and Bolt Bend are all real
     * and all reach {@code ChangeTargetsEffect}'s default "choose any new
     * legal target" branch, which retargets at a player as readily as a
     * card. {@code getTargetCards()} filters to {@code Card} instances
     * only, so a {@code Player} target would be silently dropped and the
     * event would still fire with an empty {@code targets} list -- a channel
     * that looks wired while reporting nothing distinguishable from
     * "retargeted at nothing." Mixed-entity targets are read as {@code
     * SnapshotBuilder.stackExtrasJson} reads them, through {@link Refs#ofAny}.
     */
    private static EffectEvent retargeted(SpellAbility sa) {
        List<String> targets = new ArrayList<>();
        if (sa.getTargets() != null) {
            for (SpellAbility changed : sa.getTargets().getTargetSpells()) {
                if (changed.getTargets() == null) {
                    continue;
                }
                for (GameEntity target : changed.getTargets().getTargetEntities()) {
                    Refs.ofAny(target).ifPresent(targets::add);
                }
            }
        }
        return new EffectEvent(EventType.TARGETS_CHANGED).param("targets", targets);
    }

    // ── text changes and combat removals ────────────────────────────────

    /**
     * {@code getCardsfromTargets(sa)} (SpellAbilityEffect.java:370), mirrored
     * the way {@link #affectedCards} already mirrors {@code getTargetCards}.
     *
     * <p>The {@code getTargetSpells()} union below is NOT dead for today's
     * cardsfolder. Seven of {@code ChangeText}'s seventeen
     * occurrences are {@code ValidTgts$ Card | TgtZone$ Stack,Battlefield} --
     * {@code artificial_evolution}, {@code sleight_of_mind} and {@code
     * magical_hack} among them -- which can target a spell on the stack, and
     * {@code TargetChoices} stores a spell target as the {@code SpellAbility}
     * itself ({@code getTargetCards()} filters to {@code Card.class}
     * instances only, per {@code TargetChoices.java}; a targeted spell is a
     * {@code SpellAbility}, not one). For that shape this union is not an
     * extra path -- it is the <em>only</em> one that produces a subject at
     * all; {@link #affectedCards}'s targeting branch reads {@code
     * getTargetCards()} alone and would see nothing.
     */
    private static List<Card> cardsFromTargets(SpellAbility sa) {
        List<Card> cards = new ArrayList<>(affectedCards(sa));
        if (sa.getTargets() != null) {
            for (SpellAbility spellTarget : sa.getTargets().getTargetSpells()) {
                cards.add(spellTarget.getHostCard());
            }
        }
        return cards;
    }

    /**
     * {@code ChangeText}'s subject list, or {@code null} when the clause
     * names no word to replace at all -- {@code ChangeTextEffect.resolve()}'s
     * target loop still fires {@code GameEventCardStatsChanged} for every
     * target even then, so the params themselves are the only signal that
     * anything textual actually changed.
     */
    private static EffectEvent changeTextEvent(SpellAbility sa) {
        if (!sa.hasParam("ChangeColorWord") && !sa.hasParam("ChangeTypeWord")) {
            return null;
        }
        EffectEvent event = new EffectEvent(EventType.TEXT_CHANGE);
        for (Card c : cardsFromTargets(sa)) {
            event.subject(SnapshotBuilder.entityId(c));
        }
        return event.subjects().isEmpty() ? null : event;
    }

    /**
     * {@code RemoveFromCombat}'s memo: which of the clause's own targets are
     * genuinely attacking or blocking right now, before the clause runs.
     * Mirrors {@link #DAMAGE_BEFORE}'s before/after shape rather than
     * replicating {@code RemoveFromCombatEffect}'s own combat-null/in-play/
     * stale-LKI guards.
     */
    private static List<Card> combatantsBefore(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        List<Card> inCombat = new ArrayList<>();
        if (combat != null) {
            for (Card c : affectedCards(sa)) {
                if (combat.isAttacking(c) || combat.isBlocking(c)) {
                    inCombat.add(c);
                }
            }
        }
        return inCombat;
    }

    /** Every memoized combatant no longer attacking or blocking, or {@code null} if none. */
    private static EffectEvent noLongerInCombat(Card host, List<Card> before) {
        if (before == null || before.isEmpty()) {
            return null;
        }
        Combat combat = host.getGame().getCombat();
        EffectEvent event = new EffectEvent(EventType.REMOVED_FROM_COMBAT);
        for (Card c : before) {
            if (combat == null || !(combat.isAttacking(c) || combat.isBlocking(c))) {
                event.subject(SnapshotBuilder.entityId(c));
            }
        }
        return event.subjects().isEmpty() ? null : event;
    }

    /**
     * {@code ChangeCombatants}'s memo: each targeted card's current attacker
     * assignment, the same pair ({@code originalDefender}/{@code defender})
     * {@code ChangeCombatantsEffect.resolve()} itself reads around its own
     * {@code addToCombat} call, for the same reselection-retargeting reason.
     * A card absent from combat entirely (not currently attacking) maps to
     * {@code null} rather than being omitted, so {@link #reselectedAttackers}
     * can tell "wasn't attacking" from "was attacking, still is" without a
     * second lookup.
     */
    private static Map<Card, GameEntity> attackerDefendersBefore(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        Map<Card, GameEntity> map = new LinkedHashMap<>();
        if (combat != null) {
            for (Card c : affectedCards(sa)) {
                map.put(c, combat.getDefenderByAttacker(c));
            }
        }
        return map;
    }

    /**
     * Every memoized attacker whose defender assignment actually changed --
     * exactly the creatures {@code SpellAbilityEffect.addToCombat}'s
     * {@code combat.removeFromCombat(c)} call (SpellAbilityEffect.java:762)
     * ran for, since that call fires only immediately before re-adding the
     * same creature against a new defender. A card that was never attacking
     * ({@code null} before) is skipped: nothing could have removed it.
     */
    private static EffectEvent reselectedAttackers(Card host, Map<Card, GameEntity> before) {
        if (before == null || before.isEmpty()) {
            return null;
        }
        Combat combat = host.getGame().getCombat();
        EffectEvent event = new EffectEvent(EventType.REMOVED_FROM_COMBAT);
        for (Map.Entry<Card, GameEntity> entry : before.entrySet()) {
            Card c = entry.getKey();
            if (entry.getValue() == null) {
                continue;
            }
            GameEntity now = combat == null ? null : combat.getDefenderByAttacker(c);
            if (!entry.getValue().equals(now)) {
                event.subject(SnapshotBuilder.entityId(c));
            }
        }
        return event.subjects().isEmpty() ? null : event;
    }
}
