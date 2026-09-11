package com.pricepredictor.connector.effects;

import forge.game.GameEntity;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The event an effect API promises, built from the clause that ran.
 *
 * <p>Most of Forge's effects broadcast nothing. Damage and zone changes reach a
 * collector because the engine fires a bus event for them; an extra turn, a
 * prevented damage, a chosen colour reach nobody, and the corpus recorded those
 * abilities as abilities that did nothing. What every one of them does carry is
 * its own {@code SpellAbility}: the parameters it was scripted with, and for
 * some, a result remembered on the host.
 *
 * <p>Keyed by API name rather than by {@code ApiType} so a Forge rename costs a
 * missing event rather than a compile error, and so this file needs no import
 * of an enum whose members move. The names are pinned by
 * {@code test_every_effect_api_maps_to_an_event} on the Python side.
 *
 * <p>An API absent from this table emits nothing. That is the right default:
 * damage, counters, taps and zone changes already arrive on the bus, and a
 * second event for them would double-count an outcome.
 */
final class ApiEvents {

    private ApiEvents() {
    }

    /** What an emitter needs from before the clause ran, or null when nothing. */
    @FunctionalInterface
    interface Memo {
        Object take(SpellAbility sa, Card host);
    }

    /** The event, given the clause and whatever its memo captured. */
    @FunctionalInterface
    interface Emitter {
        EffectEvent emit(SpellAbility sa, Card host, Object memo);
    }

    /**
     * @param eventType named in the once-only failure line an emitter's throw prints
     */
    record Rule(String eventType, Memo memo, Emitter emitter) {
    }

    /**
     * Damage on the clause's affected cards, before it heals any of it away.
     *
     * <p>Declared ahead of {@link #RULES} on purpose: a static field read by
     * simple name from another field's own initializer has to be declared
     * first or the reference is an illegal forward reference and the class
     * does not compile at all -- unlike the emitter lambdas below (and
     * {@link #affectedCards}, which this memo's lambda body calls), which
     * call methods declared later in this file with no such restriction.
     *
     * <p>Fixed in fix round 2 (task-8-fix-2.md, Finding 1). The original
     * version read {@code sa.getTargets().getTargetCards()} directly --
     * the clause's own {@code TargetChoices}. But {@code
     * HealDamageEffect.resolve()} finds its cards through {@code
     * SpellAbilityEffect.getTargetCards(sa)}, which consults {@code
     * sa.getTargets()} only when {@code sa.usesTargeting()} is true, and
     * neither real {@code HealDamage} card in the cardsfolder declares any
     * targeting on the clause itself -- both use {@code Defined$
     * Replaced*}, resolved through a replacement effect's {@code
     * getReplacingObject}, which never touches {@code sa.getTargets()}. So
     * the direct read found the clause's own (always-empty, per {@code
     * SpellAbility.targetChosen}'s eager initialisation) {@code
     * TargetChoices} on every real firing and reported {@code amount: 0} --
     * a wrong value indistinguishable from a real "healed an undamaged
     * creature," worse than a missing record because nothing downstream
     * could tell the two apart.
     */
    private static final Memo DAMAGE_BEFORE = (sa, host) -> {
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
     * the same forward-reference reason as {@link #DAMAGE_BEFORE} (a field
     * initializer cannot read a sibling field declared later by simple name).
     *
     * <p>This is a strict subset of the {@code rewrite} record this connector
     * already writes for every replacement, from {@code
     * ReplacementHandler.observeReplacement} -- see the comment on {@code
     * EventType.REPLACEMENT_APPLIED} in {@code event_schema.py} for why the two
     * must not be read as independent evidence of the same replacement.
     */
    private static final Emitter REPLACEMENT_APPLIED_EMITTER = (sa, host, memo) ->
            new EffectEvent(EffectEvent.REPLACEMENT_APPLIED).subject(SnapshotBuilder.entityId(host));

    private static final Map<String, Rule> RULES = Map.ofEntries(
            Map.entry("AddTurn", new Rule(EffectEvent.TURN_ADDED,
                    (sa, host) -> affectedPlayers(sa),
                    (sa, host, memo) -> withSubjects(
                            new EffectEvent(EffectEvent.TURN_ADDED)
                                    .param("count", amount(sa, host, "NumTurns", "1")),
                            memo))),
            Map.entry("SkipTurn", new Rule(EffectEvent.TURN_SKIPPED,
                    (sa, host) -> affectedPlayers(sa),
                    (sa, host, memo) -> withSubjects(
                            new EffectEvent(EffectEvent.TURN_SKIPPED)
                                    .param("count", amount(sa, host, "NumTurns", "1")),
                            memo))),
            Map.entry("AddPhase", new Rule(EffectEvent.PHASE_ADDED, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.PHASE_ADDED)
                            .subject(playerOf(sa))
                            .param("phase", sa.getParamOrDefault("ExtraPhase", "?"))
                            .param("count", amount(sa, host, "NumPhases", "1")))),
            Map.entry("SkipPhase", new Rule(EffectEvent.PHASE_SKIPPED,
                    (sa, host) -> affectedPlayers(sa),
                    (sa, host, memo) -> withSubjects(
                            new EffectEvent(EffectEvent.PHASE_SKIPPED)
                                    .param("phase", sa.getParamOrDefault("Phase", "?")),
                            memo))),
            Map.entry("ChangeX", new Rule(EffectEvent.X_CHANGED, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.X_CHANGED)
                            .param("value", amount(sa, host, "Value", "0")))),
            Map.entry("GainOwnership", new Rule(EffectEvent.OWNERSHIP_CHANGE, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.OWNERSHIP_CHANGE)
                            .param("owner", ownerOf(sa)))),

            // ── Task 8: choices, healed damage, granted abilities, retargets ──
            //
            // Ruling R18 (task-8-brief.md) corrected two keys below that were NOT
            // spelled like their effect classes. Verified against ApiType.java
            // before use:
            //   NameCard      (ApiType.java:135) declares ChooseCardNameEffect.class
            //   GenericChoice (ApiType.java:109) declares ChooseGenericEffect.class
            // A rule keyed by the CLASS stem matches nothing and fails silently.
            //
            // R18 also added a ChooseSector entry here, on the reasoning that it fits
            // choice_made and that leaving it out would be a voluntary omission. That
            // check covered ApiType.java and the Card accessor but never the Python
            // schema, which already had the answer -- see the comment below
            // GenericChoice, where that entry would otherwise sit. Fix round 1 removed
            // it.
            //
            // The brief's sample code for every entry below constructed `Rule` with
            // only two arguments (a memo-or-null, then the emitter), omitting the
            // eventType this record actually declares first -- that does not compile
            // against the real three-argument Rule above, so each entry here supplies
            // the real constant the same way every pre-existing rule already does.
            //
            // GenericChoice, ChooseDirection, ChooseEvenOdd and NameCard are not
            // optional extras needing a judgment call: all four are present in
            // event_schema.py's EFFECT_API_EVENTS (mapped to CHOICE_MADE) and absent
            // from EXCLUDED_EFFECT_APIS, so the schema itself declares that these APIs
            // produce choice_made. A rule that matches here and reads nothing back
            // would be worse than one left out of this table entirely -- it looks
            // wired. Their choice() read branches, below, are required, not optional.
            Map.entry("ChooseColor", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseType", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseCard", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("NameCard", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseNumber", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChoosePlayer", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseDirection", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseEvenOdd", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseSource", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("GenericChoice", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),

            // ChooseSector is deliberately NOT wired here. Ruling R18 added it after
            // checking ApiType.java (a real member, ApiType.java:54) and
            // Card.getChosenSector() (a real accessor -- ChooseSectorEffect.java:13
            // calls setChosenSector) -- but never checked the Python schema, which
            // already had the answer: event_schema.py's EXCLUDED_EFFECT_APIS carries
            //   "ChooseSector": "Unfinity attraction; not reachable in sealed or draft"
            // with ChooseSector correspondingly absent from EFFECT_API_EVENTS. This
            // corpus is collected from sealed and draft pools only, which never
            // include Unfinity's silver-bordered attraction cards, so the clause this
            // would read can never actually run here. If ChooseSector caught your eye
            // in ApiType.java and you are about to re-add it the way R18 did: don't,
            // without re-checking the schema first -- everyChoiceMadeApiHasAReadableChoice
            // in ApiEventsTest will fail the moment you do, which is the point.
            Map.entry("HealDamage", new Rule(EffectEvent.DAMAGE_HEALED, DAMAGE_BEFORE,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.DAMAGE_HEALED)
                            .param("amount", memo instanceof Integer n ? n : 0))),
            Map.entry("Animate", new Rule(EffectEvent.ABILITY_CHANGE, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.ABILITY_CHANGE)
                            .param("abilities", sa.getParamOrDefault("Abilities", ""))
                            .param("removed", sa.hasParam("RemoveAllAbilities")))),
            Map.entry("AnimateAll", new Rule(EffectEvent.ABILITY_CHANGE, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.ABILITY_CHANGE)
                            .param("abilities", sa.getParamOrDefault("Abilities", ""))
                            .param("removed", sa.hasParam("RemoveAllAbilities")))),
            Map.entry("Effect", new Rule(EffectEvent.CONTINUOUS_EFFECT_CREATED, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.CONTINUOUS_EFFECT_CREATED)
                            .param("layers", sa.getParamOrDefault("StaticAbilities", "")))),
            Map.entry("ChangeTargets", new Rule(EffectEvent.TARGETS_CHANGED, null,
                    (sa, host, memo) -> retargeted(sa))),

            // ── Task 12: the eight highest-reach unemitted types ─────────

            // delayed_trigger_created: the card that scheduled it. Always the
            // host -- neither effect reads Defined$/ValidTgts$ for the trigger
            // itself, only for what the delayed trigger later acts on.
            Map.entry("DelayedTrigger", new Rule(EffectEvent.DELAYED_TRIGGER_CREATED, null,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.DELAYED_TRIGGER_CREATED)
                            .subject(SnapshotBuilder.entityId(host)))),
            // ImmediateTriggerEffect.resolve() returns before registering
            // anything when TriggerAmount$ (default 1) calculates to <= 0 --
            // CR 603.12a's "once for each of those times" can be zero times.
            // Captured as a memo (read the same way ApiEvents' own #amount
            // helper reads it elsewhere) rather than re-read in the emitter,
            // matching this file's own convention: nothing in the effect
            // mutates whatever TriggerAmount$'s expression depends on before
            // this is the first thing resolve() computes.
            Map.entry("ImmediateTrigger", new Rule(EffectEvent.DELAYED_TRIGGER_CREATED,
                    (sa, host) -> amount(sa, host, "TriggerAmount", "1"),
                    (sa, host, memo) -> memo instanceof Integer n && n <= 0 ? null
                            : new EffectEvent(EffectEvent.DELAYED_TRIGGER_CREATED)
                                    .subject(SnapshotBuilder.entityId(host)))),

            // replacement_applied: the card whose replacement applied. See
            // REPLACEMENT_APPLIED_EMITTER's own javadoc above for why all six
            // share one emitter and why this is a strict subset of `rewrite`.
            Map.entry("ReplaceEffect", new Rule(EffectEvent.REPLACEMENT_APPLIED, null,
                    REPLACEMENT_APPLIED_EMITTER)),
            Map.entry("ReplaceCounter", new Rule(EffectEvent.REPLACEMENT_APPLIED, null,
                    REPLACEMENT_APPLIED_EMITTER)),
            Map.entry("ReplaceDamage", new Rule(EffectEvent.REPLACEMENT_APPLIED, null,
                    REPLACEMENT_APPLIED_EMITTER)),
            Map.entry("ReplaceMana", new Rule(EffectEvent.REPLACEMENT_APPLIED, null,
                    REPLACEMENT_APPLIED_EMITTER)),
            Map.entry("ReplaceSplitDamage", new Rule(EffectEvent.REPLACEMENT_APPLIED, null,
                    REPLACEMENT_APPLIED_EMITTER)),
            Map.entry("ReplaceToken", new Rule(EffectEvent.REPLACEMENT_APPLIED, null,
                    REPLACEMENT_APPLIED_EMITTER)),

            // monarch_changed: the new monarch, and only when the monarch
            // genuinely changed. GameAction.becomeMonarch is a no-op (no
            // state change, no trigger) when the named player already is the
            // monarch or canBecomeMonarch() says no (a static ability, e.g.
            // Crown of Vigor's granted "monarch can't lose the crown" combined
            // with a would-be usurper who cannot become monarch some other
            // way) -- replicating that condition here would be exactly the
            // duplicated-classification shortcut ruling R12 rejected for
            // damage_prevented. Reading the actual before/after monarch
            // instead needs no such duplication and is correct regardless of
            // how many players BecomeMonarchEffect's loop iterates.
            Map.entry("BecomeMonarch", new Rule(EffectEvent.MONARCH_CHANGED,
                    (sa, host) -> host.getGame().getMonarch(),
                    (sa, host, memo) -> {
                        Player after = host.getGame().getMonarch();
                        if (after == null || after.equals(memo)) {
                            return null;
                        }
                        return new EffectEvent(EffectEvent.MONARCH_CHANGED)
                                .subject(SnapshotBuilder.playerId(after));
                    })),

            // ring_tempts: the tempted player, always the activator --
            // RingTemptsYouEffect.resolve() reads sa.getActivatingPlayer()
            // unconditionally, never Defined$/targeting. The Ring-bearer is a
            // separable second fact from the same clause (Card.setRingBearer,
            // read back post-resolution since it is only known once the
            // player's controller has chosen one) and is named as a second
            // subject when one was actually chosen -- null when the player
            // controls no creatures, per chooseSingleEntityForEffect's own
            // contract on an empty candidate list.
            Map.entry("RingTemptsYou", new Rule(EffectEvent.RING_TEMPTS, null,
                    (sa, host, memo) -> {
                        Player p = sa.getActivatingPlayer();
                        if (p == null) {
                            return null;
                        }
                        EffectEvent event = new EffectEvent(EffectEvent.RING_TEMPTS)
                                .subject(SnapshotBuilder.playerId(p));
                        Card ringBearer = p.getRingBearer();
                        if (ringBearer != null) {
                            event.subject(SnapshotBuilder.entityId(ringBearer));
                        }
                        return event;
                    })),

            // removed_from_combat: every creature actually removed, not just
            // the first (Task 6's shipped defect, repeated here as a named
            // risk). RemoveFromCombatEffect gates each target on several
            // conditions (combat null, not in play, a stale LKI reference)
            // this file cannot see from outside; reading whether each
            // memoized combatant is still attacking or blocking afterward
            // sidesteps replicating them, the same before/after-diff shape as
            // BecomeMonarch above.
            Map.entry("RemoveFromCombat", new Rule(EffectEvent.REMOVED_FROM_COMBAT,
                    (sa, host) -> combatantsBefore(sa, host),
                    (sa, host, memo) -> noLongerInCombat(host, memo))),
            // ChangeCombatants reselects an attacker's defender by calling
            // addToCombat, whose only removeFromCombat call sits immediately
            // before re-adding the same creature against the new defender
            // (SpellAbilityEffect.java:762) -- so "removed from combat" here
            // means a targeted attacker's defender assignment actually
            // changed, read the same way ChangeCombatantsEffect itself reads
            // its own originalDefender/defender pair for retargeting
            // triggers, not replicated from the Optional-decline or
            // already-attacking-that-defender guards.
            Map.entry("ChangeCombatants", new Rule(EffectEvent.REMOVED_FROM_COMBAT,
                    (sa, host) -> attackerDefendersBefore(sa, host),
                    (sa, host, memo) -> reselectedAttackers(host, memo))),

            // initiative_taken: the player taking the initiative.
            // GameAction.takeInitiative fires its trigger unconditionally --
            // "You can take the initiative even if you already have it" is in
            // the engine's own comment -- unlike becomeMonarch, which is a
            // no-op when the target already holds the title. So this is NOT
            // a before/after diff (that would wrongly suppress the
            // already-has-it re-affirmation): it reads the post-resolution
            // holder and confirms they are one of this clause's own targets,
            // which is true whether the holder changed or was reconfirmed.
            Map.entry("TakeInitiative", new Rule(EffectEvent.INITIATIVE_TAKEN, null,
                    (sa, host, memo) -> {
                        Player holder = host.getGame().getHasInitiative();
                        if (holder == null || !affectedPlayers(sa).contains(holder)) {
                            return null;
                        }
                        return new EffectEvent(EffectEvent.INITIATIVE_TAKEN)
                                .subject(SnapshotBuilder.playerId(holder));
                    })),

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
            Map.entry("ChangeText", new Rule(EffectEvent.TEXT_CHANGE, null,
                    (sa, host, memo) -> changeTextEvent(sa)))
    );

    /**
     * What the emitter for this clause needs from before it runs, if anything.
     *
     * <p>Contained the way {@link #after} is (final-fix-1.md F3): a memo reads
     * engine state before the clause has resolved, which can throw on an
     * edge-case board the same way an emitter can. Left uncontained, the
     * throw used to propagate out of this call before
     * {@code PatchedCollectors.clauseHandler()}'s own {@code
     * clauseMemos.push(...)} ever ran, so the {@code Deque} that hook pairs
     * pushes and pops through went one push short: the next sibling or
     * enclosing clause's {@code onClauseResolved} then popped <em>this</em>
     * clause's absent memo and got someone else's instead, and the clause
     * that should have received that memo popped an empty deque and got
     * {@code null} in its place -- one throw, two clauses misattributed.
     *
     * <p>Catching here and always returning (a real memo or {@code null})
     * keeps {@code clauseMemos.push(...)} unconditional: a memo-less API and
     * a memo that failed both read back as {@code null} to the caller, which
     * is indistinguishable to the deque and is exactly what keeps it
     * balanced. The alternative -- skip this push, and have the matching pop
     * skip too -- was rejected: nothing about the failure is visible at
     * {@code onClauseResolved} time except by consulting a second piece of
     * state kept in lockstep with this one, and that second state would
     * itself have to be pushed unconditionally to stay balanced, which is
     * this fix again with extra steps.
     */
    static Object before(SpellAbility sa) {
        Rule rule = ruleFor(sa);
        if (rule == null || rule.memo() == null) {
            return null;
        }
        try {
            return rule.memo().take(sa, sa.getHostCard());
        } catch (RuntimeException e) {
            // Own report, own latch (final-fix-3.md item 4) -- a memo that
            // fails must not look identical to "this mechanic never
            // happens", the exact failure this whole plan exists to end.
            reportMemoFailure(rule.eventType(), e);
            return null;
        }
    }

    /** The event this clause promises, or null when its API promises none. */
    static EffectEvent after(SpellAbility sa, Object memo) {
        Rule rule = ruleFor(sa);
        if (rule == null) {
            return null;
        }
        try {
            return rule.emitter().emit(sa, sa.getHostCard(), memo);
        } catch (RuntimeException e) {
            // An emitter reads engine state that a fizzled or redirected clause
            // may have left in a shape it did not expect. A missing event is a
            // gap; a thrown one would end the resolution the observer is only
            // watching -- so the throw stays swallowed. But swallowed and
            // "this mechanic never happens" must not look identical, which is
            // the failure this whole plan exists to end, so the first one is
            // not silent.
            reportEmitterFailure(rule.eventType(), e);
            return null;
        }
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
    static void reportEmitterFailure(String eventType, RuntimeException e) {
        if (EMITTER_FAILURE_REPORTED.compareAndSet(false, true)) {
            System.err.println("ApiEvents: the emitter for \"" + eventType
                    + "\" threw and was ignored; further emitter failures will "
                    + "not be logged: " + e);
        }
    }

    /** Package-private seam for this hook's own test; production code never calls it. */
    static void resetEmitterFailureLatchForTest() {
        EMITTER_FAILURE_REPORTED.set(false);
    }

    /** Set the first time any rule's memo throws, so the line below prints once. */
    private static final AtomicBoolean MEMO_FAILURE_REPORTED = new AtomicBoolean(false);

    /**
     * As {@link #reportEmitterFailure}, for a rule's memo instead of its
     * emitter (final-fix-3.md item 4).
     *
     * <p>Previously routed through {@link #reportEmitterFailure} itself,
     * which is wrong on two counts once F4 made these lines reachable by a
     * human operator rather than only by a test. First, the wording: a memo
     * ({@link #before}) and an emitter ({@link #after}) are two different
     * halves of one {@link Rule} -- the memo reads engine state
     * <em>before</em> a clause resolves, the emitter builds the event
     * <em>after</em> -- so a report naming "the emitter" for a memo failure
     * sends whoever reads it to the wrong function, with confidence; a wrong
     * name is worse than no name at all. Second, the shared {@code
     * AtomicBoolean}: {@link #EMITTER_FAILURE_REPORTED} latching on a memo
     * failure would silence a later, unrelated, genuine emitter failure on a
     * different clause for the rest of the JVM's run -- two independent
     * failure modes in two independent pieces of code should not be able to
     * mute each other. A dedicated latch is also this branch's own
     * convention rather than an exception to it: {@code
     * AbilityUtils.CLAUSE_LISTENER_FAILURE_REPORTED}, {@code
     * EffectRecordOutcomes.OUTCOME_LISTENER_FAILURE_REPORTED} and {@code
     * PatchedCollectors.UNKNOWN_OUTCOME_TYPE_REPORTED} are each their own
     * latch for their own diagnostic, not one shared across unrelated
     * failure classes.
     */
    static void reportMemoFailure(String eventType, RuntimeException e) {
        if (MEMO_FAILURE_REPORTED.compareAndSet(false, true)) {
            System.err.println("ApiEvents: the memo for \"" + eventType
                    + "\" threw and was ignored; further memo failures will "
                    + "not be logged: " + e);
        }
    }

    /** Package-private seam for this hook's own test; production code never calls it. */
    static void resetMemoFailureLatchForTest() {
        MEMO_FAILURE_REPORTED.set(false);
    }

    /**
     * Package-private seam for {@code ApiEventsTest}'s {@code choice_made}
     * completeness guard: every live {@code RULES} key whose declared event
     * type is {@code choice_made}, read off the table itself rather than
     * hand-copied, so a new one wired in later is picked up automatically
     * instead of the guard quietly checking a stale list.
     */
    static java.util.Set<String> choiceMadeApiKeysForTest() {
        java.util.Set<String> keys = new java.util.TreeSet<>();
        for (Map.Entry<String, Rule> entry : RULES.entrySet()) {
            if (EffectEvent.CHOICE_MADE.equals(entry.getValue().eventType())) {
                keys.add(entry.getKey());
            }
        }
        return keys;
    }

    /**
     * Test-only override, consulted ahead of {@link #RULES} so this hook's
     * own test can force one API's {@link Memo} to throw without a Forge
     * state that genuinely fails.
     *
     * <p>No real script is known to make a memo throw: {@code
     * AbilityUtils.getDefinedPlayers}/{@code getDefinedCards} degrade to an
     * empty list for every unrecognized {@code Defined$} token this branch's
     * own review found rather than throwing, so there is no real card to
     * reach for -- and reaching for one anyway would risk exactly the trap
     * this branch's own test bar warns against, a test that hand-feeds the
     * code a state the engine never produces. This mirrors {@code
     * EffectRecordClauseHookTest}'s {@code ThrowingListener} on the sibling
     * hook in the forge repo: a deliberately-throwing test double standing
     * in for a fault the engine could produce, used to prove the container
     * around it, not to prove anything about the fault itself.
     *
     * <p>Keyed by API name, not global, so a test can make one clause of a
     * multi-clause chain fail while its sibling keeps using the real rule
     * from {@link #RULES} unmodified -- the shape {@code
     * nestedClausesDoNotCrossTheirMemos} already established for the
     * non-throwing case. Empty in production; nothing but
     * {@code ClauseContractTest} calls the setter below.
     */
    private static final Map<String, Rule> RULE_OVERRIDES_FOR_TEST = new java.util.HashMap<>();

    /** Package-private seam for this hook's own test; production code never calls it. */
    static void setRuleOverrideForTest(String apiName, Rule rule) {
        RULE_OVERRIDES_FOR_TEST.put(apiName, rule);
    }

    /** Package-private seam for this hook's own test; production code never calls it. */
    static void clearRuleOverridesForTest() {
        RULE_OVERRIDES_FOR_TEST.clear();
    }

    private static Rule ruleFor(SpellAbility sa) {
        if (sa == null || sa.getApi() == null) {
            return null;
        }
        String name = sa.getApi().name();
        Rule override = RULE_OVERRIDES_FOR_TEST.get(name);
        return override != null ? override : RULES.get(name);
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
     * <p>{@code AddTurnEffect.resolve}, {@code SkipTurnEffect.resolve} and
     * {@code SkipPhaseEffect.resolve} all iterate {@code getTargetPlayers(sa)}
     * — {@code SpellAbilityEffect.getPlayers(false, "Defined", sa)}, not
     * callable from here since it is private to a different package's class —
     * where {@code definedFirst=false} means <b>targeting wins over
     * {@code Defined$}</b> whenever the ability targets at all. The activator
     * is not a safe default: {@code empty_city_ruse.txt} and
     * {@code stonehorn_dignitary.txt} target {@code Opponent},
     * {@code blinding_angel.txt} reads {@code Defined$ TriggeredTarget}, and
     * of roughly eighteen real {@code SkipPhase} scripts, sixteen name a
     * player other than whoever cast or activated the ability.
     *
     * <p>Captured as a memo, before the clause resolves, rather than
     * re-read in {@link #after}: none of these three effects' {@code
     * resolve()} mutates {@code sa}'s own targets today, so reading either
     * side of resolution gives the same answer, but capturing before is the
     * pattern the rest of this codebase already uses for exactly this shape
     * of risk, and it is what lets the clause hook's push/pop pairing —
     * otherwise exercised only on its always-empty path by every other rule
     * here — be pinned against a real, non-trivial value instead of null.
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
    private static EffectEvent withSubjects(EffectEvent event, Object memo) {
        if (memo instanceof List<?> players) {
            for (Object player : players) {
                if (player instanceof Player p) {
                    event.subject(SnapshotBuilder.playerId(p));
                }
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
     * {@code GainOwnership}, but this plan's accuracy constraint does not
     * permit an approximation that is this cheap to close.
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
     * distinction, so this reads that instead; the named-card branch was
     * already guarded correctly by its paired {@code !isEmpty()} check.
     *
     * <p>The direction, even/odd and mode branches are not in the brief's
     * sample for this method, though the {@code ChooseDirection}, {@code
     * ChooseEvenOdd} and {@code GenericChoice} rules above all route through
     * it. These are not optional extras: {@code event_schema.py}'s {@code
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
     * <p>A {@code ChooseSector} branch reading {@code getChosenSector()}
     * lived here briefly too. Removed along with the {@code RULES} entry
     * (see the comment above {@code HealDamage}) once it turned out {@code
     * event_schema.py} already excludes {@code ChooseSector} as unreachable
     * in this corpus's sealed/draft pools -- so the accessor is still real,
     * but nothing in {@code RULES} routes to this method carrying that API
     * any more, and the branch would have been dead weight at best and a
     * stale-state misattribution risk at worst (a host that happened to
     * carry a leftover {@code chosenSector} from something else entirely
     * would have outranked a later, real choice on the same card).
     */
    private static EffectEvent choice(SpellAbility sa, Card host) {
        if (host == null) {
            return null;
        }
        String kind = null;
        String value = null;
        if (host.hasChosenColor()) {
            kind = "color";
            value = host.getChosenColor();
        } else if (host.getChosenType() != null && !host.getChosenType().isEmpty()) {
            kind = "type";
            value = host.getChosenType();
        } else if (host.getChosenNumber() != null) {
            kind = "number";
            value = String.valueOf(host.getChosenNumber());
        } else if (host.getNamedCard() != null && !host.getNamedCard().isEmpty()) {
            kind = "card_name";
            value = host.getNamedCard();
        } else if (host.getChosenPlayer() != null) {
            kind = "player";
            value = SnapshotBuilder.playerId(host.getChosenPlayer());
        } else if (host.getChosenCards() != null && !host.getChosenCards().isEmpty()) {
            kind = "cards";
            value = String.valueOf(host.getChosenCards().size());
        } else if (host.getChosenDirection() != null) {
            kind = "direction";
            value = host.getChosenDirection().name();
        } else if (host.getChosenEvenOdd() != null) {
            kind = "even_odd";
            value = host.getChosenEvenOdd().name();
        } else if (host.getChosenMode() != null && !host.getChosenMode().isEmpty()) {
            kind = "mode";
            value = host.getChosenMode();
        }
        if (kind == null) {
            // The clause chose something this has no reading for. An event
            // naming the choice kind "?" would train the head on a distinction
            // the corpus cannot make.
            return null;
        }
        return new EffectEvent(EffectEvent.CHOICE_MADE)
                .param("choice_kind", kind)
                .param("value", value);
    }

    /**
     * What the retargeted spells point at now.
     *
     * <p>Read after the clause rather than reported by it, because a spell it
     * retargeted is still on the stack carrying its new {@code TargetChoices}
     * -- this is the one outcome in this group that survives on an object
     * other than the host.
     *
     * <p>Fixed in fix round 2 (task-8-fix-2.md, Finding 2) to read {@code
     * getTargetEntities()} rather than only {@code getTargetCards()}.
     * Redirecting a burn spell at a <em>player</em> is the classic use of
     * this mechanic -- Deflection, Misdirection and Bolt Bend are all real
     * and all reach {@code ChangeTargetsEffect}'s default "choose any new
     * legal target" branch, which retargets at a player as readily as a
     * card. {@code getTargetCards()} filters to {@code Card} instances
     * only, so a {@code Player} target was silently dropped and the event
     * still fired with an empty {@code targets} list -- the channel looked
     * wired while reporting nothing distinguishable from "retargeted at
     * nothing." {@code SnapshotBuilder.stackExtrasJson} already reads
     * mixed-entity targets the same way (a {@code GameEntity} loop plus an
     * {@code instanceof} branch, rather than {@code SnapshotBuilder}'s
     * separate {@code entityId(Card)}/{@code playerId(Player)} overloads
     * folded into one serializer), so this follows that file's own
     * precedent rather than inventing a second style for the same problem.
     */
    private static EffectEvent retargeted(SpellAbility sa) {
        List<String> targets = new ArrayList<>();
        if (sa.getTargets() != null) {
            for (SpellAbility changed : sa.getTargets().getTargetSpells()) {
                if (changed.getTargets() == null) {
                    continue;
                }
                for (GameEntity target : changed.getTargets().getTargetEntities()) {
                    if (target instanceof Card card) {
                        targets.add(SnapshotBuilder.entityId(card));
                    } else if (target instanceof Player player) {
                        targets.add(SnapshotBuilder.playerId(player));
                    }
                }
            }
        }
        return new EffectEvent(EffectEvent.TARGETS_CHANGED).param("targets", targets);
    }

    // ── Task 12 helpers ───────────────────────────────────────────────────

    /**
     * {@code getCardsfromTargets(sa)} (SpellAbilityEffect.java:370), mirrored
     * the way {@link #affectedCards} already mirrors {@code getTargetCards}:
     * every real {@code ChangeText} script in the cardsfolder targets {@code
     * ValidTgts$ Card | TgtZone$ Stack,Battlefield} or {@code Permanent} --
     * read through {@link #affectedCards}'s own targeting-first logic -- so
     * the extra {@code getTargetSpells()} union below is dead for today's
     * cardsfolder and kept only because {@code ChangeTextEffect} itself reads
     * it; a future card using the {@code ValidTgts$ Spell} shape would
     * otherwise silently lose its subject.
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
        EffectEvent event = new EffectEvent(EffectEvent.TEXT_CHANGE);
        boolean any = false;
        for (Card c : cardsFromTargets(sa)) {
            event.subject(SnapshotBuilder.entityId(c));
            any = true;
        }
        return any ? event : null;
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
    private static EffectEvent noLongerInCombat(Card host, Object memo) {
        if (!(memo instanceof List<?> before) || before.isEmpty()) {
            return null;
        }
        Combat combat = host.getGame().getCombat();
        EffectEvent event = new EffectEvent(EffectEvent.REMOVED_FROM_COMBAT);
        boolean any = false;
        for (Object o : before) {
            if (o instanceof Card c
                    && (combat == null || !(combat.isAttacking(c) || combat.isBlocking(c)))) {
                event.subject(SnapshotBuilder.entityId(c));
                any = true;
            }
        }
        return any ? event : null;
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
    private static EffectEvent reselectedAttackers(Card host, Object memo) {
        if (!(memo instanceof Map<?, ?> before) || before.isEmpty()) {
            return null;
        }
        Combat combat = host.getGame().getCombat();
        EffectEvent event = new EffectEvent(EffectEvent.REMOVED_FROM_COMBAT);
        boolean any = false;
        for (Map.Entry<?, ?> entry : before.entrySet()) {
            if (!(entry.getKey() instanceof Card c) || entry.getValue() == null) {
                continue;
            }
            GameEntity now = combat == null ? null : combat.getDefenderByAttacker(c);
            if (!entry.getValue().equals(now)) {
                event.subject(SnapshotBuilder.entityId(c));
                any = true;
            }
        }
        return any ? event : null;
    }
}
