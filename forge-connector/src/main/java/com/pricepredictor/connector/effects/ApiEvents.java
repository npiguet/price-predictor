package com.pricepredictor.connector.effects;

import com.google.common.base.Strings;
import forge.game.GameEntity;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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

    /** The table, one family of APIs at a time. */
    private static Map<String, Rule<?>> rules() {
        Map<String, Rule<?>> rules = new HashMap<>();
        turnAndPhaseRules(rules);
        valueAndOwnershipRules(rules);
        choiceRules(rules);
        healGrantAndRetargetRules(rules);
        scheduleAndReplacementRules(rules);
        titleRules(rules);
        combatRemovalRules(rules);
        textRules(rules);
        return Map.copyOf(rules);
    }

    /** One rule for each API that reports the same event the same way. */
    private static void put(Map<String, Rule<?>> into, Rule<?> rule, String... apis) {
        for (String api : apis) {
            if (into.putIfAbsent(api, rule) != null) {
                throw new IllegalStateException("two rules for " + api);
            }
        }
    }

    // ── turns and phases ────────────────────────────────────────────────

    private static void turnAndPhaseRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.TURN_ADDED,
                        (sa, host) -> ClauseTargets.affectedPlayers(sa), ApiEvents::turns),
                "AddTurn");
        put(rules, Rule.shaped(EventType.TURN_SKIPPED,
                        (sa, host) -> ClauseTargets.affectedPlayers(sa), ApiEvents::turns),
                "SkipTurn");
        put(rules, Rule.shaped(EventType.PHASE_ADDED, null, ApiEvents::addedPhase),
                "AddPhase");
        put(rules, Rule.shaped(EventType.PHASE_SKIPPED,
                        (sa, host) -> ClauseTargets.affectedPlayers(sa), ApiEvents::skippedPhase),
                "SkipPhase");
    }

    /** How many turns are added or skipped, for each player affected. */
    private static EffectEvent turns(
            EffectEvent event, SpellAbility sa, Card host, List<Player> players) {
        return withSubjects(event.param("count", amount(sa, host, "NumTurns", "1")), players);
    }

    /** The phase added, how many times, for the activating player. */
    private static EffectEvent addedPhase(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        return event
                .subject(playerOf(sa))
                .param("phase", sa.getParamOrDefault("ExtraPhase", "?"))
                .param("count", amount(sa, host, "NumPhases", "1"));
    }

    /** The phase skipped, for each player affected. */
    private static EffectEvent skippedPhase(
            EffectEvent event, SpellAbility sa, Card host, List<Player> players) {
        return withSubjects(event.param("phase", sa.getParamOrDefault("Phase", "?")), players);
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
                event.subject(Refs.of(player));
            }
        }
        return event;
    }

    // ── X and ownership ─────────────────────────────────────────────────

    private static void valueAndOwnershipRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.X_CHANGED, null, (event, sa, host, memo) ->
                        event.param("value", amount(sa, host, "Value", "0"))),
                "ChangeX");
        put(rules, Rule.shaped(EventType.OWNERSHIP_CHANGE, null, (event, sa, host, memo) ->
                        event.param("owner", ownerOf(sa))),
                "GainOwnership");
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
            List<Player> named = ClauseTargets.definedPlayers(sa, sa.getParam("DefinedPlayer"));
            if (!named.isEmpty()) {
                return Refs.of(named.get(0));
            }
        }
        return playerOf(sa);
    }

    // ── choices ─────────────────────────────────────────────────────────

    /**
     * Every Choose* API, read back off what it left on its host.
     *
     * <p>Two keys are NOT spelled like their effect classes ({@code
     * ApiType}): {@code NameCard} declares {@code ChooseCardNameEffect.class},
     * {@code GenericChoice} declares {@code ChooseGenericEffect.class}. A rule
     * keyed by the CLASS stem matches nothing and fails silently.
     *
     * <p>{@code GenericChoice}, {@code ChooseDirection}, {@code ChooseEvenOdd}
     * and {@code NameCard} are not optional extras needing a judgment call:
     * all four are present in {@code event_schema.py}'s {@code
     * EFFECT_API_EVENTS} (mapped to {@code CHOICE_MADE}) and absent from
     * {@code EXCLUDED_EFFECT_APIS}, so the schema itself declares that these
     * APIs produce {@code choice_made}. A rule that matches here and reads
     * nothing back would be worse than one left out of this table entirely --
     * it looks wired. Their {@link #choice} read branches are required, not
     * optional.
     *
     * <p>{@code ChooseSector} is deliberately NOT wired here, though {@code
     * ApiType} has it and {@code Card.getChosenSector()} is a real accessor:
     * {@code event_schema.py}'s {@code EXCLUDED_EFFECT_APIS} carries {@code
     * "ChooseSector": "Unfinity attraction; not reachable in sealed or
     * draft"} with {@code ChooseSector} correspondingly absent from {@code
     * EFFECT_API_EVENTS}. This corpus is collected from sealed and draft pools
     * only, which never include Unfinity's silver-bordered attraction cards,
     * so the clause this would read can never actually run here. Wiring it
     * needs the schema changed first -- {@code
     * everyChoiceMadeApiHasAReadableChoice} in {@code ApiEventsTest} fails the
     * moment the two disagree, which is the point.
     */
    private static void choiceRules(Map<String, Rule<?>> rules) {
        put(rules, new Rule<>(EventType.CHOICE_MADE, null, (sa, host, memo) -> choice(host)),
                "ChooseColor", "ChooseType", "ChooseCard", "NameCard", "ChooseNumber",
                "ChoosePlayer", "ChooseDirection", "ChooseEvenOdd", "ChooseSource",
                "GenericChoice");
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
     * guarded by {@link Strings#emptyToNull}.
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
     * is no {@code ChooseSector} rule ({@link #choiceRules}): nothing routes
     * that API here, and the reading would be a stale-state misattribution
     * risk (a host carrying a leftover {@code chosenSector} from something
     * else entirely would outrank a later, real choice on the same card).
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

    /**
     * The accessors {@link #choice} reads, in its priority order.
     *
     * <p>Read only when an emitter runs, never while {@link #RULES} is built,
     * so being declared below it is safe.
     */
    private static final List<ChoiceReading> CHOICES = List.of(
            new ChoiceReading("color",
                    host -> host.hasChosenColor() ? host.getChosenColor() : null),
            new ChoiceReading("type", host -> Strings.emptyToNull(host.getChosenType())),
            new ChoiceReading("number", host -> host.getChosenNumber() == null
                    ? null : String.valueOf(host.getChosenNumber())),
            new ChoiceReading("card_name", host -> Strings.emptyToNull(host.getNamedCard())),
            new ChoiceReading("player", host -> Refs.ofNullable(host.getChosenPlayer())),
            new ChoiceReading("cards",
                    host -> host.getChosenCards() == null || host.getChosenCards().isEmpty()
                            ? null : String.valueOf(host.getChosenCards().size())),
            new ChoiceReading("direction", host -> host.getChosenDirection() == null
                    ? null : host.getChosenDirection().name()),
            new ChoiceReading("even_odd", host -> host.getChosenEvenOdd() == null
                    ? null : host.getChosenEvenOdd().name()),
            new ChoiceReading("mode", host -> Strings.emptyToNull(host.getChosenMode())));

    // ── healed damage, granted abilities, retargets ─────────────────────

    private static void healGrantAndRetargetRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.DAMAGE_HEALED, ApiEvents::damageBefore,
                        (event, sa, host, memo) -> event.param("amount", memo == null ? 0 : memo)),
                "HealDamage");
        put(rules, Rule.shaped(EventType.ABILITY_CHANGE, null, ApiEvents::grantedAbilities),
                "Animate", "AnimateAll");
        put(rules, Rule.shaped(EventType.CONTINUOUS_EFFECT_CREATED, null,
                        (event, sa, host, memo) ->
                                event.param("layers", sa.getParamOrDefault("StaticAbilities", ""))),
                "Effect");
        put(rules, new Rule<>(EventType.TARGETS_CHANGED, null, (sa, host, memo) -> retargeted(sa)),
                "ChangeTargets");
    }

    /**
     * Damage on the clause's affected cards, before it heals any of it away.
     *
     * <p>Read through {@link ClauseTargets#affectedCards} rather than off {@code
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
    private static Integer damageBefore(SpellAbility sa, Card host) {
        int total = 0;
        for (Card card : ClauseTargets.affectedCards(sa)) {
            total += card.getDamage();
        }
        return total;
    }

    /** The abilities an animation grants, and whether it removes all others. */
    private static EffectEvent grantedAbilities(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        return event
                .param("abilities", sa.getParamOrDefault("Abilities", ""))
                .param("removed", sa.hasParam("RemoveAllAbilities"));
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
     * EntityJson.stackExtrasJson} reads them, through {@link Refs#ofAny}.
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

    // ── schedules and replacements ──────────────────────────────────────

    private static void scheduleAndReplacementRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.DELAYED_TRIGGER_CREATED, null, ApiEvents::scheduler),
                "DelayedTrigger");
        put(rules, Rule.shaped(EventType.DELAYED_TRIGGER_CREATED,
                        ApiEvents::triggerAmount, ApiEvents::immediateTrigger),
                "ImmediateTrigger");
        put(rules, Rule.shaped(EventType.REPLACEMENT_APPLIED, null, ApiEvents::replacedBy),
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
     * Captured as a memo (read the way {@link #amount} reads it elsewhere)
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

    // ── titles ──────────────────────────────────────────────────────────

    private static void titleRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.MONARCH_CHANGED,
                        (sa, host) -> host.getGame().getMonarch(), ApiEvents::newMonarch),
                "BecomeMonarch");
        put(rules, Rule.shaped(EventType.RING_TEMPTS, null, ApiEvents::tempted),
                "RingTemptsYou");
        put(rules, Rule.shaped(EventType.INITIATIVE_TAKEN, null, ApiEvents::initiativeTaker),
                "TakeInitiative");
    }

    /**
     * monarch_changed: the new monarch, and only when the monarch genuinely
     * changed.
     *
     * <p>{@code GameAction.becomeMonarch} is a no-op (no state change, no
     * trigger) when the named player already is the monarch or {@code
     * canBecomeMonarch()} says no (a static ability, e.g. Crown of Vigor's
     * granted "monarch can't lose the crown" combined with a would-be usurper
     * who cannot become monarch some other way) -- replicating that condition
     * here would duplicate the engine's own classification, a second copy that
     * drifts. Reading the actual before/after monarch instead needs no such
     * duplication and is correct regardless of how many players {@code
     * BecomeMonarchEffect}'s loop iterates.
     *
     * @param before the monarch before the clause ran
     */
    private static EffectEvent newMonarch(
            EffectEvent event, SpellAbility sa, Card host, Player before) {
        Player after = host.getGame().getMonarch();
        if (after == null || after.equals(before)) {
            return null;
        }
        return event.subject(Refs.of(after));
    }

    /**
     * ring_tempts: the tempted player, always the activator --
     * {@code RingTemptsYouEffect.resolve()} reads {@code
     * sa.getActivatingPlayer()} unconditionally, never {@code
     * Defined$}/targeting.
     *
     * <p>The Ring-bearer is a separable second fact from the same clause
     * ({@code Card.setRingBearer}, read back post-resolution since it is only
     * known once the player's controller has chosen one) and is named as a
     * second subject when one was actually chosen -- null when the player
     * controls no creatures, per {@code chooseSingleEntityForEffect}'s own
     * contract on an empty candidate list.
     */
    private static EffectEvent tempted(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        Player p = sa.getActivatingPlayer();
        if (p == null) {
            return null;
        }
        event.subject(Refs.of(p));
        Card ringBearer = p.getRingBearer();
        if (ringBearer != null) {
            event.subject(Refs.of(ringBearer));
        }
        return event;
    }

    /**
     * initiative_taken: the player taking the initiative.
     *
     * <p>{@code GameAction.takeInitiative} fires its trigger unconditionally
     * -- "You can take the initiative even if you already have it" is in the
     * engine's own comment -- unlike {@code becomeMonarch}, which is a no-op
     * when the target already holds the title. So this is NOT a before/after
     * diff (that would wrongly suppress the already-has-it re-affirmation): it
     * reads the post-resolution holder and confirms they are one of this
     * clause's own targets, which is true whether the holder changed or was
     * reconfirmed.
     *
     * <p>Known limit (record-schema.md "initiative_taken can miss a
     * redirect"): a target who satisfies {@code isInGame()} but has also
     * already {@code hasLost()} triggers {@code GameAction.takeInitiative}'s
     * own redirect to the next player, which that method's own unconditional
     * fall-through then overwrites back to the original (lost) target before
     * this rule ever reads {@code game.getHasInitiative()} -- so the event
     * still fires, but can name the lost player instead of who the redirect
     * actually landed on. Not replicated here: doing so would mean
     * re-deriving {@code GameAction}'s own fall-through as a second copy of
     * it, which drifts the way any duplicated classification does.
     */
    private static EffectEvent initiativeTaker(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        Player holder = host.getGame().getHasInitiative();
        if (holder == null || !ClauseTargets.affectedPlayers(sa).contains(holder)) {
            return null;
        }
        return event.subject(Refs.of(holder));
    }

    // ── combat removals ─────────────────────────────────────────────────

    private static void combatRemovalRules(Map<String, Rule<?>> rules) {
        put(rules, new Rule<>(EventType.REMOVED_FROM_COMBAT,
                        ApiEvents::combatantsBefore,
                        (sa, host, memo) -> noLongerInCombat(host, memo)),
                "RemoveFromCombat");
        put(rules, new Rule<>(EventType.REMOVED_FROM_COMBAT,
                        ApiEvents::attackerDefendersBefore,
                        (sa, host, memo) -> reselectedAttackers(host, memo)),
                "ChangeCombatants");
        put(rules, Rule.shaped(EventType.COMBAT_ENDED,
                        ApiEvents::combatAttacker, ApiEvents::endedCombatOf),
                "EndCombatPhase");
    }

    /**
     * {@code RemoveFromCombat}'s memo: which of the clause's own targets are
     * genuinely attacking or blocking right now, before the clause runs.
     *
     * <p>removed_from_combat names every creature actually removed, not just
     * the first. {@code RemoveFromCombatEffect} gates each target on several
     * conditions (combat null, not in play, a stale LKI reference) this file
     * cannot see from outside; reading whether each memoized combatant is
     * still attacking or blocking afterward ({@link #noLongerInCombat})
     * sidesteps replicating them -- the before/after shape of {@link
     * #damageBefore} and {@link #newMonarch}.
     */
    private static List<Card> combatantsBefore(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        List<Card> inCombat = new ArrayList<>();
        if (combat != null) {
            for (Card c : ClauseTargets.affectedCards(sa)) {
                if (CombatShape.isCombatant(combat, c)) {
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
            if (!CombatShape.isCombatant(combat, c)) {
                event.subject(Refs.of(c));
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
            for (Card c : ClauseTargets.affectedCards(sa)) {
                map.put(c, combat.getDefenderByAttacker(c));
            }
        }
        return map;
    }

    /**
     * Every memoized attacker whose defender assignment actually changed --
     * exactly the creatures {@code SpellAbilityEffect.addToCombat}'s
     * {@code combat.removeFromCombat(c)} call ran for, since that call fires
     * only immediately before re-adding the same creature against a new
     * defender. A card that was never attacking
     * ({@code null} before) is skipped: nothing could have removed it.
     *
     * <p>So "removed from combat" for {@code ChangeCombatants} means a
     * targeted attacker's defender assignment actually changed, read the same
     * way {@code ChangeCombatantsEffect} itself reads its own {@code
     * originalDefender}/{@code defender} pair for retargeting triggers, not
     * replicated from the {@code Optional}-decline or
     * already-attacking-that-defender guards.
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
                event.subject(Refs.of(c));
            }
        }
        return event.subjects().isEmpty() ? null : event;
    }

    /**
     * {@code EndCombatPhase}'s memo: the attacking player whose combat phase
     * the clause ends, read before the clause clears it, or null outside
     * combat. {@code EndCombatPhaseEffect} does nothing outside a combat
     * phase (CR 723.2g), so no combat emits nothing.
     */
    private static Player combatAttacker(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        return combat == null ? null : combat.getAttackingPlayer();
    }

    /** combat_ended: the attacking player whose combat phase ended, if one did. */
    private static EffectEvent endedCombatOf(
            EffectEvent event, SpellAbility sa, Card host, Player attacker) {
        return attacker == null ? null : event.subject(Refs.of(attacker));
    }

    // ── text changes ────────────────────────────────────────────────────

    /**
     * {@code ChangeText} alone: {@code ExchangeTextBox}'s own contribution to
     * text_change is a {@code note()} call from {@code TextBoxExchangeEffect}
     * itself (see that class), not a rule here -- its two early returns
     * ({@code checkValidDuration}, {@code tgtCards.size() < 2}) are not
     * cheaply replicable from outside without duplicating engine logic.
     */
    private static void textRules(Map<String, Rule<?>> rules) {
        put(rules, new Rule<>(EventType.TEXT_CHANGE, null, (sa, host, memo) -> changeTextEvent(sa)),
                "ChangeText");
    }

    /**
     * text_change: the card(s) whose text changed, or {@code null} when the
     * clause names no word to replace at all -- {@code
     * ChangeTextEffect.resolve()}'s target loop still fires {@code
     * GameEventCardStatsChanged} for every target even then, so the params
     * themselves, read directly with no post-resolution state needed, are the
     * only signal that anything textual actually changed.
     */
    private static EffectEvent changeTextEvent(SpellAbility sa) {
        if (!sa.hasParam("ChangeColorWord") && !sa.hasParam("ChangeTypeWord")) {
            return null;
        }
        EffectEvent event = new EffectEvent(EventType.TEXT_CHANGE);
        for (Card c : ClauseTargets.cardsFromTargets(sa)) {
            event.subject(Refs.of(c));
        }
        return event.subjects().isEmpty() ? null : event;
    }

    // ── readings several families share ─────────────────────────────────

    private static int amount(SpellAbility sa, Card host, String key, String fallback) {
        return AbilityUtils.calculateAmount(host, sa.getParamOrDefault(key, fallback), sa);
    }

    private static String playerOf(SpellAbility sa) {
        return Refs.ofNullable(sa.getActivatingPlayer());
    }
}
