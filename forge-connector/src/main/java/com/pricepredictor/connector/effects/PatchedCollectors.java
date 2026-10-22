package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;

import java.lang.reflect.InvocationHandler;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * The four record kinds the engine patch unlocks.
 *
 * <p>{@code rewrite}, {@code trigger}, {@code playability} and {@code mana}
 * records all need a hook at a point the bus never publishes: what a replacement
 * received before it rewrote it, a trigger condition that evaluated to false, a
 * legality verdict the rules computed and discarded, a mana ability that resolved
 * on the inline path. Each is installed reflectively and simply does not run on
 * an unpatched checkout.
 *
 * <p>Two caps live here rather than in the supervisor, because both are
 * per-worker-process quantities the Python side cannot see: {@code --mana-cap}
 * counts resolution records per unique mana-ability text in this JVM, and
 * {@code --playability-rate} samples the {@code decision}-subkind logging points
 * (the {@code attackers} and {@code blockers} ones are always logged, because
 * they fire once per combat rather than once per candidate).
 *
 * <p>{@code continuous} records take no cap at all: coalescing per stable board
 * is itself the cap. A static that applies on every recompute would otherwise
 * write a record per priority pass, and none of them would say anything the
 * first did not.
 */
public final class PatchedCollectors implements AutoCloseable {

    private final LiveGame live;
    private final CollectorContext context;
    private final SnapshotBuilder snapshots;
    private final AttributionMode mode;

    /**
     * The bracket this game's clause events are filed into:
     * {@code recordClauseEvent} is the door, and only the bracket collector for
     * this same game knows which resolution is open.
     */
    private final BusBracketCollector bracket;
    /** The records this game has written, shared by every channel that deduplicates. */
    private final RecordDedup dedup = new RecordDedup();
    private final java.util.Random sampler;
    private final TriggerChannel triggers;
    private final PlayabilityChannel playability;
    private final ContinuousChannel continuous;
    private final ManaChannel mana;

    /** A hook this collector installs a listener on. */
    private record Listener(PatchHooks.Hook hook, Supplier<InvocationHandler> handler) {
    }

    /** Every listener this collector installs, in the order it installs them. */
    private final List<Listener> listeners;


    /** @param bracket where the clause and outcome hooks file their events */
    PatchedCollectors(
            Game game, CollectorContext context, CollectionCaps caps, long seed,
            BusBracketCollector bracket) {
        this.live = new LiveGame(game);
        this.context = context;
        this.snapshots = live.snapshots();
        this.mode = context.mode();
        this.sampler = new java.util.Random(GameRecording.scramble(seed));
        this.bracket = bracket;
        this.triggers = new TriggerChannel(
                live, context, dedup, new TriggerNegativeSampler(sampler));
        this.playability = new PlayabilityChannel(
                live, context, dedup, sampler, caps.playabilityRate());
        this.continuous = new ContinuousChannel(live, context, dedup);
        this.mana = new ManaChannel(
                live, context, new ManaReservoir(caps.manaCap(), sampler));
        this.listeners = List.of(
                new Listener(PatchHooks.REPLACEMENT_LISTENER, this::rewriteHandler),
                new Listener(PatchHooks.TRIGGER_LISTENER, triggers::handler),
                new Listener(PatchHooks.PLAYABILITY_LISTENER, playability::decisionHandler),
                new Listener(PatchHooks.MANA_LISTENER, mana::handler),
                new Listener(PatchHooks.COMBAT_LISTENER, playability::legalityHandler),
                new Listener(PatchHooks.CLAUSE_LISTENER, this::clauseHandler),
                new Listener(PatchHooks.OUTCOME_LISTENER, this::outcomeHandler));
    }

    /** The trigger-condition channel. */
    TriggerChannel triggers() {
        return triggers;
    }

    /** The decision and legality channel, which the random seat also writes through. */
    PlayabilityChannel playability() {
        return playability;
    }

    /** The continuous channel, collected at each phase boundary rather than on a hook. */
    ContinuousChannel continuous() {
        return continuous;
    }

    /** The mana channel. */
    ManaChannel mana() {
        return mana;
    }

    /**
     * Install every hook the checkout offers.
     *
     * @return how many were found
     */
    int install() {
        if (mode == AttributionMode.DEGRADED) {
            return 0;
        }
        int installed = 0;
        for (Listener listener : listeners) {
            if (listener.hook().install(listener.handler().get())) {
                installed++;
            }
        }
        return installed;
    }

    @Override
    public void close() {
        // Before the hooks come down: this is the last moment the game can be
        // said to have ended, and the mana reservoir is only decided then.
        mana.flush();
        for (Listener listener : listeners) {
            listener.hook().uninstall();
        }
        // Last, and outside the hook teardown: the question the tally answers
        // is about the run, so it is asked once per game whatever the game did.
        JVM_REWRITES.endOfGame();
    }

    // ── rewrite records ─────────────────────────────────────────────────

    /**
     * One record per replacement, carrying the event it received, the one it
     * produced, which of the five results it returned and what ran instead.
     *
     * <p>Two replacements stacked on one event each get their own record, and
     * the second's incoming is the first's outgoing — which only holds because
     * the hook deep-copies the parameter map before the call.
     *
     * <p>The last two arguments are read by length rather than assumed present.
     * The listener is installed reflectively over whatever interface the patched
     * checkout declares, so its arity is a runtime fact: against a Forge jar
     * built before the result and the substituted ability were added, this is
     * still a three-argument call and the record honestly says it learnt
     * neither. Indexing {@code args[3]} unguarded would instead throw inside a
     * dynamic proxy, and the engine would surface that in the middle of a
     * replacement as an UndeclaredThrowableException.
     *
     * <p>Package-private rather than private, which none of the other handlers
     * needs to be: that argument order is a runtime contract with no compiler
     * behind it, so {@code RewriteContractTest} calls this with a synthesized
     * argument array in the compiler's place.
     */
    InvocationHandler rewriteHandler() {
        return PatchHooks.REPLACEMENT_LISTENER.listener((callback, args) -> {
            // A fork's own replacements resolve through this same JVM-static
            // hook (ForkCollector.forceResolution
            // runs the fork's forced ability through the real resolution
            // pipeline, which is where replacements apply). args[0] is
            // declared ReplacementEffect on Forge's side -- always a
            // CardTraitBase with a real host -- so a positively-identified
            // other game is dropped; anything this cannot identify is left
            // alone rather than newly dropped.
            CardTraitBase trait = args.get(0, CardTraitBase.class);
            if (trait != null && !live.owns(trait)) {
                return;
            }
            EffectRecord record = rewriteRecord(
                    args.get(0), args.get(1), args.get(2), args.optional(3), args.optional(4));
            if (!dedup.allowDistinct(
                    RecordKind.REWRITE.wireName(), record.payloadJson(), record.stateJson())) {
                return;
            }
            emit(record);
        });
    }

    /**
     * One replacement's record, built from the effect the hook hands over.
     *
     * <p>Package-private rather than inlined into the handler: what the record
     * says about the acting line is the part worth testing, and a dynamic proxy
     * over an interface that only exists on a patched checkout is not a thing a
     * unit test can call.
     *
     * <p>The host is taken as delivered, deliberately not re-resolved through
     * {@code game.getCardState(host)} the way {@code executeReplacementUnobserved}
     * does: that answers an alternate-state card, and the entity id in
     * {@code state.entities} is the one this object carries.
     *
     * <p><b>Every replacement writes a record, including the ones that declined.</b>
     * There used to be a rule here dropping a pair whose halves read alike, and
     * it took 87% of them, because a substitution leaves the map untouched and
     * looked exactly like a replacement that did nothing. The result argument is
     * what tells those apart, so there is no longer such a thing as an
     * uninformative rewrite record: a {@code not_replaced} is this channel's
     * negative, the same role a non-fired evaluation plays on the trigger
     * channel, and it is what teaches when a replacement applies rather than
     * only what it does when it has.
     *
     * @param result     Forge's {@code ReplacementResult}, or null where the
     *                   replacement threw or the listener predates the argument
     * @param replacedBy the {@code ReplaceWith$} ability that stood in for the
     *                   event, or null where none ran
     */
    EffectRecord rewriteRecord(
            Object trait, Object before, Object after, Object result,
            Object replacedBy) {
        CardTraitBase acting = trait instanceof CardTraitBase ctb ? ctb : null;
        Card source = acting == null ? null : acting.getHostCard();
        EffectEvent incoming = RunParamNormalizer.describeParams(trait, before);
        EffectEvent rendered = RunParamNormalizer.describeParams(trait, after);
        // Null rather than a copy, and this is the whole of the contract. The
        // two halves were byte-identical on 87% of the candidates because a
        // substitution does not edit the map -- it runs another ability and
        // returns a result -- so a copy here says "changed nothing observable"
        // about a replacement that exiled the card instead of killing it. What
        // survives the comparison is the genuine in-place edit:
        // ReplaceCounterEffect.setCount, a halved draw, Orim's Cure taking five
        // combat damage down to one.
        EffectEvent outgoing =
                incoming.toJson().equals(rendered.toJson()) ? null : rendered;
        CardTraitBase ran = replacedBy instanceof CardTraitBase ctb ? ctb : null;
        // Keyed through the same resolver as the record's own acting line, so
        // both join the provenance sidecar identically. Usually it answers that
        // same printed line, because ReplaceWith$ names an SVar on the card the
        // replacement is printed on -- what the field carries that `ability`
        // does not is that an ability ran at all, and the rarer case where the
        // overriding ability was granted from somewhere else.
        List<ProvenanceKey> ranKeys = ran == null ? List.of() : ProvenanceKey.keysOf(ran);
        String wireResult = EffectRecord.rewriteResult(result);
        JVM_REWRITES.record(
                RunParamNormalizer.modeOf(trait), wireResult, outgoing != null,
                !ranKeys.isEmpty(), before, after);
        return context.newRecord(RecordKind.REWRITE)
                .actor(live.controllerOf(source))
                .ability(ProvenanceKey.resolve(acting))
                .state(snapshots.ofTrait(source, incoming).render())
                .payload(EffectRecord.rewritePayload(
                        incoming, outgoing, wireResult, ranKeys));
    }

    /** The worker's rewrite tally, shared by every game it plays. */
    static final RewriteTally JVM_REWRITES = new RewriteTally();

    // ── clause events (an API's own parameters) ─────────────────────────

    /**
     * The events an effect API promises, taken from the clause that ran.
     *
     * <p>The memo is kept per thread and not per collector: a clause can resolve
     * another, and the engine runs games on more than one thread. Each memo is
     * wrapped because a deque holds no null, and a clause with no memo still
     * pushes one so that every pop meets its own push.
     */
    private final ThreadLocal<Deque<java.util.Optional<Object>>> clauseMemos =
            ThreadLocal.withInitial(ArrayDeque::new);

    /**
     * Dispatches by method name on a listener this side never compiles
     * against, so the argument order below is a runtime contract rather than
     * one the compiler enforces: {@code onClauseResolving} is called with
     * {@code args[0]} the clause; {@code onClauseResolved} with
     * {@code args[0]} the same clause and {@code args[1]} whether it threw.
     *
     * <p>Package-private rather than private, which none of the other
     * handlers but {@link #rewriteHandler} needs to be: that argument order
     * has no compiler behind it either, so {@code ClauseContractTest} calls
     * this with a synthesized argument array in the compiler's place.
     */
    InvocationHandler clauseHandler() {
        return clauseHandler(ApiEvents::ruleFor);
    }

    /**
     * The same, building each clause's event from the rules {@code rules}
     * names, so a test can stand a rule that throws in for one API.
     */
    InvocationHandler clauseHandler(Function<ApiType, ApiEvents.Rule> rules) {
        return PatchHooks.CLAUSE_LISTENER.listener((callback, args) -> {
            SpellAbility clause = args.get(0, SpellAbility.class);
            if (clause == null) {
                return;
            }
            switch (callback) {
                case ON_CLAUSE_RESOLVING -> clauseResolving(clause, rules);
                case ON_CLAUSE_RESOLVED -> clauseResolved(
                        clause, Boolean.TRUE.equals(args.optional(1)), rules);
            }
        });
    }

    private void clauseResolving(SpellAbility clause, Function<ApiType, ApiEvents.Rule> rules) {
        clauseMemos.get().push(
                java.util.Optional.ofNullable(ApiEvents.before(clause, rules)));
        // A chosen mode's first clause is where a modal resolution is
        // cut into per-mode halves (FR-029a): the live bracket opens
        // the next half here, and a fork running on this thread marks
        // the same boundary in its own event stream. Nested, so
        // "resolved" cannot delimit modes; "resolving" of the first
        // clause is the one moment FR-029b's snapshot has to be taken.
        if (live.owns(clause)) {
            if (bracket != null) {
                CharmModes.ModeStart start = CharmModes.modeStartOf(clause);
                if (start != null) {
                    bracket.beginMode(clause, start.option());
                }
            }
        } else {
            ForkCollector.noteClauseResolving(clause);
        }
    }

    private void clauseResolved(
            SpellAbility clause, boolean threw, Function<ApiType, ApiEvents.Rule> rules) {
        Deque<java.util.Optional<Object>> memos = clauseMemos.get();
        Object memo = memos.isEmpty() ? null : memos.pop().orElse(null);
        if (threw) {
            // A clause that threw did not finish, and an event claiming it
            // did would be a fact the game never contained.
            return;
        }
        if (bracket != null && live.owns(clause)) {
            bracket.recordClauseEvent(ApiEvents.after(clause, memo, rules));
        }
    }

    // ── outcome records (what an effect computed and discarded) ─────────

    /**
     * The outcome types an effect is allowed to name, built from the
     * {@code EffectEvent} constants themselves rather than listed in prose.
     *
     * <p>{@code test_event_schema_completeness.py}'s
     * {@code _emitted_event_types()} is a <em>textual</em> scan for
     * {@code EffectEvent.<NAME>} in this package's source, and a javadoc
     * listing satisfies that scan exactly as well as this set does. Were the
     * types referenced only inside a comment, deleting an effect's {@code
     * note(...)} call, or misspelling the event name, would go unnoticed by
     * the guard built specifically to catch that. This set is what
     * {@link #outcomeHandler()} actually validates incoming types against,
     * so the reference is load-bearing code, not prose a rename can silently
     * strand.
     *
     * <p>{@code CARD_REVEALED} and {@code DAMAGE_PREVENTED} come from the two
     * engine choke points ({@code GameAction.reveal},
     * {@code ReplacementHandler.runSingleReplaceDamageEffect}); the other
     * twelve, from an effect directly. {@code SPELL_COUNTERED} ({@code
     * CounterEffect}), {@code TOKEN_CREATED} ({@code TokenEffectBase}, shared
     * by the six token-making effects) and {@code DICE_ROLLED} ({@code
     * RollDiceEffect}) are the same; {@code PLAYER_WON} is a bus
     * subscription (see {@code BusEventReader#onGameOutcome}), not an
     * outcome-hook type, and does not belong here. {@code TURN_ENDED}
     * ({@code EndTurnEffect}) and {@code TEXT_CHANGE} ({@code
     * TextBoxExchangeEffect} -- {@code ChangeText}'s own contribution to
     * {@code TEXT_CHANGE} is an {@code ApiEvents.RULES} entry, not this hook)
     * are noted by their effects too; the types an {@code ApiEvents.RULES}
     * entry emits never reach this set at all.
     */
    private static final Set<String> KNOWN_OUTCOME_TYPES = Set.of(
            EffectEvent.COIN_FLIPPED, EffectEvent.CLASH_RESOLVED, EffectEvent.VOTE_TAKEN,
            EffectEvent.PILES_MADE, EffectEvent.DUNGEON_VENTURED, EffectEvent.SPELL_COPIED,
            EffectEvent.PERMANENT_COPIED, EffectEvent.CARD_MADE, EffectEvent.RESTRICTION_CHANGE,
            EffectEvent.CARD_REVEALED, EffectEvent.DAMAGE_PREVENTED, EffectEvent.SPELL_COUNTERED,
            EffectEvent.TOKEN_CREATED, EffectEvent.DICE_ROLLED, EffectEvent.TURN_ENDED,
            EffectEvent.TEXT_CHANGE);

    /** Set the first time {@link #outcomeHandler()} sees a type outside {@link #KNOWN_OUTCOME_TYPES}. */
    private static final AtomicBoolean UNKNOWN_OUTCOME_TYPE_REPORTED = new AtomicBoolean();

    /**
     * Prints one line to stderr the first time {@link #outcomeHandler()}
     * receives a type outside {@link #KNOWN_OUTCOME_TYPES}, then stays quiet
     * for every one after it — the same latched-once treatment
     * {@code ApiEvents.reportEmitterFailure} and
     * {@code AbilityUtils.reportClauseListenerFailure} give their own hooks'
     * failures, for the same reason: a deleted {@code note(...)} call or a
     * misspelled event name must not go dark silently.
     */
    private static void reportUnknownOutcomeType(String type) {
        if (UNKNOWN_OUTCOME_TYPE_REPORTED.compareAndSet(false, true)) {
            System.err.println("PatchedCollectors: outcomeHandler() received "
                    + "an outcome type not in KNOWN_OUTCOME_TYPES (\"" + type
                    + "\"); further unrecognized types will not be logged. A "
                    + "renamed or deleted EffectRecordOutcomes.note(...) call "
                    + "in Forge would produce exactly this.");
        }
    }

    /**
     * An effect's own account of what it did.
     *
     * <p>Bound by argument position, because {@code PatchHooks} installs a
     * proxy rather than compiling against the interface: 0 is the ability, 1
     * the event type, 2 the parameters. The event type is a plain string the
     * effect names for itself rather than an enum this side switches on --
     * {@code EffectRecordOutcomes}'s own javadoc explains why: "the caller
     * names the event type, which keeps the vocabulary in the collector that
     * consumes it" -- checked against {@link #KNOWN_OUTCOME_TYPES}, the real
     * vocabulary.
     *
     * <p>The event's subjects are exactly what the effect names under
     * {@code "subjects"}, and an outcome that names none has none. The acting
     * ability's host is never a default: a coin flip or a token made is not
     * something done to the card whose ability did it, and Detain's host is
     * the spell that cast the restriction, not one of the cards it landed on.
     *
     * <p>Three params entries are not generic parameters:
     * <ul>
     *   <li>{@code "subjects"}, carrying the raw Card/Player list of who the
     *   outcome happened to (the cards a restriction landed on, the players
     *   whose library a reveal came from, the player or creature prevented
     *   damage would have been dealt to), is pulled out and converted to refs
     *   on {@link EffectEvent#subject} instead -- the schema's per-type rows
     *   do not list it, so a raw {@code subjects} entry left in {@code params}
     *   would fail the per-type allow-list in the Python {@code EventRecord}'s
     *   own construction.
     *   <li>{@code "source"} (currently only {@code damage_prevented}),
     *   carrying a raw Card, is converted to an entity ref the same way --
     *   every other producer of a {@code source} key ({@code damage_dealt},
     *   and the trigger-derived {@code damage_prevented}) renders one, and a
     *   display name here would silently fail a join against those refs and
     *   collide two sources sharing a name.
     *   <li>{@code "sample"} ({@code token_created} only), one token standing
     *   for its batch, becomes {@code token_script_id} and
     *   {@code characteristics}, rendered by {@link SnapshotBuilder}'s own
     *   rules so the description reads the same as the token's entry on the
     *   board: the same script id, and keywords spelled the way an entity's
     *   are. The token itself never reaches the event.
     * </ul>
     *
     * <p>Package-private rather than private, the same reason
     * {@code clauseHandler} is: that argument order has no compiler behind it
     * either, so this hook's own test calls it directly, the way
     * {@code ClauseContractTest} calls {@code clauseHandler}.
     */
    InvocationHandler outcomeHandler() {
        return PatchHooks.OUTCOME_LISTENER.listener((callback, args) -> {
            String type = args.get(1, String.class);
            if (type == null) {
                return;
            }
            if (!KNOWN_OUTCOME_TYPES.contains(type)) {
                reportUnknownOutcomeType(type);
                return;
            }
            EffectEvent event = new EffectEvent(type);
            if (args.get(2) instanceof Map<?, ?> params) {
                for (Map.Entry<?, ?> entry : params.entrySet()) {
                    if ("subjects".equals(entry.getKey())) {
                        addOutcomeSubjects(event, entry.getValue());
                        continue;
                    }
                    if ("source".equals(entry.getKey())) {
                        event.param("source", outcomeRefOf(entry.getValue()));
                        continue;
                    }
                    if ("sample".equals(entry.getKey())) {
                        if (entry.getValue() instanceof Card token) {
                            event.param("token_script_id", SnapshotBuilder.tokenScriptIdOf(token));
                            event.param("characteristics", tokenCharacteristics(token));
                        }
                        continue;
                    }
                    event.param(String.valueOf(entry.getKey()), entry.getValue());
                }
            }
            SpellAbility ability = args.get(0, SpellAbility.class);
            boolean liveGame = ability != null
                    ? live.owns(ability)
                    : live.ownsNullAbilityOutcome();
            if (bracket != null && liveGame) {
                bracket.recordClauseEvent(event);
            }
        });
    }

    /**
     * Converts a raw Card/Player list reported under the {@code "subjects"}
     * params key into the event's own subject refs -- see
     * {@link #outcomeHandler()} for why that key is not a generic param.
     */
    private static void addOutcomeSubjects(EffectEvent event, Object value) {
        if (!(value instanceof Iterable<?> subjects)) {
            return;
        }
        for (Object subject : subjects) {
            Refs.ofAny(subject).ifPresent(event::subject);
        }
    }

    /**
     * What a made token is, for a {@code token_created} event: its base
     * power and toughness, its type line and its keywords — see
     * {@link #outcomeHandler()} for why these are rendered here.
     */
    private static Map<String, Object> tokenCharacteristics(Card token) {
        Map<String, Object> traits = new LinkedHashMap<>();
        traits.put("power", token.getBasePower());
        traits.put("toughness", token.getBaseToughness());
        traits.put("types", token.getType().toString());
        traits.put("keywords", SnapshotBuilder.keywordNames(token.getKeywords()));
        return traits;
    }

    /**
     * Converts a raw Card/Player reported under the {@code "source"} params
     * key into an entity/player ref -- see {@link #outcomeHandler()} for why
     * that key is not a generic param. Anything else passes through
     * unconverted, which is what lets a future producer of a differently
     * shaped {@code source} degrade to its own {@code String.valueOf} rather
     * than being silently dropped.
     */
    private static Object outcomeRefOf(Object value) {
        return Refs.ofAny(value).<Object>map(ref -> ref).orElse(value);
    }

    private void emit(EffectRecord record) {
        context.emit(record);
    }
}
