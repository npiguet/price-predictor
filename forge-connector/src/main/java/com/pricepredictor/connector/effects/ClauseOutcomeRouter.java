package com.pricepredictor.connector.effects;

import com.google.common.collect.Multiset;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.EnumSet;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Function;

/**
 * Files what an effect says it did into the live game's open bracket.
 *
 * <p>Two hooks, one door. The clause hook builds the events an effect API's
 * own parameters promise, and marks where a modal resolution's next chosen
 * mode begins; the outcome hook carries what an effect computed and would
 * otherwise discard — a coin flip, a vote, a token made. Both file through
 * {@link BusBracketCollector#recordClauseEvent}, because only the bracket
 * collector of this same game knows which resolution is open, and both hand
 * the events a fork reports on the same JVM-static hooks to the probe
 * running on their thread ({@link Forks#forkHooks}), if any. A forced
 * resolution's are its fork-bound recording's ({@link ForcedResolution}).
 */
final class ClauseOutcomeRouter {

    private final LiveGame live;
    /** The bracket this game's clause events are filed into. */
    private final BusBracketCollector bracket;
    /** Told each clause's end, to close a mana activation whose effect never reported its mana. */
    private final ManaChannel mana;

    ClauseOutcomeRouter(LiveGame live, BusBracketCollector bracket, ManaChannel mana) {
        this.live = live;
        this.bracket = bracket;
        this.mana = mana;
    }

    // ── clause events (an API's own parameters) ─────────────────────────

    /**
     * The events an effect API promises, taken from the clause that ran.
     *
     * <p>Each clause's observation -- its rule and what that rule's memo took --
     * is kept per thread and not per collector: a clause can resolve another,
     * and the engine runs games on more than one thread. A clause whose API
     * promises nothing, or whose memo threw, still pushes one so that every
     * pop meets its own push. A fork's clause pushes and pops its own
     * observation exactly as a live one does, because the game check comes
     * after the bookkeeping, so the check cannot desynchronize this deque.
     */
    private final ThreadLocal<Deque<ApiEvents.ClauseObservation<?>>> clauseObservations =
            ThreadLocal.withInitial(ArrayDeque::new);

    /**
     * Dispatches by method name on a listener this side never compiles
     * against, so the argument order below is a runtime contract rather than
     * one the compiler enforces: {@code onClauseResolving} is called with
     * {@code args[0]} the clause; {@code onClauseResolved} with
     * {@code args[0]} the same clause and {@code args[1]} Forge's {@code
     * ClauseRun}, read by its constant's name ({@link
     * PatchHooks#clauseRunName}).
     *
     * <p>Forge tells "resolved" where the clause's own effect ends, before its
     * sub-abilities resolve, so the clauses of a chain, a charm's chosen modes
     * included, are told one after another, never one inside another (A1).
     * A triggered ability's wrapper is no clause and is never told (A2).
     *
     * <p>Package-private: that argument order has no compiler behind it, so
     * {@code ClauseContractTest} calls this with a synthesized argument array
     * in the compiler's place.
     */
    Listener clauseHandler() {
        return clauseHandler(ApiEvents::ruleFor);
    }

    /**
     * The same, building each clause's event from the rules {@code rules}
     * names, so a test can stand a rule that throws in for one API.
     */
    Listener clauseHandler(Function<ApiType, ApiEvents.Rule<?>> rules) {
        return PatchHooks.CLAUSE_LISTENER.listener((callback, args) -> {
            SpellAbility clause = args.get(0, SpellAbility.class);
            if (clause == null) {
                return;
            }
            switch (callback) {
                case ON_CLAUSE_RESOLVING -> clauseResolving(clause, rules);
                case ON_CLAUSE_RESOLVED -> clauseResolved(
                        clause, PatchHooks.clauseRunName(args.optional(1)), rules);
                default -> { }
            }
        });
    }

    private void clauseResolving(SpellAbility clause, Function<ApiType, ApiEvents.Rule<?>> rules) {
        clauseObservations.get().push(ApiEvents.observe(clause, rules));
        // A chosen mode's first clause is where a modal resolution is
        // cut into per-mode halves (FR-029a): the live bracket opens
        // the next half here; a forced resolution's fork-bound recording
        // hears its own fork's clauses. "Resolving" of the
        // first clause is the one moment FR-029b's snapshot has to be taken.
        if (live.owns(clause)) {
            // The timestamp a floating effect's rows will carry is drawn
            // while this clause runs; the note is their only join to a line.
            ClauseNotes.of(live.game()).clauseResolving(live.game(), clause);
            CharmModes.ModeStart start = CharmModes.modeStartOf(clause);
            if (start != null) {
                bracket.beginMode(clause, start.option());
            }
        }
    }

    /**
     * A clause's own effect is over: its events are filed, and whether its
     * condition held (FR-034), as Forge decided it.
     *
     * <p>Only a clause whose own effect ran files its events (A3): one whose
     * condition failed, whose unless cost decided against it or that threw
     * did nothing its API promises, though its sub-abilities may still
     * resolve. A jar whose listener passes no {@code ClauseRun} is read as
     * the clause having run, the one answer such a jar could not tell.
     *
     * @param run the {@code ClauseRun} constant's name, or null
     */
    private void clauseResolved(
            SpellAbility clause, String run, Function<ApiType, ApiEvents.Rule<?>> rules) {
        Deque<ApiEvents.ClauseObservation<?>> observations = clauseObservations.get();
        ApiEvents.ClauseObservation<?> observed = observations.isEmpty()
                ? ApiEvents.unobserved(clause, rules)
                : observations.pop();
        boolean ran = ran(run);
        if (!live.owns(clause)) {
            forkClauseResolved(clause, ran ? observed : null);
            return;
        }
        ClauseNotes.of(live.game()).clauseResolved(live.game(), clause);
        mana.clauseResolved(clause);
        // Filed into the half the clause resolved in: a later mode's first
        // clause has not begun yet (A1).
        bracket.recordClauseCondition(clause, ConditionsMet.of(clause, run));
        if (ran) {
            bracket.recordClauseRan(clause);
            observed.finish().forEach(bracket::recordClauseEvent);
        }
    }

    /**
     * Whether a clause's own effect ran, by the {@code ClauseRun} Forge
     * reported: one whose condition failed, whose unless cost decided against
     * it or that threw did nothing its API promises. A jar whose listener
     * passes no {@code ClauseRun} is read as the clause having run, the one
     * answer such a jar could not tell.
     *
     * @param run the {@code ClauseRun} constant's name, or null
     */
    static boolean ran(String run) {
        return run == null || "RAN".equals(run);
    }

    /**
     * A fork's clause's events, told to the probe listening, as the live
     * bracket files them: built only for a probe listening, which a fork
     * played on or a forced resolution is not (its own recording's router
     * files them).
     *
     * @param ran the clause's observation when its own effect ran, else null
     */
    private static void forkClauseResolved(SpellAbility clause, ApiEvents.ClauseObservation<?> ran) {
        if (ran != null && Forks.forkHooksListening()) {
            ran.finish().forEach(event -> Forks.forkHooks().clauseEvent(clause, event));
        }
    }

    // ── outcome records (what an effect computed and discarded) ─────────

    /**
     * The outcome types an effect is allowed to name, built from the
     * {@link EventType} members themselves rather than listed in prose.
     *
     * <p>{@code test_event_schema_completeness.py}'s
     * {@code _emitted_event_types()} is a <em>textual</em> scan for
     * {@code EventType.<NAME>} in this package's source, and a javadoc
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
     * fourteen, from an effect directly. {@code SPELL_COUNTERED} ({@code
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
    private static final Set<EventType> KNOWN_OUTCOME_TYPES = EnumSet.of(
            EventType.COIN_FLIPPED, EventType.CLASH_RESOLVED, EventType.VOTE_TAKEN,
            EventType.PILES_MADE, EventType.DUNGEON_VENTURED, EventType.SPELL_COPIED,
            EventType.PERMANENT_COPIED, EventType.CARD_MADE, EventType.RESTRICTION_CHANGE,
            EventType.CARD_REVEALED, EventType.DAMAGE_PREVENTED, EventType.SPELL_COUNTERED,
            EventType.TOKEN_CREATED, EventType.DICE_ROLLED, EventType.TURN_ENDED,
            EventType.TEXT_CHANGE);

    /** Raised the first time {@link #outcomeHandler()} sees a type outside {@link #KNOWN_OUTCOME_TYPES}. */
    private static final OnceWarning UNKNOWN_OUTCOME_TYPE = new OnceWarning();

    /**
     * Prints one line to stderr the first time {@link #outcomeHandler()}
     * receives a type outside {@link #KNOWN_OUTCOME_TYPES}, then stays quiet
     * for every one after it — the same latched-once treatment
     * {@code ApiEvents.reportEmitterFailure} and Forge's
     * {@code EffectRecordHook.contain} give their own hooks' failures, for
     * the same reason: a deleted {@code note(...)} call or a misspelled event
     * name must not go dark silently.
     */
    private static void reportUnknownOutcomeType(String type) {
        UNKNOWN_OUTCOME_TYPE.raise(() -> "ClauseOutcomeRouter: outcomeHandler() received "
                + "an outcome type not in KNOWN_OUTCOME_TYPES (\"" + type
                + "\"); further unrecognized types will not be logged. A "
                + "renamed or deleted EffectRecordOutcomes.note(...) call "
                + "in Forge would produce exactly this.");
    }

    /**
     * An effect's own account of what it did.
     *
     * <p>Bound by argument position, because {@code PatchHooks} installs a
     * proxy rather than compiling against the interface: 0 is the ability, 1
     * the event type, 2 the parameters. The event type is a plain string the
     * effect names for itself rather than an enum the engine shares with this
     * side -- {@code EffectRecordOutcomes}'s own javadoc explains why: "the
     * event type names the outcome, and the collector that consumes it owns
     * what each type means" -- read here through
     * {@link EventType#fromWire} and checked against
     * {@link #KNOWN_OUTCOME_TYPES}, the real vocabulary.
     *
     * <p>The event's subjects are exactly what the effect names under
     * {@code "subjects"}, and an outcome that names none has none. The acting
     * ability's host is never a default: a token made is not something done
     * to the card whose ability did it, and Detain's host is the spell that
     * cast the restriction, not one of the cards it landed on. A coin flip
     * names the player who flipped, as dice name their roller.
     *
     * <p>Filed as any event is ({@link BusBracketCollector#recordClauseEvent}):
     * a coin a cost flips (cumulative upkeep's {@code CostFlipCoin}) is the
     * resolution's that asks for it while it resolves, and otherwise the
     * payment's, which reaches no record and is counted as dropped.
     * Which params entries are not generic parameters is
     * {@link #outcomeEvent}'s to say.
     *
     * <p>Package-private rather than private, the same reason
     * {@code clauseHandler} is: that argument order has no compiler behind it
     * either, so this hook's own test calls it directly, the way
     * {@code ClauseContractTest} calls {@code clauseHandler}.
     */
    Listener outcomeHandler() {
        return PatchHooks.OUTCOME_LISTENER.listener((callback, args) -> {
            String wireName = args.get(1, String.class);
            if (wireName == null) {
                return;
            }
            Optional<EventType> type = EventType.fromWire(wireName)
                    .filter(KNOWN_OUTCOME_TYPES::contains);
            if (type.isEmpty()) {
                reportUnknownOutcomeType(wireName);
                return;
            }
            Map<?, ?> params = paramsOf(args);
            EffectEvent event = outcomeEvent(type.get(), params);
            if (RunParamNormalizer.preventedNothing(event)) {
                return;
            }
            SpellAbility ability = args.get(0, SpellAbility.class);
            if (live.ownsOutcome(ability, params)) {
                bracket.recordClauseEvent(event);
            } else {
                // A forced resolution's or a probe's, which hears it through
                // the fork hooks; nobody's when no such fork is listening.
                Forks.forkHooks().outcome(ability, event);
            }
        });
    }

    /** The outcome's params, empty where the hook passed none. */
    private static Map<?, ?> paramsOf(Args args) {
        return args.get(2) instanceof Map<?, ?> given ? given : Map.of();
    }

    /**
     * The event an outcome's params describe: every entry a generic
     * parameter, except five.
     *
     * <ul>
     *   <li>{@code "subjects"}, carrying the raw Card/Player list of who the
     *   outcome happened to (the cards a restriction landed on, the players
     *   who revealed cards, the player or creature prevented
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
     *   <li>{@code "from_zone"} ({@code card_revealed}) and {@code "to_zone"}
     *   ({@code card_made}), Forge's {@code ZoneType} names, are spelled as
     *   every other zone of a record is ({@link CardSpelling#zoneName}),
     *   lower-cased; the placeholder Forge writes for no zone ({@code "?"},
     *   {@code "null"}) writes no parameter.
     *   <li>{@code "sample"} ({@code token_created} only), one token standing
     *   for its batch, becomes {@code token_script_id} and
     *   {@code characteristics}, rendered by {@link SnapshotBuilder}'s own
     *   rules so the description reads the same as the token's entry on the
     *   board: the same script id, and keywords spelled the way an entity's
     *   are. The token itself never reaches the event.
     * </ul>
     */
    private static EffectEvent outcomeEvent(EventType type, Map<?, ?> params) {
        EffectEvent event = new EffectEvent(type);
        // In key order: Forge builds most outcomes' params with Map.of, whose
        // iteration order changes from one JVM to the next, and the event's
        // parameters are written in the order they are put.
        Map<String, Object> ordered = new TreeMap<>();
        params.forEach((key, value) -> ordered.put(String.valueOf(key), value));
        ordered.entrySet().forEach(entry -> addOutcomeParam(event, entry));
        return event;
    }

    /** One entry of an outcome's params, filed as {@link #outcomeEvent} says. */
    private static void addOutcomeParam(EffectEvent event, Map.Entry<String, Object> entry) {
        String key = entry.getKey();
        Object value = entry.getValue();
        switch (key) {
            case "subjects" -> addOutcomeSubjects(event, value);
            case "source" -> event.param(EventParam.SOURCE, outcomeRefOf(value));
            case "from_zone", "to_zone" -> event.paramNamed(key, outcomeZoneOf(value));
            case "sample" -> addTokenSample(event, entry);
            default -> event.paramNamed(key, inKeyOrder(value));
        }
    }

    /** The token standing for its batch, as {@code token_script_id} and {@code characteristics}; anything else is no sample. */
    private static void addTokenSample(EffectEvent event, Map.Entry<String, Object> sample) {
        if (sample.getValue() instanceof Card token) {
            event.param(EventParam.TOKEN_SCRIPT_ID, CardSpelling.tokenScriptIdOf(token));
            event.param(EventParam.CHARACTERISTICS, tokenCharacteristics(token));
        }
    }

    /**
     * {@code value} with every map inside it in key order and every set in
     * the order of its members' spelling, lists kept as they are: a nested
     * map Forge built with {@code Map.of} (a die's {@code natural} and
     * {@code final}) iterates in an order that changes from one JVM to the
     * next, and is written in the order it iterates.
     */
    static Object inKeyOrder(Object value) {
        if (value instanceof Map<?, ?> map) {
            return mapInKeyOrder(map);
        }
        if (value instanceof Set<?> set) {
            return setInKeyOrder(set);
        }
        if (value instanceof List<?> list) {
            return listInKeyOrder(list);
        }
        return value;
    }

    private static Map<String, Object> mapInKeyOrder(Map<?, ?> map) {
        Map<String, Object> ordered = new TreeMap<>();
        map.forEach((key, member) -> ordered.put(String.valueOf(key), inKeyOrder(member)));
        return ordered;
    }

    private static List<Object> setInKeyOrder(Set<?> set) {
        List<Object> members = new ArrayList<>();
        for (Object member : set) {
            members.add(inKeyOrder(member));
        }
        members.sort(Comparator.comparing(String::valueOf));
        return members;
    }

    private static List<Object> listInKeyOrder(List<?> list) {
        List<Object> members = new ArrayList<>(list.size());
        for (Object member : list) {
            members.add(inKeyOrder(member));
        }
        return members;
    }

    /**
     * Converts a raw Card/Player list reported under the {@code "subjects"}
     * params key into the event's own subject refs -- see
     * {@link #outcomeEvent} for why that key is not a generic param.
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
     * power and toughness, its type line, its keywords, who controls it and
     * the counters it entered with ({@code counters}, counter name to count
     * as an entity spells them, empty for none) — see {@link #outcomeEvent}
     * for why these are rendered here. Read off the token on the
     * battlefield, which the counters it was made with have already reached;
     * Forge batches tokens by controller and counters too, so the sample
     * speaks for its whole batch.
     */
    private static Map<String, Object> tokenCharacteristics(Card token) {
        Map<String, Object> traits = new LinkedHashMap<>();
        traits.put("power", token.getBasePower());
        traits.put("toughness", token.getBaseToughness());
        traits.put("types", token.getType().toString());
        traits.put("keywords", CardSpelling.keywordNames(token.getKeywords()));
        traits.put("controller", Refs.ofNullable(token.getController()));
        Map<String, Integer> counters = new TreeMap<>();
        for (Multiset.Entry<CounterType> entry : token.getCounters().entrySet()) {
            counters.merge(CardSpelling.counterName(entry.getElement()), entry.getCount(), Integer::sum);
        }
        traits.put("counters", counters);
        return traits;
    }

    /** A zone an outcome names, as a record spells it, or null for none. */
    private static String outcomeZoneOf(Object value) {
        // "?" and "null" are what Forge writes for no zone.
        return "?".equals(value) || "null".equals(value) ? null : CardSpelling.zoneName(value);
    }

    /**
     * Converts a raw Card/Player reported under the {@code "source"} params
     * key into an entity/player ref -- see {@link #outcomeEvent} for why
     * that key is not a generic param. Anything else passes through
     * unconverted, which is what lets a future producer of a differently
     * shaped {@code source} degrade to its own {@code String.valueOf} rather
     * than being silently dropped.
     */
    private static Object outcomeRefOf(Object value) {
        return Refs.ofAny(value).<Object>map(ref -> ref).orElse(value);
    }
}
