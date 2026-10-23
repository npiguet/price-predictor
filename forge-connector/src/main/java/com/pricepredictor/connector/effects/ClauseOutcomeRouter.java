package com.pricepredictor.connector.effects;

import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.lang.reflect.InvocationHandler;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Files what an effect says it did into the live game's open bracket.
 *
 * <p>Two hooks, one door. The clause hook builds the events an effect API's
 * own parameters promise, and marks where a modal resolution's next chosen
 * mode begins; the outcome hook carries what an effect computed and would
 * otherwise discard — a coin flip, a vote, a token made. Both file through
 * {@link BusBracketCollector#recordClauseEvent}, because only the bracket
 * collector of this same game knows which resolution is open, and both drop
 * what a fork's resolution reports on the same JVM-static hooks.
 */
final class ClauseOutcomeRouter {

    private final LiveGame live;
    /** The bracket this game's clause events are filed into. */
    private final BusBracketCollector bracket;

    ClauseOutcomeRouter(LiveGame live, BusBracketCollector bracket) {
        this.live = live;
        this.bracket = bracket;
    }

    // ── clause events (an API's own parameters) ─────────────────────────

    /**
     * The events an effect API promises, taken from the clause that ran.
     *
     * <p>The memo is kept per thread and not per collector: a clause can resolve
     * another, and the engine runs games on more than one thread. Each memo is
     * wrapped because a deque holds no null, and a clause with no memo still
     * pushes one so that every pop meets its own push. A fork's clause pushes
     * and pops its own memo exactly as a live one does, because the game
     * check comes after the bookkeeping, so the check cannot desynchronize
     * this deque.
     */
    private final ThreadLocal<Deque<Optional<Object>>> clauseMemos =
            ThreadLocal.withInitial(ArrayDeque::new);

    /**
     * Dispatches by method name on a listener this side never compiles
     * against, so the argument order below is a runtime contract rather than
     * one the compiler enforces: {@code onClauseResolving} is called with
     * {@code args[0]} the clause; {@code onClauseResolved} with
     * {@code args[0]} the same clause and {@code args[1]} whether it threw.
     *
     * <p>Package-private: that argument order has no compiler behind it, so
     * {@code ClauseContractTest} calls this with a synthesized argument array
     * in the compiler's place.
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
        clauseMemos.get().push(Optional.ofNullable(ApiEvents.before(clause, rules)));
        // A chosen mode's first clause is where a modal resolution is
        // cut into per-mode halves (FR-029a): the live bracket opens
        // the next half here, and a fork running on this thread marks
        // the same boundary in its own event stream. Nested, so
        // "resolved" cannot delimit modes; "resolving" of the first
        // clause is the one moment FR-029b's snapshot has to be taken.
        if (live.owns(clause)) {
            CharmModes.ModeStart start = CharmModes.modeStartOf(clause);
            if (start != null) {
                bracket.beginMode(clause, start.option());
            }
        } else {
            ForkCollector.noteClauseResolving(clause);
        }
    }

    private void clauseResolved(
            SpellAbility clause, boolean threw, Function<ApiType, ApiEvents.Rule> rules) {
        Deque<Optional<Object>> memos = clauseMemos.get();
        Object memo = memos.isEmpty() ? null : memos.pop().orElse(null);
        if (threw) {
            // A clause that threw did not finish, and an event claiming it
            // did would be a fact the game never contained.
            return;
        }
        if (live.owns(clause)) {
            bracket.recordClauseEvent(ApiEvents.after(clause, memo, rules));
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
    private static final Set<EventType> KNOWN_OUTCOME_TYPES = EnumSet.of(
            EventType.COIN_FLIPPED, EventType.CLASH_RESOLVED, EventType.VOTE_TAKEN,
            EventType.PILES_MADE, EventType.DUNGEON_VENTURED, EventType.SPELL_COPIED,
            EventType.PERMANENT_COPIED, EventType.CARD_MADE, EventType.RESTRICTION_CHANGE,
            EventType.CARD_REVEALED, EventType.DAMAGE_PREVENTED, EventType.SPELL_COUNTERED,
            EventType.TOKEN_CREATED, EventType.DICE_ROLLED, EventType.TURN_ENDED,
            EventType.TEXT_CHANGE);

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
            System.err.println("ClauseOutcomeRouter: outcomeHandler() received "
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
     * effect names for itself rather than an enum the engine shares with this
     * side -- {@code EffectRecordOutcomes}'s own javadoc explains why: "the
     * caller names the event type, which keeps the vocabulary in the
     * collector that consumes it" -- read here through
     * {@link EventType#fromWire} and checked against
     * {@link #KNOWN_OUTCOME_TYPES}, the real vocabulary.
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
            EffectEvent event = new EffectEvent(type.get());
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
            if (liveGame) {
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
}
