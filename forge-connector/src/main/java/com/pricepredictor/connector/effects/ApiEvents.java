package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
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
    private interface Memo {
        Object take(SpellAbility sa, Card host);
    }

    /** The event, given the clause and whatever its memo captured. */
    @FunctionalInterface
    private interface Emitter {
        EffectEvent emit(SpellAbility sa, Card host, Object memo);
    }

    /** @param eventType named in the once-only failure line an emitter's throw prints */
    private record Rule(String eventType, Memo memo, Emitter emitter) {
    }

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
                            .param("owner", ownerOf(sa))))
    );

    /** What the emitter for this clause needs from before it runs, if anything. */
    static Object before(SpellAbility sa) {
        Rule rule = ruleFor(sa);
        if (rule == null || rule.memo() == null) {
            return null;
        }
        return rule.memo().take(sa, sa.getHostCard());
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

    private static Rule ruleFor(SpellAbility sa) {
        if (sa == null || sa.getApi() == null) {
            return null;
        }
        return RULES.get(sa.getApi().name());
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
}
