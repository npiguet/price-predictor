package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Map;

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

    private record Rule(Memo memo, Emitter emitter) {
    }

    private static final Map<String, Rule> RULES = Map.ofEntries(
            Map.entry("AddTurn", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.TURN_ADDED)
                            .subject(playerOf(sa))
                            .param("count", amount(sa, host, "NumTurns", "1")))),
            Map.entry("SkipTurn", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.TURN_SKIPPED)
                            .subject(playerOf(sa))
                            .param("count", amount(sa, host, "NumTurns", "1")))),
            Map.entry("AddPhase", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.PHASE_ADDED)
                            .subject(playerOf(sa))
                            .param("phase", sa.getParamOrDefault("ExtraPhase", "?"))
                            .param("count", amount(sa, host, "NumPhases", "1")))),
            Map.entry("SkipPhase", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.PHASE_SKIPPED)
                            .subject(playerOf(sa))
                            .param("phase", sa.getParamOrDefault("Phase", "?")))),
            Map.entry("ChangeX", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.X_CHANGED)
                            .param("value", amount(sa, host, "Value", "0")))),
            Map.entry("GainOwnership", new Rule(null, (sa, host, memo) ->
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
            // watching.
            return null;
        }
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
            List<Player> named = AbilityUtils.getDefinedPlayers(
                    sa.getHostCard(), sa.getParam("DefinedPlayer"), sa);
            if (!named.isEmpty()) {
                return SnapshotBuilder.playerId(named.get(0));
            }
        }
        return playerOf(sa);
    }
}
