package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.ApiEvents.amount;
import static com.pricepredictor.connector.effects.ApiEvents.playerOf;
import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for turns and phases: a turn or a phase
 * added or skipped.
 */
final class TurnAndPhaseApiRules {

    private TurnAndPhaseApiRules() {
    }

    static void turnAndPhaseRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.TURN_ADDED,
                        (sa, host) -> ClauseTargets.affectedPlayers(sa), TurnAndPhaseApiRules::turns),
                "AddTurn");
        put(rules, Rule.shaped(EventType.TURN_SKIPPED,
                        (sa, host) -> ClauseTargets.affectedPlayers(sa), TurnAndPhaseApiRules::turns),
                "SkipTurn");
        put(rules, Rule.shaped(EventType.PHASE_ADDED, null, TurnAndPhaseApiRules::addedPhase),
                "AddPhase");
        put(rules, Rule.shaped(EventType.PHASE_SKIPPED,
                        (sa, host) -> ClauseTargets.affectedPlayers(sa), TurnAndPhaseApiRules::skippedPhase),
                "SkipPhase");
    }

    /** How many turns are added or skipped, for each player affected. */
    private static EffectEvent turns(
            EffectEvent event, SpellAbility sa, Card host, List<Player> players) {
        return withSubjects(event.param(EventParam.COUNT, amount(sa, host, "NumTurns", "1")), players);
    }

    /** The phase added, how many times, for the activating player. */
    private static EffectEvent addedPhase(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        return event
                .subject(playerOf(sa))
                .param(EventParam.PHASE, sa.getParamOrDefault("ExtraPhase", "?"))
                .param(EventParam.COUNT, amount(sa, host, "NumPhases", "1"));
    }

    /**
     * The phase or step skipped, for each player affected.
     *
     * <p>{@code SkipPhaseEffect} reads {@code Phase$}, else {@code Step$}
     * (Elfhame Sanctuary's {@code Step$ Draw}). Spelled as the snapshot spells
     * its {@code phase} ({@link SnapshotBuilder#phaseName}), never as the
     * script does: {@code BeginCombat} is {@code combat_begin}. Fatespinner's
     * {@code Phase$ Main} is both main phases, which Forge's replacement reads
     * it as, joined by a comma; a value Forge cannot read is {@code "?"}.
     */
    private static EffectEvent skippedPhase(
            EffectEvent event, SpellAbility sa, Card host, List<Player> players) {
        String named = sa.hasParam("Phase") ? sa.getParam("Phase") : sa.getParam("Step");
        return withSubjects(event.param(EventParam.PHASE, phaseSpelling(named)), players);
    }

    /** A script's phase or step as the snapshot's {@code phase}, members of a range joined by commas. */
    private static String phaseSpelling(String scripted) {
        if (scripted == null) {
            return "?";
        }
        try {
            List<String> spelled = new ArrayList<>();
            PhaseType.parseRange(scripted).forEach(phase -> spelled.add(SnapshotBuilder.phaseName(phase)));
            return String.join(",", spelled);
        } catch (IllegalArgumentException e) {
            return "?";
        }
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
}
