package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementType;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 614.10a, a trigger scheduled for a skipped step does not happen, on
 * scenario {@code 614.10a-2}: with Eon Hub on the battlefield A's upkeep is
 * skipped, so Phyrexian Arena's beginning-of-upkeep trigger never happens.
 *
 * <p>The base board, read at a boundary: the untap step of the active player
 * begins (the scenario's setup is the turn before, but a directed fork plays
 * only to the end of its own turn, so the fork is taken in the turn whose
 * upkeep is skipped), a permanent on the battlefield skips that player's
 * upkeep step by a replacement effect (Eon Hub), and a permanent on the
 * battlefield has a trigger at the beginning of that player's upkeep. It binds
 * the skipping permanent.
 *
 * <p>Variant {@code upkeep_happens} ({@code remove_permanent}): the skipping
 * permanent is not on the battlefield, so the upkeep step happens and its
 * trigger with it.
 */
public final class SkippedUpkeepTrigger {

    public static final String RULE = "614.10a";
    static final String SCENARIO = "614.10a-2";
    static final String UPKEEP_HAPPENS = "upkeep_happens";
    static final String UPKEEP = "Upkeep";

    private SkippedUpkeepTrigger() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SkippedUpkeepTrigger::bind, UPKEEP_HAPPENS,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UNTAP)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        boolean triggered = live.getCardsIn(ZoneType.Battlefield).stream()
                .flatMap(card -> card.getTriggers().stream())
                .anyMatch(trigger -> atUpkeepOf(trigger, active));
        if (!triggered) {
            return Optional.empty();
        }
        return live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> skipsUpkeepOf(card, active))
                .findFirst()
                .map(card -> Binding.empty().with("permanent", card));
    }

    /** Whether one of the permanent's replacement effects skips {@code player}'s upkeep step. */
    static boolean skipsUpkeepOf(Card permanent, Player player) {
        for (ReplacementEffect replacement : permanent.getReplacementEffects()) {
            if (replacement.getMode() == ReplacementType.BeginPhase
                    && Boards.lists(replacement.getParam("Phase"), UPKEEP)
                    && "True".equals(replacement.getParam("Skip"))
                    && concerns(replacement.getParam("ValidPlayer"), player, permanent)) {
                return true;
            }
        }
        return false;
    }

    /** Whether the trigger fires at the beginning of {@code player}'s upkeep. */
    static boolean atUpkeepOf(Trigger trigger, Player player) {
        return trigger.getMode() == TriggerType.Phase && Boards.lists(trigger.getParam("Phase"), UPKEEP)
                && concerns(trigger.getParam("ValidPlayer"), player, trigger.getHostCard());
    }

    /** Whether a {@code ValidPlayer$} restriction admits {@code player}; an absent one admits every player. */
    private static boolean concerns(String validPlayer, Player player, Card host) {
        return validPlayer == null || player.isValid(validPlayer.split(","), host.getController(), host, null);
    }
}
