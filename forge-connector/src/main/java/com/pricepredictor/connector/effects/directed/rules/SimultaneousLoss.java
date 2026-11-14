package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * 104.4a, if all remaining players lose simultaneously the game is a draw,
 * on scenario {@code 104.4a-1}: Flame Rift deals 4 damage to each player at
 * 4 life, both lose at the same state-based check, and the game is a draw.
 *
 * <p>The base board, read at a boundary: the upkeep begins and the active
 * player controls a permanent whose beginning-of-your-upkeep trigger deals N
 * damage to each player or makes each player lose N life, without a choice or
 * a condition on its way (Plague Spitter, Festering Evil, Stormfist
 * Crusader), and every player's life is N or less, so the trigger's
 * resolution leaves both at 0 or less and the one state-based check after it
 * loses them both. The trigger is the engine's own, so no seat chooses
 * whether it happens. It binds the permanent and the active player's
 * opponent.
 *
 * <p>Variant {@code b_has_more} ({@code life}): the opponent has N + 1 life,
 * as B has 5 instead of 4, so it stays in the game at 1 and wins.
 */
public final class SimultaneousLoss {

    public static final String RULE = "104.4a";
    static final String SCENARIO = "104.4a-1";
    static final String B_HAS_MORE = "b_has_more";

    private SimultaneousLoss() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SimultaneousLoss::bind, B_HAS_MORE,
                new Application(ForkOperation.LIFE, SimultaneousLoss::oneMore));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP) || live.getPlayers().size() != 2) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Player opponent = live.getPlayers().stream().filter(player -> player != active).findFirst().orElseThrow();
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            Optional<Integer> amount = lifeTakenFromEach(permanent);
            if (amount.isPresent() && live.getPlayers().stream().allMatch(player -> player.getLife() <= amount.get()
                    && player.canLoseLife() && !player.cantLoseForZeroOrLessLife())) {
                return Optional.of(Binding.empty().with("permanent", permanent).with("opponent", opponent));
            }
        }
        return Optional.empty();
    }

    /** The APIs a trigger's chain may hold: none asks a player anything. */
    private static Set<String> plainApis() {
        return Set.of(ApiType.DamageAll.name(), ApiType.LoseLife.name(), ApiType.Draw.name());
    }

    /**
     * The life each player loses when the permanent's beginning-of-your-upkeep
     * trigger resolves: the amount of the chain's one {@code DamageAll} on
     * every player ({@code ValidPlayers$ Player}) or {@code LoseLife} by every
     * player ({@code Defined$ Player}), a number. Empty for a trigger with a
     * condition, an optional or unless part, or any other API in its chain.
     */
    static Optional<Integer> lifeTakenFromEach(Card permanent) {
        for (Trigger trigger : permanent.getTriggers()) {
            if (trigger.getMode() != TriggerType.Phase || !"Upkeep".equals(trigger.getParam("Phase"))
                    || !"You".equals(trigger.getParam("ValidPlayer"))
                    || trigger.hasParam("IsPresent") || trigger.hasParam("CheckSVar")
                    || trigger.hasParam("OptionalDecider")) {
                continue;
            }
            Optional<Integer> amount = amountOf(permanent, chain(permanent, Boards.executed(trigger)));
            if (amount.isPresent()) {
                return amount;
            }
        }
        return Optional.empty();
    }

    private static List<Map<String, String>> chain(Card host, Map<String, String> root) {
        List<Map<String, String>> found = new ArrayList<>();
        for (Map<String, String> segment = root; !segment.isEmpty() && found.size() < 8;
                segment = Boards.script(host, segment.get("SubAbility"))) {
            found.add(segment);
        }
        return found;
    }

    private static Optional<Integer> amountOf(Card host, List<Map<String, String>> chain) {
        Optional<Integer> amount = Optional.empty();
        for (Map<String, String> segment : chain) {
            String api = Boards.api(segment);
            if (!plainApis().contains(api) || segment.keySet().stream().anyMatch(name ->
                    name.startsWith("Condition") || name.startsWith("Unless") || name.equals("Optional"))) {
                return Optional.empty();
            }
            if (ApiType.DamageAll.name().equals(api) && "Player".equals(segment.get("ValidPlayers"))) {
                amount = Boards.number(segment.get("NumDmg"));
            } else if (ApiType.LoseLife.name().equals(api) && "Player".equals(segment.get("Defined"))) {
                amount = Boards.number(segment.get("LifeAmount"));
            }
        }
        return amount;
    }

    private static LifeOperation oneMore(Binding fork) {
        int amount = lifeTakenFromEach(fork.card("permanent")).orElseThrow(
                () -> new IllegalStateException("the fork's permanent takes no life from each player"));
        return new LifeOperation(fork.player("opponent"), amount + 1);
    }
}
