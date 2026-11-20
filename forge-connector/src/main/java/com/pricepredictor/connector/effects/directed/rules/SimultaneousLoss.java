package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
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
import java.util.stream.Stream;

/**
 * 104.4a, if all remaining players lose simultaneously the game is a draw,
 * inverted (FR-014a) on scenario {@code 104.4a-2}: A's Plague Spitter deals
 * 1 damage to each player at its controller's upkeep, and both players, at
 * 20, stay in the game; with each player at 1 life both lose at the same
 * state-based check, and the game is a draw.
 *
 * <p>The base board, read at a boundary: the upkeep begins and the active
 * player controls a permanent whose beginning-of-your-upkeep trigger deals N
 * damage to each player or makes each player lose N life, without a choice or
 * a condition on its way (Plague Spitter, Festering Evil, Stormfist
 * Crusader), every player can lose life and loses the game at 0 life, and a
 * player has more than N life: the common board, where the trigger's
 * resolution leaves someone in the game. The trigger is the engine's own, so
 * no seat chooses whether it happens. It binds the permanent, the active
 * player and its opponent.
 *
 * <p>Variant {@code both_at_one} ({@code life}): each player above N life is
 * at N, as A and B have 1 instead of 20, so the trigger's resolution leaves
 * both at 0 and the one state-based check after it loses them both, the
 * rule's point. The edit never raises a life total.
 */
public final class SimultaneousLoss {

    public static final String RULE = "104.4a";
    static final String SCENARIO = "104.4a-2";
    static final String BOTH_AT_ONE = "both_at_one";

    private SimultaneousLoss() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SimultaneousLoss::bind, BOTH_AT_ONE,
                new Application(ForkOperation.LIFE, SimultaneousLoss::inReach));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP) || !twoPlayersLosingAtZeroLife(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Player opponent = live.getPlayers().stream().filter(player -> player != active).findFirst().orElseThrow();
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            Optional<Integer> amount = lifeTakenFromEach(permanent);
            if (amount.isPresent() && live.getPlayers().stream().anyMatch(player -> player.getLife() > amount.get())) {
                return Optional.of(Binding.empty().with("permanent", permanent).with("player", active)
                        .with("opponent", opponent));
            }
        }
        return Optional.empty();
    }

    /** Whether the game has two players, each of whose life can go down and who loses the game at 0 life. */
    private static boolean twoPlayersLosingAtZeroLife(Game live) {
        return live.getPlayers().size() == 2 && live.getPlayers().stream()
                .allMatch(player -> player.canLoseLife() && !player.cantLoseForZeroOrLessLife());
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
            if (!plainUpkeepTrigger(trigger)) {
                continue;
            }
            Optional<Integer> amount = amountOf(permanent, chain(permanent, Scripts.executed(trigger)));
            if (amount.isPresent()) {
                return amount;
            }
        }
        return Optional.empty();
    }

    /** Whether the trigger is its controller's beginning-of-upkeep trigger, with no condition or choice of its own. */
    private static boolean plainUpkeepTrigger(Trigger trigger) {
        return trigger.getMode() == TriggerType.Phase && "Upkeep".equals(trigger.getParam("Phase"))
                && "You".equals(trigger.getParam("ValidPlayer"))
                && Stream.of("IsPresent", "CheckSVar", "OptionalDecider").noneMatch(trigger::hasParam);
    }

    private static List<Map<String, String>> chain(Card host, Map<String, String> root) {
        List<Map<String, String>> found = new ArrayList<>();
        for (Map<String, String> segment = root; !segment.isEmpty() && found.size() < 8;
                segment = Scripts.script(host, segment.get("SubAbility"))) {
            found.add(segment);
        }
        return found;
    }

    private static Optional<Integer> amountOf(Card host, List<Map<String, String>> chain) {
        Optional<Integer> amount = Optional.empty();
        for (Map<String, String> segment : chain) {
            String api = Scripts.api(segment);
            if (!plainApis().contains(api) || segment.keySet().stream().anyMatch(name ->
                    name.startsWith("Condition") || name.startsWith("Unless") || name.equals("Optional"))) {
                return Optional.empty();
            }
            if (ApiType.DamageAll.name().equals(api) && "Player".equals(segment.get("ValidPlayers"))) {
                amount = Scripts.number(segment.get("NumDmg"));
            } else if (ApiType.LoseLife.name().equals(api) && "Player".equals(segment.get("Defined"))) {
                amount = Scripts.number(segment.get("LifeAmount"));
            }
        }
        return amount;
    }

    /** The edit: each player of the fork above the trigger's amount at that amount. */
    private static BoardEdit inReach(Binding fork) {
        int amount = lifeTakenFromEach(fork.card("permanent")).orElseThrow(
                () -> new IllegalStateException("the fork's permanent takes no life from each player"));
        return new EachPlayerAtMost(List.of(fork.player("player"), fork.player("opponent")), amount);
    }

    /**
     * {@code life} on each of {@code players}: one above {@code life} is set
     * to it ({@link LifeOperation}), one at or below it keeps its own.
     */
    record EachPlayerAtMost(List<Player> players, int life) implements BoardEdit {

        @Override
        public ForkOperation operation() {
            return ForkOperation.LIFE;
        }

        @Override
        public void apply(Game fork) {
            for (Player player : players) {
                if (player.getLife() > life) {
                    new LifeOperation(player, life).apply(fork);
                }
            }
        }
    }
}
