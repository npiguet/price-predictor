package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
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
    private static final Role<Card> PERMANENT = Role.card("permanent");
    private static final Role<Player> PLAYER = Role.player("player");
    private static final Role<Player> OPPONENT = Role.player("opponent");

    private SimultaneousLoss() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SimultaneousLoss::bind, new DirectedRule.Variant(BOTH_AT_ONE,
                new Application(ForkOperation.LIFE, SimultaneousLoss::inReach)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP) || !twoPlayersLosingAtZeroLife(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Player opponent = live.getPlayers().stream().filter(player -> player != active).findFirst().orElseThrow();
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(permanent -> lifeTakenFromEach(permanent).filter(amount -> someoneAbove(live, amount)).isPresent())
                .findFirst()
                .map(permanent -> Binding.empty().with(PERMANENT, permanent).with(PLAYER, active)
                        .with(OPPONENT, opponent));
    }

    private static boolean someoneAbove(Game live, int amount) {
        return live.getPlayers().stream().anyMatch(player -> player.getLife() > amount);
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
        return permanent.getTriggers().stream()
                .filter(SimultaneousLoss::plainUpkeepTrigger)
                .map(trigger -> amountOf(permanent, chain(permanent, Scripts.executed(trigger))))
                .flatMap(Optional::stream)
                .findFirst();
    }

    /** Whether the trigger is its controller's beginning-of-upkeep trigger, with no condition or choice of its own. */
    private static boolean plainUpkeepTrigger(Trigger trigger) {
        return Triggers.atUpkeep(trigger) && "You".equals(trigger.getParam("ValidPlayer"))
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
            if (!plainSegment(segment)) {
                return Optional.empty();
            }
            if (takesFromEach(segment)) {
                amount = Scripts.number(lossText(segment));
            }
        }
        return amount;
    }

    /** Whether the segment has one of the {@link #plainApis} and no condition, unless part or optional. */
    private static boolean plainSegment(Map<String, String> segment) {
        return plainApis().contains(Scripts.api(segment)) && segment.keySet().stream().noneMatch(name ->
                name.startsWith("Condition") || name.startsWith("Unless") || name.equals("Optional"));
    }

    /** Whether the segment is the chain's {@code DamageAll} or {@code LoseLife} on every player. */
    private static boolean takesFromEach(Map<String, String> segment) {
        String api = Scripts.api(segment);
        return ApiType.DamageAll.name().equals(api) && "Player".equals(segment.get("ValidPlayers"))
                || ApiType.LoseLife.name().equals(api) && "Player".equals(segment.get("Defined"));
    }

    /** The text of the number such a segment makes every player lose. */
    private static String lossText(Map<String, String> segment) {
        return segment.get(ApiType.DamageAll.name().equals(Scripts.api(segment)) ? "NumDmg" : "LifeAmount");
    }

    /** The edit: each player of the fork above the trigger's amount at that amount. */
    private static BoardEdit inReach(Binding fork) {
        int amount = lifeTakenFromEach(fork.card(PERMANENT)).orElseThrow(
                () -> new IllegalStateException("the fork's permanent takes no life from each player"));
        return new EachPlayerAtMost(List.of(fork.player(PLAYER), fork.player(OPPONENT)), amount);
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
