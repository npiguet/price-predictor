package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.Optional;

/**
 * 702.105a, dethrone triggers when its creature attacks the player with the
 * most life or tied for the most life, on scenario {@code 702.105a-1}: A (15
 * life) attacks B (20 life) with Enraged Revolutionary, and dethrone puts a
 * +1/+1 counter on it.
 *
 * <p>The base board, read at a boundary: the declare-attackers step begins,
 * before attackers are declared, the active player controls an untapped
 * creature with dethrone that can attack this turn, and an opponent has the
 * most life or is tied for it, while every other player has 2 life or more.
 * It binds the creature and that opponent.
 *
 * <p>Variant {@code less-life} ({@code life}): the opponent has one life less
 * than the most any other player has, so attacking it does not trigger
 * dethrone. The other players having 2 or more keeps the opponent alive.
 */
public final class Dethrone {

    public static final String RULE = "702.105a";
    static final String SCENARIO = "702.105a-1";
    static final String LESS_LIFE = "less-life";

    private Dethrone() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Dethrone::bind, LESS_LIFE,
                new Application(ForkOperation.LIFE, Dethrone::lessLife));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.COMBAT_DECLARE_ATTACKERS)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> attacker = Boards.untappedCreatures(active).stream()
                .filter(creature -> creature.hasKeyword(Keyword.DETHRONE) && !creature.isSick())
                .findFirst();
        if (attacker.isEmpty()) {
            return Optional.empty();
        }
        for (Player opponent : active.getOpponents()) {
            int most = Boards.mostLifeBesides(opponent);
            if (opponent.getLife() >= most && most >= 2) {
                return Optional.of(Binding.empty()
                        .with("attacker", attacker.get())
                        .with("opponent", opponent));
            }
        }
        return Optional.empty();
    }

    private static BoardEdit lessLife(Binding fork) {
        Player opponent = fork.player("opponent");
        return new LifeOperation(opponent, Boards.mostLifeBesides(opponent) - 1);
    }
}
