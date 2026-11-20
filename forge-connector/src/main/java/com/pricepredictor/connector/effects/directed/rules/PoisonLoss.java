package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.game.Game;
import forge.game.zone.ZoneType;
import forge.game.trigger.TriggerType;
import forge.game.player.Player;
import forge.game.card.Card;
import forge.game.ability.ApiType;
import forge.game.card.CounterEnumType;

import java.util.Optional;

/**
 * 104.3d, a player with ten or more poison counters loses the game, on
 * scenario {@code 104.3d-1}: B has 8 poison counters when A casts Ichor Rats,
 * whose enters trigger gives each player one more, and B stays in the game
 * at 9; at 9 B would lose.
 *
 * <p>The base board, read at a boundary: a main phase begins, the active
 * player holds a card it has the lands for that gives each player a poison
 * counter with no choice (Ichor Rats: {@link #poisonsEachPlayer}), and an
 * opponent of it has fewer than nine poison counters, however few: the
 * common side, where that counter loses no one the game (the scenario's
 * eight is one such board). It binds that opponent, the active player and
 * the card.
 *
 * <p>Variant {@code nine} ({@code counters}): the opponent has nine poison
 * counters, whatever it had, so the turn's next poison counter given to it
 * ends the game.
 * The fork forces the card's cast (FR-010b), whose resolution gives the
 * tenth counter in the record its declaration reads; an attack's poison
 * would land in a combat record, which it does not read.
 */
public final class PoisonLoss {

    public static final String RULE = "104.3d";
    static final String SCENARIO = "104.3d-1";
    static final String NINE = "nine";
    /** The poison counters one short of the ten that lose the game. */
    static final int ONE_SHORT = 9;

    private PoisonLoss() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PoisonLoss::bind, NINE,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.player("player"), CounterEnumType.POISON, ONE_SHORT))
                        .forcing(ForcedPlay.cast("caster", "card")));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> source = active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> poisonsEachPlayer(card) && Mana.affordable(active, card))
                .findFirst();
        if (source.isEmpty()) {
            return Optional.empty();
        }
        return active.getOpponents().stream()
                .filter(player -> player.getPoisonCounters() < ONE_SHORT)
                .findFirst()
                .map(player -> Binding.empty().with("player", player)
                        .with("caster", active).with("card", source.get()));
    }

    /**
     * Whether the card gives each player a poison counter with no choice: its
     * spell, or the ability its enters trigger executes, is a {@code Poison}
     * of {@code Defined$ Player} (Ichor Rats).
     */
    static boolean poisonsEachPlayer(Card card) {
        boolean spell = card.getSpells().stream()
                .anyMatch(ability -> ability.getApi() == ApiType.Poison && "Player".equals(ability.getParam("Defined")));
        boolean entering = card.getTriggers().stream()
                .filter(Triggers::entersItself)
                .map(Scripts::executed)
                .anyMatch(script -> ApiType.Poison.name().equals(Scripts.api(script))
                        && "Player".equals(script.get("Defined")));
        return spell || entering;
    }
}
