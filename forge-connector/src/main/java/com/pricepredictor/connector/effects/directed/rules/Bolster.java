package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Optional;

/**
 * 701.39a, bolster N puts N +1/+1 counters on the creature its controller
 * controls with the least toughness, on scenario {@code 701.39a-2}: A's
 * Cached Defenses bolsters A's Grizzly Bears and not B's Savannah Lions.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls exactly one creature, and it holds a card it has the
 * lands for whose spell bolsters (Cached Defenses). It binds that creature.
 *
 * <p>Variant {@code no_creature} ({@code remove_permanent}): the creature is
 * not on the battlefield, so the bolster finds no creature of its controller's
 * and puts no counter anywhere.
 */
public final class Bolster {

    public static final String RULE = "701.39a";
    static final String SCENARIO = "701.39a-2";
    static final String NO_CREATURE = "no_creature";

    private Bolster() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Bolster::bind, NO_CREATURE,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("creature"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> creatures = List.copyOf(active.getCreaturesInPlay());
        if (creatures.size() != 1 || Boards.castableInHand(active, Bolster::bolsters).isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with("creature", creatures.get(0)));
    }

    /** Whether the spell bolsters as its first act: a {@code PutCounter} root with {@code Bolster$}. */
    static boolean bolsters(SpellAbility spell) {
        return spell.getApi() == ApiType.PutCounter && spell.hasParam("Bolster");
    }
}
