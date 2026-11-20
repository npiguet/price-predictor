package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
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
 * begins, it controls no creature, and it holds a card it has the lands for
 * whose spell bolsters (Cached Defenses), which finds no creature to put its
 * counters on. It binds the active player and the card.
 *
 * <p>Variant {@code has_creature} ({@code add_permanent}): Grizzly Bears is on
 * the battlefield under the player's control, so the bolster puts its
 * counters on it: the side of the rule the counts lack. The fork forces the
 * card's cast (FR-010b), which the live game's AI may not make this turn.
 */
public final class Bolster {

    public static final String RULE = "701.39a";
    static final String SCENARIO = "701.39a-2";
    static final String HAS_CREATURE = "has_creature";
    static final String BEARS = Cards.BEARS;
    /** A spell of the card that bolsters ({@link #bolsters}): what the precondition finds and the fork casts. */
    static final ForcedPlay.Line BOLSTERING = ForcedPlay.Line.spell("bolster", Bolster::bolsters);

    private Bolster() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Bolster::bind, HAS_CREATURE,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(BEARS, fork.player("player"), true))
                        .forcing(ForcedPlay.cast("player", "card").playing(BOLSTERING)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> spells = Scan.castableInHand(active, BOLSTERING);
        if (!active.getCreaturesInPlay().isEmpty() || spells.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with("player", active).with("card", spells.get(0)));
    }

    /** Whether the spell bolsters as its first act: a {@code PutCounter} root with {@code Bolster$}. */
    static boolean bolsters(SpellAbility spell) {
        return spell.getApi() == ApiType.PutCounter && spell.hasParam("Bolster");
    }
}
