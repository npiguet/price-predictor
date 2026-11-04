package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 121.4, a player who attempted to draw from an empty library loses at the
 * next state-based check, on scenario {@code 121.4-1}: A casts Divination
 * with one card in its library, draws it, fails the second draw and loses.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, its library holds exactly one card, and it holds a spell it has the
 * lands to cast that makes it draw cards (Divination). It binds the library's
 * card and the spell.
 *
 * <p>Variant {@code empty} ({@code move_card}): the library is empty, the
 * card moved to its owner's graveyard (a move is the operation's only way to
 * empty a library), so every draw the spell makes fails and the player loses.
 */
public final class DrawFromEmptyLibrary {

    public static final String RULE = "121.4";
    static final String SCENARIO = "121.4-1";
    static final String EMPTY = "empty";

    private DrawFromEmptyLibrary() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, DrawFromEmptyLibrary::bind, EMPTY,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card("card"), ZoneType.Graveyard, null)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> library = List.copyOf(active.getCardsIn(ZoneType.Library));
        if (library.size() != 1) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> Boards.affordable(active, card) && drawsForCaster(card))
                .findFirst()
                .map(spell -> Binding.empty().with("card", library.get(0)).with("spell", spell));
    }

    /** Whether the card is a spell that makes its caster draw cards. */
    static boolean drawsForCaster(Card card) {
        for (SpellAbility spell : card.getSpells()) {
            if (spell.getApi() == ApiType.Draw && !spell.usesTargeting()
                    && "You".equals(spell.getParamOrDefault("Defined", "You"))) {
                return true;
            }
        }
        return false;
    }
}
