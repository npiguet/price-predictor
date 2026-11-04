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
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 730.2g, a face-down merged permanent containing an instant or sorcery card
 * cannot be turned face up, on scenario {@code 730.2g-1}: A manifests Shock
 * from the top of its library, mutates Cloudpiercer onto it, Ixidron turns it
 * face down, and B's Break Open only reveals it.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card that manifests or cloaks the top card of its
 * library, which it has the lands for, and the top card of its library is an
 * instant or sorcery. It binds that card. The scenario's merge and face-down
 * turn follow from decisions no board read can foresee; the board read waits
 * for the face-down permanent's card to be chosen, which the variant changes.
 *
 * <p>Variant {@code creature-card} ({@code move_card}): Grizzly Bears is in
 * that card's place, so the face-down permanent holds a creature card and can
 * be turned face up.
 */
public final class UnturnableFaceDown {

    public static final String RULE = "730.2g";
    static final String SCENARIO = "730.2g-1";
    static final String CREATURE_CARD = "creature-card";
    static final String BEARS = "Grizzly Bears";

    private UnturnableFaceDown() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, UnturnableFaceDown::bind, CREATURE_CARD,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), BEARS)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (!Boards.holdsActing(active, ApiType.Manifest) && !Boards.holdsActing(active, ApiType.Cloak)) {
            return Optional.empty();
        }
        Card top = active.getCardsIn(ZoneType.Library).isEmpty()
                ? null : active.getCardsIn(ZoneType.Library).get(0);
        return top != null && (top.isInstant() || top.isSorcery())
                ? Optional.of(Binding.empty().with("card", top)) : Optional.empty();
    }
}
