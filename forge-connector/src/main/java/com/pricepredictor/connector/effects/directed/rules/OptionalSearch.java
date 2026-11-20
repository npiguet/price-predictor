package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Comparator;
import java.util.Optional;

/**
 * 118.12b, an "if you do" after an optional search asks whether the player
 * chose to search, not what the search found, inverted (FR-014a) on scenario
 * {@code 118.12b-2}: A's upkeep begins with no Elfhame Sanctuary; with one
 * on A's battlefield, its upkeep trigger lets A search for a basic land, and
 * A skips its draw step because it searched.
 *
 * <p>The base board, read at a boundary: the upkeep begins and the active
 * player's library holds a basic land card, the card Elfhame Sanctuary's
 * search finds: an ordinary board. The precondition does not wait for
 * Elfhame Sanctuary itself, which Forge's deck builder leaves out of every
 * AI deck ({@code AI:RemoveDeck:All}), so a live game almost never holds
 * it. It binds the active player.
 *
 * <p>Variant {@code sanctuary_added} ({@code add_permanent}): Elfhame
 * Sanctuary is on the player's battlefield since before this turn, so its
 * upkeep trigger triggers as the step's triggers are put on the stack (a
 * copy taken at the step's phase event sits ahead of them). The fork answers
 * its "you may search" yes (FR-010b), which Forge's AI, playing a card no AI
 * deck holds, has no logic of its own for; the search, the shuffle and the
 * skipped draw step follow: the rule's {@code fires} side, which the counts
 * lack. The Sanctuary is a card the edit makes, bound as {@code sanctuary} on
 * the edited copy ({@link Application#making}).
 */
public final class OptionalSearch {

    public static final String RULE = "118.12b";
    static final String SCENARIO = "118.12b-2";
    static final String SANCTUARY_ADDED = "sanctuary_added";
    static final String SANCTUARY = "Elfhame Sanctuary";

    private static final Role<Player> PLAYER = Role.player("player");
    private static final Role<Card> SANCTUARY_ROLE = Role.card("sanctuary");

    private OptionalSearch() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, OptionalSearch::bind, SANCTUARY_ADDED,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(SANCTUARY, fork.player(PLAYER), false))
                        .making(SANCTUARY_ROLE, OptionalSearch::added)
                        .forcing(ForcedPlay.answer(PLAYER, SANCTUARY_ROLE, true)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        // Elfhame Sanctuary's search finds a basic land card (ChangeType$ Land.Basic).
        boolean findsOne = active.getCardsIn(ZoneType.Library).stream().anyMatch(card -> card.getType().isBasicLand());
        return findsOne ? Optional.of(Binding.empty().with(PLAYER, active)) : Optional.empty();
    }

    /** The Elfhame Sanctuary the edit put on the player's battlefield: the newest card of that name there. */
    static Card added(Binding fork, Game game) {
        return fork.player(PLAYER).getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(SANCTUARY))
                .max(Comparator.comparingInt(Card::getId))
                .orElseThrow(() -> new IllegalStateException("the edit put no " + SANCTUARY + " in play"));
    }
}
