package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.124j, "partner with [name]" searches its player's library for the named
 * card as the permanent enters, on scenario {@code 702.124j-1}: Pir,
 * Imaginative Rascal enters and A puts Toothy, Imaginary Friend from its
 * library into its hand.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with partner with it has the lands for, and its
 * library holds exactly one card of the partner's name. It binds that card.
 *
 * <p>Variant {@code named_card_absent} ({@code move_card}): a Forest is in
 * that card's place, so the search finds nothing.
 */
public final class PartnerWithSearch {

    public static final String RULE = "702.124j";
    static final String SCENARIO = "702.124j-1";
    static final String NAMED_CARD_ABSENT = "named_card_absent";
    static final String FOREST = "Forest";

    private PartnerWithSearch() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PartnerWithSearch::bind, NAMED_CARD_ABSENT,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), FOREST)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card card : active.getCardsIn(ZoneType.Hand)) {
            if (!Boards.affordable(active, card)) {
                continue;
            }
            for (KeywordInterface instance : card.getKeywords(Keyword.PARTNER_WITH)) {
                if (!(instance instanceof KeywordWithType partner) || partner.getValidType() == null) {
                    continue;
                }
                List<Card> named = active.getCardsIn(ZoneType.Library).stream()
                        .filter(other -> other.getName().equals(partner.getValidType()))
                        .toList();
                if (named.size() == 1) {
                    return Optional.of(Binding.empty().with("card", named.get(0)));
                }
            }
        }
        return Optional.empty();
    }
}
