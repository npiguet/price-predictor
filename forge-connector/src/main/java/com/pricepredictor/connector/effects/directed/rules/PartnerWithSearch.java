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
 * card as the permanent enters, on scenario {@code 702.124j-1}: A's library
 * holds three Forests and no Toothy, Imaginary Friend when Pir, Imaginative
 * Rascal enters, and the search finds nothing; with Toothy there it would
 * find it.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with partner with it has the lands for, and its
 * library holds no card of the partner's name, and at least one card. It
 * binds that card with partner with and the library's top card. The live
 * game, where the search finds nothing, is the common side; the fork adds the
 * rare fact.
 *
 * <p>Variant {@code named_card_in_library} ({@code move_card}): a card of the
 * partner's name is in the top card's place, so the search finds it and
 * moves it to the hand, the move 702.124j-1's declaration reads. Whether the
 * card with partner with is cast is its seat's own decision.
 */
public final class PartnerWithSearch {

    public static final String RULE = "702.124j";
    static final String SCENARIO = "702.124j-1";
    static final String NAMED_CARD_IN_LIBRARY = "named_card_in_library";

    private PartnerWithSearch() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PartnerWithSearch::bind, NAMED_CARD_IN_LIBRARY,
                new Application(ForkOperation.MOVE_CARD, fork -> MoveCardOperation.replace(fork.card("card"),
                        partnerName(fork.card("partner with")).orElseThrow(
                                () -> new IllegalStateException("the fork's card names no partner")))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> library = List.copyOf(active.getCardsIn(ZoneType.Library));
        if (library.isEmpty()) {
            return Optional.empty();
        }
        for (Card card : active.getCardsIn(ZoneType.Hand)) {
            if (!Boards.affordable(active, card)) {
                continue;
            }
            Optional<String> partner = partnerName(card);
            if (partner.isPresent() && library.stream().noneMatch(other -> other.getName().equals(partner.get()))) {
                return Optional.of(Binding.empty().with("partner with", card).with("card", library.get(0)));
            }
        }
        return Optional.empty();
    }

    /** The name the card's partner with names, if it has the keyword. */
    static Optional<String> partnerName(Card card) {
        for (KeywordInterface instance : card.getKeywords(Keyword.PARTNER_WITH)) {
            if (instance instanceof KeywordWithType partner && partner.getValidType() != null) {
                return Optional.of(partner.getValidType());
            }
        }
        return Optional.empty();
    }
}
