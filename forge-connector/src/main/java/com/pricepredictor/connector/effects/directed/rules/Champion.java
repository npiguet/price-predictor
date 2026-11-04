package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardLists;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * 607.2k, champion's return ability returns only the card its own exile
 * ability exiled, on scenario {@code 607.2k-1}: A casts Changeling Hero and
 * champions Grizzly Bears, which returns when B exiles the Hero.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with champion that it has the mana to cast, and it
 * controls exactly one other permanent of the type the card champions. It
 * binds that permanent and the card.
 *
 * <p>Variant {@code no_creature_to_champion} ({@code remove_permanent}): the
 * permanent is not on the battlefield, so the champion ability exiles nothing,
 * the card is sacrificed, and nothing ever returns.
 */
public final class Champion {

    public static final String RULE = "607.2k";
    static final String SCENARIO = "607.2k-1";
    static final String NO_CREATURE_TO_CHAMPION = "no_creature_to_champion";

    private Champion() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Champion::bind, NO_CREATURE_TO_CHAMPION,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("creature"))));
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
            for (KeywordInterface champion : card.getKeywords(Keyword.CHAMPION)) {
                List<Card> candidates = championable(champion, card, active);
                if (candidates.size() == 1) {
                    return Optional.of(Binding.empty().with("creature", candidates.get(0)).with("card", card));
                }
            }
        }
        return Optional.empty();
    }

    /**
     * The permanents {@code player} controls that the champion keyword of
     * {@code card} could exile, its types read as Forge's champion trigger
     * reads them ({@code Champion:Creature}, {@code Champion:Faerie}).
     */
    static List<Card> championable(KeywordInterface champion, Card card, Player player) {
        String[] parts = champion.getOriginal().split(":");
        if (parts.length < 2) {
            return List.of();
        }
        String[] valid = Arrays.stream(parts[1].split(",")).map(type -> type + ".YouCtrl+Other")
                .toArray(String[]::new);
        return CardLists.getValidCards(player.getCardsIn(ZoneType.Battlefield), valid, player, card, null);
    }
}
