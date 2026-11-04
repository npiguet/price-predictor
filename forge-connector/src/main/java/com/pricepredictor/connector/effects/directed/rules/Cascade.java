package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.85a, cascade exiles cards from the top of the library until a nonland
 * card of lesser mana value, casts it without paying its mana cost and puts
 * the rest on the bottom, on scenario {@code 702.85a-1}: A's Bloodbraid Elf
 * (mana value 4) cascades past a Mountain into Shock.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with cascade it has the lands for, of mana value at
 * most 6, and its library holds exactly one nonland card of lesser mana value
 * than that card's. It binds that library card.
 *
 * <p>Variant {@code no_cheaper_card} ({@code move_card}): a Craw Wurm (mana
 * value 6) is in that card's place, so the cascade exiles the whole library
 * and casts nothing.
 */
public final class Cascade {

    public static final String RULE = "702.85a";
    static final String SCENARIO = "702.85a-1";
    static final String NO_CHEAPER_CARD = "no_cheaper_card";
    static final String WURM = "Craw Wurm";
    /** Craw Wurm's mana value: no cascade card of at most this much finds it. */
    private static final int WURM_VALUE = 6;

    private Cascade() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Cascade::bind, NO_CHEAPER_CARD,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), WURM)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card cascader : active.getCardsIn(ZoneType.Hand)) {
            int value = cascader.getCMC();
            if (!cascader.hasKeyword(Keyword.CASCADE) || value > WURM_VALUE || !Boards.affordable(active, cascader)) {
                continue;
            }
            List<Card> cheaper = active.getCardsIn(ZoneType.Library).stream()
                    .filter(card -> !card.isLand() && card.getCMC() < value)
                    .toList();
            if (cheaper.size() == 1) {
                return Optional.of(Binding.empty().with("card", cheaper.get(0)));
            }
        }
        return Optional.empty();
    }
}
