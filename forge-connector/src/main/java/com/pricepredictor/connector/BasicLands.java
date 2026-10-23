package com.pricepredictor.connector;

import forge.item.PaperCard;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Which card names are basic lands, by Forge's own type rule — the one
 * {@link PlayedCardCollector}, {@link PoolGenerator} and {@link DeckBuilder}
 * apply to cards — so a corpus writer and the game that fed it agree.
 *
 * <p>Needs an initialized Forge; a name Forge does not know is not a basic land.
 */
final class BasicLands {

    private static final Map<String, Boolean> BY_NAME = new ConcurrentHashMap<>();

    private BasicLands() {
    }

    /** Whether the card Forge knows by this name is a basic land; false for a name it does not know. */
    static boolean isBasic(String cardName) {
        return BY_NAME.computeIfAbsent(cardName,
                name -> DeckMaterializer.lookUp(name).map(BasicLands::isBasic).orElse(false));
    }

    /** Whether the card's main face carries the basic land type, by Forge's type rule. */
    static boolean isBasic(PaperCard card) {
        return card.getRules().getMainPart().getType().isBasicLand();
    }
}
