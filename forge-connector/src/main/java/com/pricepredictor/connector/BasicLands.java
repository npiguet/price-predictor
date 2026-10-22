package com.pricepredictor.connector;

import forge.item.PaperCard;
import forge.model.FModel;

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

    static boolean isBasic(String cardName) {
        return BY_NAME.computeIfAbsent(cardName, BasicLands::lookUp);
    }

    private static boolean lookUp(String cardName) {
        PaperCard card = FModel.getMagicDb().getCommonCards().getCard(cardName);
        return card != null && card.getRules().getMainPart().getType().isBasicLand();
    }
}
