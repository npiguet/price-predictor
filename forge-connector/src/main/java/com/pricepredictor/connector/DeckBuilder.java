package com.pricepredictor.connector;

import forge.deck.Deck;
import forge.gamemodes.limited.SealedDeckBuilder;
import forge.item.PaperCard;
import forge.util.MyRandom;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Builds a 40-card sealed deck from a booster pool using one of four weighted methods.
 *
 * <p>Method weights: 1=40%, 2=30%, 3=20%, 4=10%.
 * <ul>
 *   <li>Method 1: Standard Forge sealed deck builder</li>
 *   <li>Method 2: Forge builder + 3 nonland card swaps + land rebalance</li>
 *   <li>Method 3: Forge builder + 8 nonland card swaps + land rebalance</li>
 *   <li>Method 4: 23 random nonland cards + land rebalance</li>
 * </ul>
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class DeckBuilder {

    private static final double[] METHOD_THRESHOLDS = {0.4, 0.7, 0.9, 1.0};
    private final Random random;

    public DeckBuilder() {
        this.random = MyRandom.getRandom();
    }

    DeckBuilder(Random random) {
        this.random = random;
    }

    /**
     * Build a deck from the given pool using a randomly selected weighted method.
     *
     * @param pool booster pool cards (no basic lands)
     * @return 40-card Deck
     */
    public Deck buildDeck(List<PaperCard> pool) {
        int method = selectMethod();
        return switch (method) {
            case 1 -> buildStandard(pool);
            case 2 -> buildWithSwaps(pool, 3);
            case 3 -> buildWithSwaps(pool, 8);
            case 4 -> buildRandom(pool);
            default -> throw new IllegalStateException("Unexpected method: " + method);
        };
    }

    int selectMethod() {
        double roll = random.nextDouble();
        for (int i = 0; i < METHOD_THRESHOLDS.length; i++) {
            if (roll < METHOD_THRESHOLDS[i]) {
                return i + 1;
            }
        }
        return 4;
    }

    Deck buildStandard(List<PaperCard> pool) {
        return new SealedDeckBuilder(new ArrayList<>(pool)).buildDeck();
    }

    Deck buildWithSwaps(List<PaperCard> pool, int swapCount) {
        // 1. Build standard deck
        List<PaperCard> poolCopy = new ArrayList<>(pool);
        Deck standardDeck = new SealedDeckBuilder(poolCopy).buildDeck();
        // Note: poolCopy may be modified by SealedDeckBuilder constructor

        // 2. Separate deck into nonland cards and basic lands
        List<PaperCard> nonlandInDeck = new ArrayList<>();
        for (PaperCard card : standardDeck.getMain().toFlatList()) {
            if (!card.getRules().getMainPart().getType().isBasicLand()) {
                nonlandInDeck.add(card);
            }
        }

        // 3. Compute remaining pool (pool cards not selected for deck)
        // Use a mutable copy and remove one instance at a time for correct duplicate handling
        List<PaperCard> remainingPool = new ArrayList<>(pool);
        for (PaperCard deckCard : standardDeck.getMain().toFlatList()) {
            remainingPool.remove(deckCard);
        }
        // Filter remaining pool to nonland cards only (pool already has no basic lands,
        // so this just ensures we don't accidentally include lands from the deck sideboard)
        List<PaperCard> nonlandRemaining = new ArrayList<>();
        for (PaperCard card : remainingPool) {
            if (!card.getRules().getMainPart().getType().isBasicLand()) {
                nonlandRemaining.add(card);
            }
        }

        // 4. Perform swaps (capped at available cards in each list)
        int actualSwaps = Math.min(swapCount, Math.min(nonlandInDeck.size(), nonlandRemaining.size()));

        List<PaperCard> chosenCards = new ArrayList<>(nonlandInDeck);
        for (int i = 0; i < actualSwaps; i++) {
            int removeIdx = random.nextInt(chosenCards.size());
            int addIdx = random.nextInt(nonlandRemaining.size());
            chosenCards.remove(removeIdx);
            chosenCards.add(nonlandRemaining.remove(addIdx));
        }

        // 5. Rebalance lands
        return rebalanceLands(chosenCards);
    }

    Deck buildRandom(List<PaperCard> pool) {
        // Pick 23 random nonland cards from pool (pool has no basic lands)
        List<PaperCard> nonlandPool = new ArrayList<>();
        for (PaperCard card : pool) {
            if (!card.getRules().getMainPart().getType().isBasicLand()) {
                nonlandPool.add(card);
            }
        }

        int pickCount = Math.min(23, nonlandPool.size());
        List<PaperCard> chosen = new ArrayList<>(pickCount);
        List<PaperCard> mutablePool = new ArrayList<>(nonlandPool);
        for (int i = 0; i < pickCount; i++) {
            int idx = random.nextInt(mutablePool.size());
            chosen.add(mutablePool.remove(idx));
        }

        return rebalanceLands(chosen);
    }

    Deck rebalanceLands(List<PaperCard> nonlandCards) {
        // Pass nonland cards to SealedDeckBuilder; it will choose colors based on the
        // card composition and add basic lands to fill to 40 cards.
        return new SealedDeckBuilder(new ArrayList<>(nonlandCards)).buildDeck();
    }
}
