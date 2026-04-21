package com.pricepredictor.connector;

import forge.StaticData;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;

import java.util.ArrayList;
import java.util.List;

/**
 * Generates sealed pools using Forge's internal booster generation API.
 *
 * <p>Each pool consists of 6 boosters from the specified set. Basic lands are
 * filtered out. {@link ForgeEnvironmentInitializer#initialize()} must have been
 * called before invoking this class.
 */
public class PoolGenerator {

    private static final int BOOSTERS_PER_POOL = 6;

    /**
     * Generate {@code poolCount} sealed pools for the given set code.
     *
     * @param setCode   MTG set code (e.g. "RVR", "MH3"). Must not be null.
     * @param poolCount Number of pools to generate.
     * @return List of pools; each pool is a list of non-basic-land card names.
     * @throws IllegalArgumentException if setCode is null or not found in Forge's booster database.
     */
    public List<List<String>> generate(String setCode, int poolCount) {
        if (setCode == null) {
            throw new IllegalArgumentException("Set code must not be null");
        }

        SealedTemplate boosterTemplate = StaticData.instance().getBoosters().get(setCode);
        if (boosterTemplate == null) {
            throw new IllegalArgumentException("Unknown or unsupported set code: " + setCode);
        }

        List<List<String>> pools = new ArrayList<>(poolCount);

        for (int p = 0; p < poolCount; p++) {
            List<String> poolCards = new ArrayList<>();
            for (PaperCard card : openSinglePool(boosterTemplate)) {
                poolCards.add(card.getName());
            }
            pools.add(poolCards);
        }

        return pools;
    }

    /**
     * Generate a single pool as {@link PaperCard}s (basic lands excluded).
     *
     * <p>Used by self-play match generation to feed deck B's pool directly into
     * {@link DeckBuilder#buildDeck(List)}, which requires {@code PaperCard}s.
     *
     * @param setCode MTG set code (must not be null)
     * @return Non-basic-land cards from 6 boosters of the given set.
     */
    public List<PaperCard> generatePool(String setCode) {
        if (setCode == null) {
            throw new IllegalArgumentException("Set code must not be null");
        }

        SealedTemplate boosterTemplate = StaticData.instance().getBoosters().get(setCode);
        if (boosterTemplate == null) {
            throw new IllegalArgumentException("Unknown or unsupported set code: " + setCode);
        }

        return openSinglePool(boosterTemplate);
    }

    private List<PaperCard> openSinglePool(SealedTemplate boosterTemplate) {
        List<PaperCard> pool = new ArrayList<>();
        for (int b = 0; b < BOOSTERS_PER_POOL; b++) {
            List<PaperCard> boosterCards = new UnOpenedProduct(boosterTemplate).get();
            for (PaperCard card : boosterCards) {
                if (!card.getRules().getMainPart().getType().isBasicLand()) {
                    pool.add(card);
                }
            }
        }
        return pool;
    }
}
