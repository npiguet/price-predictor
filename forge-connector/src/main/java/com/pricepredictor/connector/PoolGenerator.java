package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardRarity;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Generates sealed pools using Forge's internal booster generation API.
 *
 * <p>Each pool consists of 6 boosters from the specified set. Basic lands are
 * filtered out. {@link ForgeEnvironmentInitializer#initialize()} must have been
 * called before invoking this class.
 *
 * <p>A pool can be <b>depleted</b> of a set of card names. That is how the
 * effect model's training corpus is collected: a training game must contain no
 * held-out card, so that no collected game has to be discarded from training.
 * The excluded card is redrawn within its own rarity rather than dropped — a
 * pool short of cards would build a different deck, and the depleted corpus is
 * meant to differ from the full-strength one only in which cards exist.
 */
public class PoolGenerator {

    private static final int BOOSTERS_PER_POOL = 6;

    /**
     * Replacement boosters opened before an unreplaceable card is dropped.
     * A rarity can genuinely run out — a set whose every mythic is excluded has
     * no mythic to offer — and looping until one appears would never return.
     */
    private static final int MAX_REPLACEMENT_BOOSTERS = 8;

    /**
     * Generate {@code poolCount} sealed pools for the given set code.
     *
     * @param setCode   MTG set code (e.g. "RVR", "MH3"). Must not be null.
     * @param poolCount Number of pools to generate.
     * @return List of pools; each pool is a list of non-basic-land card names.
     * @throws IllegalArgumentException if setCode is null or not found in Forge's booster database.
     */
    public List<List<String>> generate(String setCode, int poolCount) {
        return generate(setCode, poolCount, Collections.emptySet());
    }

    /**
     * Generate {@code poolCount} sealed pools with {@code excluded} depleted out.
     *
     * @param setCode   MTG set code (e.g. "RVR", "MH3"). Must not be null.
     * @param poolCount Number of pools to generate.
     * @param excluded  Card names no pool may contain; empty for an ordinary pool.
     * @return List of pools; each pool is a list of non-basic-land card names.
     * @throws IllegalArgumentException if setCode is null or not found in Forge's booster database.
     */
    public List<List<String>> generate(String setCode, int poolCount, Set<String> excluded) {
        SealedTemplate boosterTemplate = boosterTemplate(setCode);

        List<List<String>> pools = new ArrayList<>(poolCount);
        for (int p = 0; p < poolCount; p++) {
            List<String> poolCards = new ArrayList<>();
            for (PaperCard card : openSinglePool(boosterTemplate, excluded)) {
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
        return generatePool(setCode, Collections.emptySet());
    }

    /**
     * Generate a single pool as {@link PaperCard}s, depleted of {@code excluded}.
     *
     * @param setCode  MTG set code (must not be null)
     * @param excluded Card names the pool may not contain; empty for an ordinary pool.
     * @return Non-basic-land cards from 6 boosters of the given set.
     */
    public List<PaperCard> generatePool(String setCode, Set<String> excluded) {
        return openSinglePool(boosterTemplate(setCode), excluded);
    }

    private SealedTemplate boosterTemplate(String setCode) {
        if (setCode == null) {
            throw new IllegalArgumentException("Set code must not be null");
        }
        SealedTemplate boosterTemplate = StaticData.instance().getBoosters().get(setCode);
        if (boosterTemplate == null) {
            throw new IllegalArgumentException("Unknown or unsupported set code: " + setCode);
        }
        return boosterTemplate;
    }

    private List<PaperCard> openSinglePool(SealedTemplate boosterTemplate, Set<String> excluded) {
        List<PaperCard> pool = new ArrayList<>();
        Map<CardRarity, Deque<PaperCard>> spares = new EnumMap<>(CardRarity.class);
        int replacementBoosters = 0;

        for (int b = 0; b < BOOSTERS_PER_POOL; b++) {
            for (PaperCard card : new UnOpenedProduct(boosterTemplate).get()) {
                if (card.getRules().getMainPart().getType().isBasicLand()) {
                    continue;
                }
                if (!excluded.contains(card.getName())) {
                    pool.add(card);
                    continue;
                }
                PaperCard replacement = null;
                while (replacement == null && replacementBoosters < MAX_REPLACEMENT_BOOSTERS) {
                    replacement = takeSpare(spares, card.getRarity());
                    if (replacement == null) {
                        stockSpares(spares, boosterTemplate, excluded);
                        replacementBoosters++;
                    }
                }
                if (replacement != null) {
                    pool.add(replacement);
                }
            }
        }
        return pool;
    }

    private PaperCard takeSpare(Map<CardRarity, Deque<PaperCard>> spares, CardRarity rarity) {
        Deque<PaperCard> queue = spares.get(rarity);
        return queue == null || queue.isEmpty() ? null : queue.poll();
    }

    /** Open one more booster and keep its usable cards, filed by rarity. */
    private void stockSpares(
            Map<CardRarity, Deque<PaperCard>> spares,
            SealedTemplate boosterTemplate,
            Set<String> excluded) {
        for (PaperCard card : new UnOpenedProduct(boosterTemplate).get()) {
            if (card.getRules().getMainPart().getType().isBasicLand()
                    || excluded.contains(card.getName())) {
                continue;
            }
            spares.computeIfAbsent(card.getRarity(), r -> new ArrayDeque<>()).add(card);
        }
    }
}
