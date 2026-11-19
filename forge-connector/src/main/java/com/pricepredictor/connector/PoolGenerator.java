package com.pricepredictor.connector;

import forge.card.CardRarity;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

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
     * Generate {@code poolCount} sealed pools with {@code excluded} depleted out.
     *
     * @param setCode   MTG set code (e.g. "RVR", "MH3"). Must not be null.
     * @param poolCount Number of pools to generate.
     * @param excluded  Cards no pool may contain; {@link HoldoutCards#NONE} for an ordinary pool.
     * @return List of pools; each pool is a list of non-basic-land card names.
     * @throws IllegalArgumentException if setCode is null or names no set with a draft booster.
     */
    public List<List<String>> generate(String setCode, int poolCount, HoldoutCards excluded) {
        SealedTemplate boosterTemplate = boosterTemplate(setCode);

        List<List<String>> pools = new ArrayList<>(poolCount);
        int shortPools = 0;
        for (int p = 0; p < poolCount; p++) {
            List<String> poolCards = new ArrayList<>();
            OpenedPool opened = openPool(boosterTemplate, excluded);
            for (PaperCard card : opened.cards()) {
                poolCards.add(card.getName());
            }
            pools.add(poolCards);
            if (opened.dropped() > 0) {
                shortPools++;
            }
        }
        if (shortPools > 0) {
            System.err.println(shortPools + " of " + poolCount
                    + " pools are short of cards: a held-out card found no replacement");
        }
        return pools;
    }

    private SealedTemplate boosterTemplate(String setCode) {
        if (setCode == null) {
            throw new IllegalArgumentException("Set code must not be null");
        }
        return BoosterTemplates.forSet(setCode).orElseThrow(
                () -> new IllegalArgumentException("Unknown or unsupported set code: " + setCode));
    }

    /**
     * Open one pool from an already-resolved template, depleted of {@code excluded}.
     *
     * <p>Package-visible because {@link MatchGenerator} resolves its own
     * template, with its own error for a set it cannot open, and must share
     * this redraw rule rather than keeping a second copy. Its pools are what
     * every collected effect record is played from.
     */
    List<PaperCard> openSinglePool(SealedTemplate boosterTemplate, HoldoutCards excluded) {
        return openPool(boosterTemplate, excluded).cards();
    }

    /** A pool and how many of its slots found no replacement for a held-out card. */
    record OpenedPool(List<PaperCard> cards, int dropped) {
    }

    /** {@link #openSinglePool}, with the count of slots left empty. */
    OpenedPool openPool(SealedTemplate boosterTemplate, HoldoutCards excluded) {
        int dropped = 0;
        List<PaperCard> pool = new ArrayList<>();
        Map<CardRarity, Deque<PaperCard>> spares = new EnumMap<>(CardRarity.class);
        int replacementBoosters = 0;

        for (int b = 0; b < BOOSTERS_PER_POOL; b++) {
            for (PaperCard card : new UnOpenedProduct(boosterTemplate).get()) {
                if (BasicLands.isBasic(card)) {
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
                } else {
                    dropped++;
                }
            }
        }
        return new OpenedPool(pool, dropped);
    }

    private PaperCard takeSpare(Map<CardRarity, Deque<PaperCard>> spares, CardRarity rarity) {
        Deque<PaperCard> queue = spares.get(rarity);
        return queue == null || queue.isEmpty() ? null : queue.poll();
    }

    /** Open one more booster and keep its usable cards, filed by rarity. */
    private void stockSpares(
            Map<CardRarity, Deque<PaperCard>> spares,
            SealedTemplate boosterTemplate,
            HoldoutCards excluded) {
        for (PaperCard card : new UnOpenedProduct(boosterTemplate).get()) {
            if (BasicLands.isBasic(card) || excluded.contains(card.getName())) {
                continue;
            }
            spares.computeIfAbsent(card.getRarity(), r -> new ArrayDeque<>()).add(card);
        }
    }
}
