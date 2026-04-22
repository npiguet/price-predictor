package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.card.MagicColor;
import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.deck.CardPool;
import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.gamemodes.limited.SealedDeckBuilder;
import forge.item.PaperCard;
import forge.model.FModel;
import forge.util.MyRandom;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Builds a 40-card sealed deck from a booster pool using one of four weighted methods.
 *
 * <p>Method weights: 1=40%, 2=30%, 3=20%, 4=10%.
 * <ul>
 *   <li>Method 1: Standard Forge sealed deck builder</li>
 *   <li>Method 2: Forge builder + 3 spell swaps + land rebalance</li>
 *   <li>Method 3: Forge builder + 8 spell swaps + land rebalance</li>
 *   <li>Method 4: 23 random spells + land rebalance</li>
 * </ul>
 *
 * <p>"Nonland card" means a spell (creature, instant, sorcery, enchantment, artifact,
 * planeswalker) — cards whose type line does not include "Land". Non-basic lands (dual
 * lands, utility lands) are treated as lands throughout.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class DeckBuilder {

    public static final String METHOD_FORGE_BEST = "forge-best";
    public static final String METHOD_FORGE_3SUB = "forge-3sub";
    public static final String METHOD_FORGE_8SUB = "forge-8sub";
    public static final String METHOD_RANDOM = "random";

    private static final double[] METHOD_THRESHOLDS = {0.4, 0.7, 0.9, 1.0};
    private final Random random;

    /** A constructed deck together with the method tag that produced it. */
    public record BuiltDeck(Deck deck, String method) {}

    public DeckBuilder() {
        this.random = MyRandom.getRandom();
    }

    DeckBuilder(Random random) {
        this.random = random;
    }

    /**
     * Build a deck from the given pool using a randomly selected weighted method,
     * returning the deck alongside the method tag so callers can record which
     * strategy was used.
     *
     * @param pool booster pool cards (no basic lands)
     * @return the 40-card deck and its method tag
     */
    public BuiltDeck buildDeck(List<PaperCard> pool) {
        int method = selectMethod();
        return switch (method) {
            case 1 -> new BuiltDeck(buildStandard(pool), METHOD_FORGE_BEST);
            case 2 -> new BuiltDeck(buildWithSwaps(pool, 3), METHOD_FORGE_3SUB);
            case 3 -> new BuiltDeck(buildWithSwaps(pool, 8), METHOD_FORGE_8SUB);
            case 4 -> new BuiltDeck(buildRandom(pool), METHOD_RANDOM);
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

        // 2. Partition standard deck into spells and non-basic lands (discard basic lands)
        List<PaperCard> spellsInDeck = new ArrayList<>();
        List<PaperCard> nonbasicLandsInDeck = new ArrayList<>();
        for (PaperCard card : standardDeck.getMain().toFlatList()) {
            if (card.getRules().getMainPart().getType().isBasicLand()) {
                // discard — will rebalance basic lands after swaps
            } else if (card.getRules().getMainPart().getType().isLand()) {
                nonbasicLandsInDeck.add(card);
            } else {
                spellsInDeck.add(card);
            }
        }

        // 3. Remaining pool = original pool minus all selected deck cards
        List<PaperCard> remainingPool = new ArrayList<>(pool);
        for (PaperCard deckCard : standardDeck.getMain().toFlatList()) {
            remainingPool.remove(deckCard);
        }
        // Split remaining pool by type (pool has no basic lands)
        List<PaperCard> spellsRemaining = new ArrayList<>();
        List<PaperCard> nonbasicLandsRemaining = new ArrayList<>();
        for (PaperCard card : remainingPool) {
            if (card.getRules().getMainPart().getType().isLand()) {
                nonbasicLandsRemaining.add(card);
            } else {
                spellsRemaining.add(card);
            }
        }

        // 4. Type-matched swaps: spells swap with spells, non-basic lands with non-basic lands.
        // Each swap picks randomly from all deck non-basics, then replaces from the matching pool.
        List<PaperCard> chosenSpells = new ArrayList<>(spellsInDeck);
        List<PaperCard> chosenNonbasics = new ArrayList<>(nonbasicLandsInDeck);
        int swapsLeft = swapCount;
        while (swapsLeft > 0) {
            int totalSwappable = chosenSpells.size() + chosenNonbasics.size();
            if (totalSwappable == 0) break;
            int pick = random.nextInt(totalSwappable);
            if (pick < chosenSpells.size()) {
                if (spellsRemaining.isEmpty()) { swapsLeft--; continue; }
                chosenSpells.remove(pick);
                chosenSpells.add(spellsRemaining.remove(random.nextInt(spellsRemaining.size())));
            } else {
                if (nonbasicLandsRemaining.isEmpty()) { swapsLeft--; continue; }
                chosenNonbasics.remove(pick - chosenSpells.size());
                chosenNonbasics.add(nonbasicLandsRemaining.remove(random.nextInt(nonbasicLandsRemaining.size())));
            }
            swapsLeft--;
        }

        // 5. Rebalance basic lands based on spell pip counts; keep non-basic lands
        return rebalanceLands(chosenSpells, chosenNonbasics);
    }

    Deck buildRandom(List<PaperCard> pool) {
        // Pick 23 random spells from pool; non-basic lands in the pool are excluded
        List<PaperCard> spellPool = new ArrayList<>();
        for (PaperCard card : pool) {
            if (!card.getRules().getMainPart().getType().isLand()) {
                spellPool.add(card);
            }
        }

        int pickCount = Math.min(23, spellPool.size());
        List<PaperCard> chosen = new ArrayList<>(pickCount);
        List<PaperCard> mutablePool = new ArrayList<>(spellPool);
        for (int i = 0; i < pickCount; i++) {
            int idx = random.nextInt(mutablePool.size());
            chosen.add(mutablePool.remove(idx));
        }

        return rebalanceLands(chosen, List.of());
    }

    /**
     * Build a 40-card deck from the given spells and non-basic lands by adding basic
     * lands proportional to the WUBRG mana pip counts of the spells.
     *
     * <p>Note: {@code LimitedDeckBuilder.addLands()} is private and cannot be called
     * externally. Passing cards to {@code SealedDeckBuilder} as a pool re-selects a
     * subset rather than treating all cards as already chosen, producing severely
     * land-heavy decks. This method reimplements the pip-proportional allocation directly.
     *
     * @param spells        chosen nonland spells (creatures, instants, etc.)
     * @param nonbasicLands non-basic lands to retain unchanged from the standard deck
     * @return 40-card Deck
     */
    Deck rebalanceLands(List<PaperCard> spells, List<PaperCard> nonbasicLands) {
        int basicLandsNeeded = 40 - spells.size() - nonbasicLands.size();

        // Count WUBRG mana pips across spells to determine basic land proportions.
        // Non-basic lands are intentionally excluded — they have no mana cost.
        int[] pips = new int[5]; // W=0, U=1, B=2, R=3, G=4
        for (PaperCard card : spells) {
            ManaCost mc = card.getRules().getMainPart().getManaCost();
            for (ManaCostShard shard : mc) {
                for (int i = 0; i < MagicColor.WUBRG.length; i++) {
                    if (shard.canBePaidWithManaOfColor(MagicColor.WUBRG[i])) {
                        pips[i]++;
                    }
                }
            }
        }

        int totalPips = Arrays.stream(pips).sum();
        if (totalPips == 0) {
            // Colorless deck: distribute equally across all five types
            Arrays.fill(pips, 1);
            totalPips = 5;
        }

        // Source basic land cards from a set in the spell pool, fallback to GRN
        String setCode = findBasicLandSet(spells);

        // Distribute basic land slots proportionally (greedy, WUBRG order;
        // last required color absorbs any rounding remainder)
        List<PaperCard> basicLands = new ArrayList<>(basicLandsNeeded);
        int remaining = basicLandsNeeded;
        int remainingPips = totalPips;

        for (int i = 0; i < 5; i++) {
            if (pips[i] == 0 || remaining == 0) continue;
            int count = (remainingPips == pips[i])
                    ? remaining
                    : Math.round((float) remaining * pips[i] / remainingPips);
            count = Math.max(0, Math.min(count, remaining));

            String landName = MagicColor.Constant.BASIC_LANDS.get(i);
            PaperCard landCard = FModel.getMagicDb().getCommonCards().getCard(landName, setCode);
            for (int j = 0; j < count && landCard != null; j++) {
                basicLands.add(landCard);
            }
            remaining -= count;
            remainingPips -= pips[i];
        }

        // Assemble: spells + non-basic lands + basic lands
        Deck deck = new Deck();
        CardPool main = deck.getOrCreate(DeckSection.Main);
        for (PaperCard card : spells) main.add(card);
        for (PaperCard land : nonbasicLands) main.add(land);
        for (PaperCard land : basicLands) main.add(land);
        return deck;
    }

    private String findBasicLandSet(List<PaperCard> cards) {
        for (PaperCard card : cards) {
            CardEdition ed = StaticData.instance().getEditions().get(card.getEdition());
            if (ed != null && CardEdition.Predicates.hasBasicLands.test(ed)) {
                return card.getEdition();
            }
        }
        return "GRN"; // fallback: Guilds of Ravnica always has basic lands
    }
}
