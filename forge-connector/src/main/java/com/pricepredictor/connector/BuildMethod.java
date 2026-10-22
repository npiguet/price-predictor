package com.pricepredictor.connector;

/**
 * The four ways {@link DeckBuilder#buildDeck} builds a Forge-side deck, rolled
 * per deck with weights 4:3:2:1.
 *
 * <p>{@link #label()} is a wire value: it is written to the
 * {@code method_A} / {@code method_B} fields of {@code match-outcomes.txt}
 * and {@code cards-played.txt}, and the sealed pipeline groups rows by it.
 * Never rename one.
 */
public enum BuildMethod {
    /** Forge's sealed deck builder as it stands. */
    FORGE_BEST("forge-best", 4, 0),
    /** Forge's deck with 3 type-matched swaps from the rest of the pool. */
    FORGE_3SUB("forge-3sub", 3, 3),
    /** Forge's deck with 8 type-matched swaps from the rest of the pool. */
    FORGE_8SUB("forge-8sub", 2, 8),
    /** 23 random spells from the pool, lands rebalanced to their pips. */
    RANDOM("random", 1, 0);

    /**
     * Sum of the weights: the denominator of the roll here, and the weight the
     * Forge methods carry together against a side-B decks file's in
     * {@link MatchGenerator#rollIsFileSample()}.
     */
    public static final int TOTAL_WEIGHT = totalWeight();

    private final String label;
    private final int weight;
    private final int swaps;

    BuildMethod(String label, int weight, int swaps) {
        this.label = label;
        this.weight = weight;
        this.swaps = swaps;
    }

    public String label() {
        return label;
    }

    int weight() {
        return weight;
    }

    /** How many spells or nonbasic lands are swapped out of Forge's deck. */
    int swaps() {
        return swaps;
    }

    /** The method a uniform draw {@code roll} in {@code [0, 1)} selects. */
    static BuildMethod forRoll(double roll) {
        int cumulative = 0;
        for (BuildMethod method : values()) {
            cumulative += method.weight;
            if (roll < (double) cumulative / TOTAL_WEIGHT) {
                return method;
            }
        }
        return RANDOM;
    }

    private static int totalWeight() {
        int total = 0;
        for (BuildMethod method : values()) {
            total += method.weight;
        }
        return total;
    }
}
