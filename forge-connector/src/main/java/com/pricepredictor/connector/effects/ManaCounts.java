package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.card.MagicColor;

/**
 * An amount of mana per symbol, rendered in one fixed order: W, U, B, R, G,
 * then C.
 *
 * <p>A player's floating pool and untapped production in the snapshot and
 * what an activation paid are all counted this way, and all render their
 * keys in this order whatever order the mana was made or spent in: dedup
 * hashes the rendered bytes, so two payments of one cost that differed only
 * in key order would be two different records.
 */
final class ManaCounts {

    /** The symbols counted, in rendering order: the five colours, then colourless. */
    private static final byte[] SYMBOLS = MagicColor.WUBRGC;

    /** No mana of any symbol. */
    static final ManaCounts NONE = new ManaCounts(new int[SYMBOLS.length]);

    /** One symbol's amount, read off something that is asked once per symbol. */
    @FunctionalInterface
    interface AmountOf {
        int of(byte symbol);
    }

    private final int[] counts;

    private ManaCounts(int[] counts) {
        this.counts = counts;
    }

    /** Each symbol's amount, asked of {@code amount} once per symbol in rendering order. */
    static ManaCounts of(AmountOf amount) {
        int[] counts = new int[SYMBOLS.length];
        for (int i = 0; i < SYMBOLS.length; i++) {
            counts[i] = amount.of(SYMBOLS[i]);
        }
        return new ManaCounts(counts);
    }

    /**
     * One mana per entry of {@code symbols}; an entry that is not one of the
     * six symbols — a multicolour mask, say — counts under none.
     */
    static ManaCounts tally(Iterable<Byte> symbols) {
        int[] counts = new int[SYMBOLS.length];
        for (byte symbol : symbols) {
            for (int i = 0; i < SYMBOLS.length; i++) {
                if (SYMBOLS[i] == symbol) {
                    counts[i]++;
                    break;
                }
            }
        }
        return new ManaCounts(counts);
    }

    /** The letter a symbol is spelled with, which is also its JSON key. */
    static String letter(byte symbol) {
        return MagicColor.toShortString(symbol);
    }

    /** Every symbol and its amount, zeros included: {@code {"W":0,…,"C":0}}. */
    String toJson() {
        return render(true);
    }

    /** Only the symbols with a positive amount: {@code {}} when there are none. */
    String toJsonOmittingZeros() {
        return render(false);
    }

    private String render(boolean zeros) {
        Json.ObjectWriter json = Json.object();
        for (int i = 0; i < SYMBOLS.length; i++) {
            if (zeros || counts[i] > 0) {
                json.value(letter(SYMBOLS[i]), counts[i]);
            }
        }
        return json.render();
    }
}
