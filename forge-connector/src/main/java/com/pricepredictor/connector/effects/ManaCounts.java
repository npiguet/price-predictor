package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.card.MagicColor;
import forge.card.mana.ManaAtom;
import forge.game.mana.Mana;

import java.util.ArrayList;
import java.util.List;

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

    /** Where colourless is counted: last. */
    private static final int COLORLESS_INDEX = SYMBOLS.length - 1;

    /** No mana of any symbol. */
    static final ManaCounts NONE = new ManaCounts(new int[SYMBOLS.length]);

    private final int[] counts;

    private ManaCounts(int[] counts) {
        this.counts = counts;
    }

    /**
     * One mana per unit of {@code units}, under its colour: what a floating
     * pool holds. A null unit counts under none.
     */
    static ManaCounts units(Iterable<Mana> units) {
        List<Byte> symbols = new ArrayList<>();
        for (Mana unit : units) {
            if (unit != null) {
                symbols.add(unit.getColor());
            }
        }
        return tally(symbols);
    }

    /**
     * One mana per entry of {@code symbols}; an entry that is not one of the
     * six symbols — a multicolour mask, say — counts under none. Colourless
     * counts under {@code C} under either byte: Forge keys a colourless
     * {@link Mana} unit as {@code ManaAtom.COLORLESS}, and a colour list
     * names it {@code MagicColor.COLORLESS}.
     */
    static ManaCounts tally(Iterable<Byte> symbols) {
        int[] counts = new int[SYMBOLS.length];
        for (byte symbol : symbols) {
            int index = indexOf(symbol);
            if (index >= 0) {
                counts[index]++;
            }
        }
        return new ManaCounts(counts);
    }

    /** Where {@code symbol} is counted, or -1 when it is none of the six. */
    private static int indexOf(byte symbol) {
        if (symbol == (byte) ManaAtom.COLORLESS) {
            return COLORLESS_INDEX;
        }
        for (int i = 0; i < SYMBOLS.length; i++) {
            if (SYMBOLS[i] == symbol) {
                return i;
            }
        }
        return -1;
    }

    /** Whether no mana of any symbol is counted. */
    boolean isZero() {
        for (int count : counts) {
            if (count != 0) {
                return false;
            }
        }
        return true;
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
