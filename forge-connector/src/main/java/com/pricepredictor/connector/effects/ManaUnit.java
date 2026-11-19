package com.pricepredictor.connector.effects;

import forge.card.mana.ManaAtom;
import forge.card.mana.ManaCostShard;

/**
 * One mana a payment could spend, or one permanent paying as mana would:
 * the colours it may be, whether it is snow, and whether it is one of a
 * waterbend's tappers, of which only the waterbend's amount pay
 * ({@link KeywordPayers}).
 *
 * @param colors  a mask of {@code ManaAtom} colours, colourless included;
 *                none for a unit that pays generic mana only
 * @param snow    whether it came from a snow source
 * @param limited whether it is a waterbend tapper
 */
record ManaUnit(byte colors, boolean snow, boolean limited) {

    /** A unit that pays only generic mana: a delved card or an improvising artifact. */
    static ManaUnit generic() {
        return new ManaUnit((byte) 0, false, false);
    }

    /** Whether this unit can pay {@code shard}. */
    boolean pays(ManaCostShard shard) {
        if (shard.isSnow() && !snow) {
            return false;
        }
        if (shard == ManaCostShard.GENERIC || shard == ManaCostShard.S) {
            return true;
        }
        for (byte color : ManaAtom.MANATYPES) {
            if ((colors & color) != 0 && shard.canBePaidWithManaOfColor(color)) {
                return true;
            }
        }
        return false;
    }
}
