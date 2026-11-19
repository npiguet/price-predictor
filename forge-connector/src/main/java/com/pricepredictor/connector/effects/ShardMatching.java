package com.pricepredictor.connector.effects;

import forge.card.mana.ManaCostShard;

import java.util.Arrays;
import java.util.List;

/**
 * A maximum matching of shards to the units that pay them, one unit per
 * shard (Kuhn's augmenting paths): which shards a set of mana, or of
 * permanents paying as mana would, can pay at once.
 */
final class ShardMatching {

    private final List<ManaUnit> units;
    private final List<ManaCostShard> slots;
    private final int[] unitOf;
    private final int[] slotOf;

    private ShardMatching(List<ManaUnit> units, List<ManaCostShard> slots) {
        this.units = units;
        this.slots = slots;
        this.unitOf = new int[slots.size()];
        this.slotOf = new int[units.size()];
        Arrays.fill(unitOf, -1);
        Arrays.fill(slotOf, -1);
    }

    /** For each of {@code slots}, the index of the unit paying it, or -1 where none is left to. */
    static int[] match(List<ManaUnit> units, List<ManaCostShard> slots) {
        ShardMatching matching = new ShardMatching(units, slots);
        for (int slot = 0; slot < slots.size(); slot++) {
            matching.augment(slot, new boolean[units.size()]);
        }
        return matching.unitOf;
    }

    /** How many of {@code slots} the units pay at once. */
    static int size(List<ManaUnit> units, List<ManaCostShard> slots) {
        return (int) Arrays.stream(match(units, slots)).filter(unit -> unit >= 0).count();
    }

    private boolean augment(int slot, boolean[] seen) {
        for (int unit = 0; unit < units.size(); unit++) {
            if (seen[unit] || !units.get(unit).pays(slots.get(slot))) {
                continue;
            }
            seen[unit] = true;
            if (slotOf[unit] < 0 || augment(slotOf[unit], seen)) {
                slotOf[unit] = slot;
                unitOf[slot] = unit;
                return true;
            }
        }
        return false;
    }
}
