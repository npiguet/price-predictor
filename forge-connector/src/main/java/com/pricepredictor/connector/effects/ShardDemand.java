package com.pricepredictor.connector.effects;

import forge.card.mana.ManaAtom;
import forge.card.mana.ManaCostShard;
import forge.game.mana.ManaCostBeingPaid;

import java.util.ArrayList;
import java.util.List;

/**
 * The shards of a cost still to pay, sorted by how they may be paid. X is
 * left out: nothing is announced at a priority stop.
 *
 * @param colored    shards paid by mana of a colour (hybrid and snow ones included)
 * @param phyrexian  shards that two life may pay instead
 * @param twoGeneric {2/C} shards that two generic mana may pay instead
 * @param generic    the generic mana
 */
record ShardDemand(
        List<ManaCostShard> colored, List<ManaCostShard> phyrexian,
        List<ManaCostShard> twoGeneric, int generic) {

    static ShardDemand of(ManaCostBeingPaid cost) {
        List<ManaCostShard> colored = new ArrayList<>();
        List<ManaCostShard> phyrexian = new ArrayList<>();
        List<ManaCostShard> twoGeneric = new ArrayList<>();
        int generic = 0;
        for (ManaCostShard shard : cost.getUnpaidShards()) {
            if (shard == ManaCostShard.GENERIC) {
                generic++;
            } else if (!shard.isOfKind(ManaAtom.IS_X)) {
                kindOf(shard, colored, phyrexian, twoGeneric).add(shard);
            }
        }
        return new ShardDemand(colored, phyrexian, twoGeneric, generic);
    }

    private static List<ManaCostShard> kindOf(ManaCostShard shard, List<ManaCostShard> colored,
                                              List<ManaCostShard> phyrexian, List<ManaCostShard> twoGeneric) {
        if (shard.isPhyrexian()) {
            return phyrexian;
        }
        return shard.isOr2Generic() ? twoGeneric : colored;
    }

    boolean isEmpty() {
        return colored.isEmpty() && phyrexian.isEmpty() && twoGeneric.isEmpty() && generic == 0;
    }

    /** Every shard but the generic mana, in the order a matching tries them. */
    List<ManaCostShard> nonGeneric() {
        return slots(0, 0);
    }

    /**
     * The shards mana must pay when the first {@code withLife} phyrexian
     * shards are paid with life and the first {@code asGeneric} {2/C}
     * shards with two generic mana.
     */
    List<ManaCostShard> slots(int withLife, int asGeneric) {
        List<ManaCostShard> slots = new ArrayList<>(colored);
        slots.addAll(phyrexian.subList(withLife, phyrexian.size()));
        slots.addAll(twoGeneric.subList(asGeneric, twoGeneric.size()));
        return slots;
    }
}
