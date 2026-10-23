package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * The mana records one game keeps, {@code --mana-cap} per mana ability (FR-037).
 *
 * <p>A Mountain taps a dozen times a game for the same R, and the repeats
 * observe a board that barely moved, so the corpus keeps a few of them per
 * game. <b>Which</b> ones is the question this answers, and taking the first
 * would be the wrong answer: a land's first tap is almost always turn one
 * against an empty board, so every mana record would describe the same early
 * game and the model would see a board-independent effect on a board that
 * never varied.
 *
 * <p>Reservoir sampling (Algorithm R) instead: the first {@code cap} fill the
 * reservoir, and the {@code n}th after that replaces a uniformly chosen one
 * with probability {@code cap/n}. Every activation in the game ends up equally
 * likely to survive, without knowing in advance how many there will be. The
 * records are therefore held until the game ends.
 *
 * <p>Keyed by the ability's text and the mana it produced, so every Mountain
 * shares one reservoir and a dual land making G and the same land making U
 * each get their own.
 */
final class ManaReservoir {

    private final int cap;
    /** The collector's sampler, shared so one seed decides every draw of a game. */
    private final Random sampler;
    /** Activations seen per key, the reservoir's {@code n}. */
    private final Map<String, Integer> seen = new HashMap<>();
    /** The sampled records per key, in the order the keys first appeared. */
    private final Map<String, List<EffectRecord>> held = new LinkedHashMap<>();

    ManaReservoir(int cap, Random sampler) {
        this.cap = cap;
        this.sampler = sampler;
    }

    /**
     * Which slot this activation takes, or -1 to skip it.
     *
     * <p>Asked before the record is built, so the expensive part — the state
     * snapshot — is paid a logarithmic number of times rather than once per
     * activation.
     */
    int slot(String key) {
        if (cap <= 0) {
            return -1;
        }
        int n = seen.merge(key, 1, Integer::sum);
        if (n <= cap) {
            return n - 1;
        }
        int candidate = sampler.nextInt(n);
        return candidate < cap ? candidate : -1;
    }

    /** Put a record in the slot {@link #slot} gave it. */
    void hold(String key, int slot, EffectRecord record) {
        List<EffectRecord> records = held.computeIfAbsent(key, k -> new ArrayList<>());
        if (slot == records.size()) {
            records.add(record);
        } else {
            records.set(slot, record);
        }
    }

    /** Every surviving record, and an empty reservoir for the next game. */
    List<EffectRecord> drain() {
        List<EffectRecord> survivors = new ArrayList<>();
        for (List<EffectRecord> records : held.values()) {
            for (EffectRecord record : records) {
                if (record != null) {
                    survivors.add(record);
                }
            }
        }
        held.clear();
        seen.clear();
        return survivors;
    }
}
