package com.pricepredictor.connector.effects;

import java.util.List;

/**
 * A {@link CollectionCaps} for a test: the defaults, with only what the test
 * varies named. A positional constructor call leaves the reader counting
 * arguments to learn that {@code 2, 0} means two interventions and no probes.
 */
public final class TestCaps {

    private int manaCap;
    private double playabilityRate;
    private int interventions;
    private int probes;
    private List<String> probeKeywords;
    private double randomSeatShare;
    private double randomSeatProbability;

    private TestCaps(CollectionCaps from) {
        manaCap = from.manaCap();
        playabilityRate = from.playabilityRate();
        interventions = from.interventionsPerGame();
        probes = from.probesPerGame();
        probeKeywords = from.probeKeywords();
        randomSeatShare = from.randomSeatShare();
        randomSeatProbability = from.randomSeatProbability();
    }

    /** {@link CollectionCaps#defaults()}, to be varied. */
    public static TestCaps caps() {
        return new TestCaps(CollectionCaps.defaults());
    }

    public TestCaps manaCap(int cap) {
        manaCap = cap;
        return this;
    }

    public TestCaps playabilityRate(double rate) {
        playabilityRate = rate;
        return this;
    }

    public TestCaps interventions(int perGame) {
        interventions = perGame;
        return this;
    }

    public TestCaps probes(int perGame) {
        probes = perGame;
        return this;
    }

    public TestCaps probeKeywords(String... keywords) {
        probeKeywords = List.of(keywords);
        return this;
    }

    public TestCaps randomSeat(double share, double probability) {
        randomSeatShare = share;
        randomSeatProbability = probability;
        return this;
    }

    public CollectionCaps build() {
        return new CollectionCaps(manaCap, playabilityRate, interventions, probes,
                probeKeywords, randomSeatShare, randomSeatProbability);
    }
}
