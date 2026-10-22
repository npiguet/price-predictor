package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Weighted categorical over agent labels, parsed from {@code label:weight,...}. */
final class AgentMix {
    private final String[] labels;
    private final int[] cumulative;
    private final int total;

    private AgentMix(String[] labels, int[] cumulative, int total) {
        this.labels = labels;
        this.cumulative = cumulative;
        this.total = total;
    }

    static AgentMix parse(String spec) {
        String[] tokens = spec.split(",");
        List<String> labels = new ArrayList<>();
        List<Integer> weights = new ArrayList<>();
        for (String token : tokens) {
            token = token.trim();
            if (token.isEmpty()) {
                continue;
            }
            int colon = token.indexOf(':');
            if (colon <= 0 || colon != token.lastIndexOf(':')) {
                throw new IllegalArgumentException(
                        "agent-mix entry must be 'label:weight', got: " + token);
            }
            String label = token.substring(0, colon).trim();
            int weight = Integer.parseInt(token.substring(colon + 1).trim());
            if (label.isEmpty() || weight < 1) {
                throw new IllegalArgumentException(
                        "agent-mix entry must have a non-empty label and weight >= 1: " + token);
            }
            labels.add(label);
            weights.add(weight);
        }
        if (labels.isEmpty()) {
            throw new IllegalArgumentException("agent-mix is empty: " + spec);
        }
        int[] cumulative = new int[labels.size()];
        int running = 0;
        for (int i = 0; i < weights.size(); i++) {
            running += weights.get(i);
            cumulative[i] = running;
        }
        return new AgentMix(labels.toArray(new String[0]), cumulative, running);
    }

    String sample(Random random) {
        int roll = random.nextInt(total);
        for (int i = 0; i < cumulative.length; i++) {
            if (roll < cumulative[i]) {
                return labels[i];
            }
        }
        return labels[labels.length - 1];  // unreachable
    }
}
