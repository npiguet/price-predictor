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
        List<Entry> entries = new ArrayList<>();
        for (String token : spec.split(",")) {
            token = token.trim();
            if (!token.isEmpty()) {
                entries.add(Entry.parse(token));
            }
        }
        if (entries.isEmpty()) {
            throw new IllegalArgumentException("agent-mix is empty: " + spec);
        }
        String[] labels = new String[entries.size()];
        int[] cumulative = new int[entries.size()];
        int running = 0;
        for (int i = 0; i < entries.size(); i++) {
            running += entries.get(i).weight();
            labels[i] = entries.get(i).label();
            cumulative[i] = running;
        }
        return new AgentMix(labels, cumulative, running);
    }

    /** One {@code label:weight} item of the spec. */
    private record Entry(String label, int weight) {
        static Entry parse(String token) {
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
            return new Entry(label, weight);
        }
    }

    /** The labels the mix draws from, in spec order. */
    List<String> labels() {
        return List.of(labels);
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
