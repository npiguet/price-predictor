package com.pricepredictor.connector;

import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/** What the draft worker itself knows about an agent label. */
final class DraftAgent {

    /** The pure Forge AI label: no pick is replaced. */
    private static final String FORGE_FULL = "forge-full";

    /**
     * Share of picks replaced by a uniform-random legal card, by label. With
     * {@link #FORGE_FULL} these are the labels the worker drafts itself; any
     * other label must be an external agent.
     */
    private static final Map<String, Double> RANDOM_OVERRIDE_FRACTION =
            Map.of("forge-r30", 0.30, "forge-r100", 1.0);

    private DraftAgent() {
    }

    /**
     * Share of this label's picks replaced by a uniform-random legal card.
     *
     * @throws IllegalArgumentException for a label the worker does not draft:
     *         treating it as pure Forge AI would draft at full strength under
     *         the wrong label
     */
    static double randomOverrideFraction(String label) {
        if (FORGE_FULL.equals(label)) {
            return 0.0;
        }
        Double fraction = RANDOM_OVERRIDE_FRACTION.get(label);
        if (fraction == null) {
            throw new IllegalArgumentException("unknown draft agent label: " + label);
        }
        return fraction;
    }

    /**
     * Refuse a label that is neither drafted by the worker nor external, so a
     * misspelt label fails the run at start rather than mid-draft.
     *
     * @throws IllegalArgumentException naming each unknown label
     */
    static void requireKnown(Collection<String> labels, Set<String> external) {
        Set<String> unknown = new TreeSet<>();
        for (String label : labels) {
            boolean known = FORGE_FULL.equals(label)
                    || RANDOM_OVERRIDE_FRACTION.containsKey(label)
                    || external.contains(label);
            if (!known) {
                unknown.add(label);
            }
        }
        if (!unknown.isEmpty()) {
            throw new IllegalArgumentException("unknown draft agent label(s) " + unknown
                    + ": each is " + FORGE_FULL + ", " + new TreeSet<>(RANDOM_OVERRIDE_FRACTION.keySet())
                    + " or listed in -D" + DraftWorkerMain.EXTERNAL_AGENTS);
        }
    }
}
