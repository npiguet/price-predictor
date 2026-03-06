package com.pricepredictor.connector;

import java.util.Set;

/**
 * Classifies MTG keywords as passive (automatic) or activatable (player pays a cost).
 */
public final class KeywordClassifier {

    private static final Set<String> ACTIVATABLE = Set.of(
            "kicker", "cycling", "equip", "ninjutsu", "evoke", "emerge", "channel",
            "flashback", "morph", "megamorph", "bestow", "dash", "unearth", "replicate",
            "madness", "foretell", "suspend", "mutate", "level up", "escape", "encore",
            "overload", "retrace", "disturb", "boast", "craft", "prototype", "prowl",
            "spectacle", "surge", "entwine", "buyback", "crew", "reconfigure", "adapt",
            "monstrosity", "scavenge", "embalm", "eternalize", "outlast", "transfigure",
            "transmute", "forecast", "fortify", "reinforce", "bloodrush"
    );

    private KeywordClassifier() {}

    /**
     * Returns true if the keyword is activatable (player pays a cost to use it).
     */
    public static boolean isActivatable(String keywordName) {
        return ACTIVATABLE.contains(keywordName.toLowerCase());
    }
}
