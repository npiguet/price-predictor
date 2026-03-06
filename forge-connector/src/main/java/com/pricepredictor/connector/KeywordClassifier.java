package com.pricepredictor.connector;

import lombok.experimental.UtilityClass;

import java.util.Set;

/**
 * Classifies MTG keywords as passive (automatic) or activatable (player pays a cost).
 */
@UtilityClass
public class KeywordClassifier {

    private final Set<String> ACTIVATABLE = Set.of(
            "kicker", "cycling", "equip", "ninjutsu", "evoke", "emerge", "channel",
            "flashback", "morph", "megamorph", "bestow", "dash", "unearth", "replicate",
            "madness", "foretell", "suspend", "mutate", "level up", "escape", "encore",
            "overload", "retrace", "disturb", "boast", "craft", "prototype", "prowl",
            "spectacle", "surge", "entwine", "buyback", "crew", "reconfigure", "adapt",
            "monstrosity", "scavenge", "embalm", "eternalize", "outlast", "transfigure",
            "transmute", "forecast", "fortify", "reinforce", "bloodrush"
    );

    /**
     * Returns true if the keyword is activatable (player pays a cost to use it).
     */
    public boolean isActivatable(String keywordName) {
        return ACTIVATABLE.contains(keywordName.toLowerCase());
    }
}
