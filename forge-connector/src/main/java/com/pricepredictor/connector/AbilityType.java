package com.pricepredictor.connector;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * Categories for ability lines in converted card output.
 */
@Getter
@RequiredArgsConstructor
public enum AbilityType {
    ACTIVATED("activated", true),
    TRIGGERED("triggered", false),
    STATIC("static", false),
    REPLACEMENT("replacement", false),
    CHAPTER("chapter", false),
    LEVEL("level", true),
    ALTERNATE_COST("alternate cost", false),
    COST_REDUCTION("cost reduction", false),
    ADDITIONAL_COST("additional cost", false),
    SPELL("spell", true),
    PLANESWALKER("planeswalker", true),
    OPTION("option", true),
    TEXT("text", false);

    private final String outputPrefix;
    private final boolean actionable;
}
